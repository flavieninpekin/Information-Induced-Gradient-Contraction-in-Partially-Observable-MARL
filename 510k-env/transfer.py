"""
Transfer evaluation: run a SINGLE-trained policy in all 3 game modes,
collecting trajectories with feature vectors for IRL.
"""
import os
import random
import json
import time
from typing import List, Optional

import numpy as np
from sb3_contrib import MaskablePPO

from env.game import Game, GameMode
from env.obs_utils import obs_for_player, action_mask_for_player, execute_action
from env.features import extract_features
from env.features import FEATURE_DIM, FEATURE_NAMES
from env.scorer import Scorer


def run_episode(model: MaskablePPO, mode: str, seed: int,
                deterministic: bool = True) -> dict:
    """Run one episode with the trained model controlling ALL 4 players.

    Returns dict with:
      - trajectories: list of (player, state_features, action, reward)
      - rewards: per-player total reward
      - finish_order
      - red_a_team (if dynamic)
      - episode_length
    """
    random.seed(seed)
    np.random.seed(seed)

    game = Game(mode=GameMode(mode), num_players=4)  # 4P mode only

    trajectories = {i: [] for i in range(4)}

    while not game.is_over:
        pid = game.current_player
        obs = obs_for_player(game, pid)
        mask = action_mask_for_player(game, pid)
        feats = extract_features(game, pid)

        action, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)

        success = execute_action(game, pid, int(action))
        if not success:
            patterns = game.get_valid_actions(pid)
            if patterns:
                game.play_cards(pid, random.choice(patterns).cards)
            elif game.can_pass(pid):
                game.pass_turn(pid)

        next_feats = extract_features(game, pid)
        trajectories[pid].append({
            'features': feats.tolist(),
            'action': int(action),
            'next_features': next_feats.tolist(),
        })

    scorer = Scorer(game)
    rewards = scorer.compute_rewards()

    for pid in range(4):
        for entry in trajectories[pid]:
            entry['reward'] = rewards[pid]

    result = {
        'mode': mode,
        'seed': seed,
        'trajectories': trajectories,
        'rewards': rewards,
        'finish_order': game.finish_order,
        'red_a_team': list(game.red_a_team) if game.red_a_team else None,
        'episode_length': game.round_count,
        'player_scores': game.player_510k_scores,
    }
    return result


def transfer_evaluate(model_path: str, mode: str,
                      n_episodes: int = 1000,
                      seed: int = 42,
                      deterministic: bool = True,
                      output_dir: str = 'transfer_data') -> dict:
    """Evaluate a model under a specific game mode, collecting trajectories."""
    os.makedirs(output_dir, exist_ok=True)

    model = MaskablePPO.load(model_path)

    all_trajectories = []
    episode_rewards = []
    stats = {'wins': 0, 'total': 0, 'finish_counts': [0]*4, 'player_scores': []}

    for ep in range(n_episodes):
        result = run_episode(model, mode, seed + ep, deterministic=deterministic)
        all_trajectories.append(result)
        episode_rewards.append(result['rewards'])

        stats['total'] += 1
        if result['finish_order']:
            stats['finish_counts'][result['finish_order'][0]] += 1
        stats['player_scores'].append(result['player_scores'])

        if (ep + 1) % 200 == 0:
            print(f'  [{mode}] {ep+1}/{n_episodes} episodes')

    # Aggregate feature expectations for MaxEnt IRL
    all_features = []
    for ep_data in all_trajectories:
        if isinstance(ep_data['trajectories'], dict):
            for traj in ep_data['trajectories'].values():
                for entry in traj:
                    all_features.append(entry['features'])

    feature_expectations = np.mean(all_features, axis=0) if all_features else np.zeros(FEATURE_DIM)

    # Per-player average features
    per_player_fe = {}
    for pid in range(4):
        feats = []
        for ep_data in all_trajectories:
            if isinstance(ep_data['trajectories'], dict):
                for entry in ep_data['trajectories'].get(pid, []):
                    feats.append(entry['features'])
        per_player_fe[pid] = np.mean(feats, axis=0) if feats else np.zeros(FEATURE_DIM)

        rew_array = np.array([list(r.values()) for r in episode_rewards]) if episode_rewards else np.zeros((0, 4))
        avg_rewards = np.mean(rew_array, axis=0) if rew_array.size else np.zeros(4)
        std_rewards = np.std(rew_array, axis=0) if rew_array.size else np.zeros(4)

    summary = {
        'mode': mode,
        'model_path': model_path,
        'n_episodes': n_episodes,
        'avg_rewards_per_player': avg_rewards.tolist(),
        'std_rewards_per_player': std_rewards.tolist(),
        'feature_expectations': feature_expectations.tolist(),
        'per_player_feature_expectations': {str(k): v.tolist() for k, v in per_player_fe.items()},
        'first_finish_counts': stats['finish_counts'],
        'avg_510k_scores': np.mean(stats['player_scores'], axis=0).tolist(),
    }

    # Save summary
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    save_path = os.path.join(output_dir, f'{model_name}_{mode}_summary.json')
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Save raw trajectories (compressed)
    import gzip, pickle
    traj_path = os.path.join(output_dir, f'{model_name}_{mode}_trajectories.pkl.gz')
    with gzip.open(traj_path, 'wb') as f:
        pickle.dump(all_trajectories, f)

    print(f'  [{mode}] Done. Avg rewards: {avg_rewards}')
    print(f'  [{mode}] Feature expectations: {feature_expectations}')
    print(f'  [{mode}] Saved to {save_path}')

    return summary


def batch_transfer_evaluate(model_dir: str = 'models',
                            checkpoint_steps: Optional[List[int]] = None,
                            output_dir: str = 'transfer_data',
                            n_episodes: int = 1000,
                            modes: Optional[List[str]] = None):
    """Run transfer evaluation for multiple checkpoints across all modes."""
    if modes is None:
        modes = ['single', 'static', 'dynamic']

    if checkpoint_steps is None:
        # Use final model by default
        checkpoint_steps = ['final']

    all_summaries = {}

    for step_label in checkpoint_steps:
        model_path = os.path.join(model_dir, f'510k_single_{step_label}.zip')
        if step_label == 'final':
            model_path = os.path.join(model_dir, '510k_single_final.zip')

        if not os.path.exists(model_path):
            print(f'Model not found: {model_path}')
            continue

        print(f'\n===== Evaluating {model_path} =====')
        for mode in modes:
            summary = transfer_evaluate(model_path, mode, n_episodes=n_episodes,
                                        output_dir=output_dir)
            all_summaries[f'{step_label}_{mode}'] = summary

    return all_summaries


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Transfer evaluation')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model zip file')
    parser.add_argument('--mode', type=str, choices=['single', 'static', 'dynamic'],
                        default='single')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--batch', action='store_true',
                        help='Batch evaluate multiple checkpoints')
    parser.add_argument('--checkpoints', type=str, nargs='*', default=None)
    args = parser.parse_args()

    if args.batch:
        batch_transfer_evaluate(n_episodes=args.episodes, checkpoint_steps=args.checkpoints)
    elif args.model:
        transfer_evaluate(args.model, args.mode, n_episodes=args.episodes)
    else:
        parser.print_help()
