"""
IRL implementation: Contrastive Reward Learning + Tabular MaxEnt IRL.

Two approaches:
  1. Contrastive: w = μ_expert - μ_random  (closed-form, fast)
  2. Tabular MaxEnt IRL on discrete MDP  (iterative, more rigorous)
"""
import os
import json
import gzip
import pickle
from typing import List, Dict, Optional, Tuple

import numpy as np

from env.features import (
    FEATURE_NAMES, FEATURE_DIM, FEATURE_BINS,
    extract_features, discretize_state, state_to_features
)


# ============================================================
#  Approach 1: Contrastive Reward Learning
# ============================================================

def compute_contrastive_weights(expert_fe: np.ndarray,
                                random_fe: np.ndarray,
                                lambda_reg: float = 0.01) -> np.ndarray:
    """Contrastive reward weights: w = (μ_E - μ_R) / λ.

    Args:
        expert_fe: expert feature expectations (5-dim)
        random_fe: random baseline feature expectations (5-dim)
        lambda_reg: L2 regularization strength

    Returns:
        w: 5-dim weight vector
    """
    return (expert_fe - random_fe) / lambda_reg


def load_feature_expectations(summary_path: str) -> np.ndarray:
    """Load per-player feature expectations from transfer summary."""
    with open(summary_path) as f:
        data = json.load(f)
    return np.array(data['feature_expectations'])


def mode_from_path(path: str) -> str:
    """Infer mode from summary filename — extract from the suffix before _summary."""
    base = os.path.basename(path)
    # Remove extension
    if base.endswith('_trajectories.pkl.gz'):
        base = base.replace('_trajectories.pkl.gz', '')
    elif base.endswith('_summary.json'):
        base = base.replace('_summary.json', '')
    # The mode is the LAST segment after the last underscore
    # e.g., 510k_single_final_single → single
    parts = base.split('_')
    for mode in ['single', 'static', 'dynamic']:
        if parts[-1] == mode:
            return mode
    return 'unknown'


def run_contrastive_irl(expert_dir: str = 'transfer_data',
                         lambda_reg: float = 0.01,
                         random_fe_path: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Run contrastive IRL for all modes present in expert_dir.

    Returns dict mapping mode -> weight vector.
    """
    # Find summary files
    summary_files = [f for f in os.listdir(expert_dir) if f.endswith('_summary.json')]
    if not summary_files:
        raise FileNotFoundError(f'No summary files found in {expert_dir}')

    # Use the first summary found. For batch runs, take the latest (highest step count).
    # For simplicity, group by mode and take the unique one.
    mode_to_path = {}
    for fname in summary_files:
        mode = mode_from_path(fname)
        if mode != 'unknown':
            mode_to_path[mode] = os.path.join(expert_dir, fname)

    # Random baseline feature expectations
    if random_fe_path is None:
        random_fe_path = os.path.join(expert_dir, 'random_baseline_features.json')

    if os.path.exists(random_fe_path):
        with open(random_fe_path) as f:
            random_data = json.load(f)
        μ_rand = np.array(random_data['feature_expectations'])
    else:
        print(f'Warning: random baseline not found at {random_fe_path}')
        print(f'Assuming zero baseline (will compute relative weights).')
        μ_rand = np.zeros(FEATURE_DIM)

    # Compute weights for each mode
    weights = {}
    for mode, path in mode_to_path.items():
        μ_exp = load_feature_expectations(path)
        w = compute_contrastive_weights(μ_exp, μ_rand, lambda_reg)
        weights[mode] = w
        print(f'  [{mode}] μ_E={μ_exp.round(3)}  w={w.round(3)}')

    return weights


def compute_random_baseline_features(n_episodes: int = 1000, seed: int = 0,
                                     output_dir: str = 'transfer_data') -> np.ndarray:
    """Run random-bot episodes and compute feature expectations."""
    from env.game import Game, GameMode
    import random

    os.makedirs(output_dir, exist_ok=True)

    all_features = []
    for ep in range(n_episodes):
        random.seed(seed + ep)
        np.random.seed(seed + ep)
        game = Game(mode=GameMode.SINGLE, num_players=4)
        while not game.is_over:
            pid = game.current_player
            feats = extract_features(game, pid)
            all_features.append(feats)
            valid = game.get_valid_actions(pid)
            if valid:
                game.play_cards(pid, random.choice(valid).cards)
            elif game.can_pass(pid):
                game.pass_turn(pid)

        if (ep + 1) % 200 == 0:
            print(f'  Random baseline: {ep+1}/{n_episodes}')

    μ_rand = np.mean(all_features, axis=0) if all_features else np.zeros(FEATURE_DIM)
    print(f'  Random baseline FE: {μ_rand.round(3)}')

    save_path = os.path.join(output_dir, 'random_baseline_features.json')
    with open(save_path, 'w') as f:
        json.dump({'feature_expectations': μ_rand.tolist(),
                   'n_episodes': n_episodes,
                   'feature_names': FEATURE_NAMES}, f, indent=2)
    return μ_rand


# ============================================================
#  Approach 2: Tabular MaxEnt IRL
# ============================================================

def build_empirical_mdp(trajectories: list, gamma: float = 0.99):
    """Build empirical MDP from trajectory data.

    Args:
        trajectories: list of episode dicts from transfer.py
        gamma: discount factor

    Returns:
        n_states, n_actions, transitions, feature_map, starts
    """
    n_states = FEATURE_BINS ** FEATURE_DIM
    max_act = 300  # match MAX_ACTIONS

    # transition_counts[s][a][s'] = count
    # state_visit_counts[s] = count
    # feature_map[s] = continuous feature vector (bin center)
    trans_count = np.zeros((n_states, max_act, n_states), dtype=np.float64)
    state_count = np.zeros(n_states, dtype=np.float64)
    action_count = np.zeros((n_states, max_act), dtype=np.float64)
    feature_map = np.zeros((n_states, FEATURE_DIM), dtype=np.float32)
    for s in range(n_states):
        feature_map[s] = state_to_features(s)

    for ep_data in trajectories:
        for pid, traj in ep_data['trajectories'].items():
            for entry in traj:
                s = discretize_state(np.array(entry['features']))
                a = entry['action']
                s_next = discretize_state(np.array(entry['next_features']))

                if a < max_act:
                    trans_count[s, a, s_next] += 1.0
                    state_count[s] += 1.0
                    action_count[s, a] += 1.0

    # Convert to probabilities
    # trans_prob[s][a] = distribution over s'
    trans_prob = np.zeros((n_states, max_act, n_states), dtype=np.float64)
    for s in range(n_states):
        for a in range(max_act):
            total = trans_count[s, a].sum()
            if total > 0:
                trans_prob[s, a] = trans_count[s, a] / total
            else:
                # Self-loop for unknown transitions
                trans_prob[s, a, s] = 1.0

    # Terminal state probability: if total is very low, treat as absorbing
    terminal = np.zeros(n_states, dtype=bool)
    for s in range(n_states):
        if state_count[s] < 1:
            terminal[s] = True  # unvisited states = terminal

    return {
        'n_states': n_states,
        'n_actions': max_act,
        'trans_prob': trans_prob,
        'feature_map': feature_map,
        'state_count': state_count,
        'action_count': action_count,
        'gamma': gamma,
        'terminal': terminal,
    }


def tabular_maxent_irl(expert_trajectories: list,
                       mdp: dict,
                       lr: float = 0.1,
                       n_iters: int = 200,
                       eps: float = 1e-6) -> np.ndarray:
    """Tabular MaxEnt IRL (Ziebart et al. 2008).

    Recovers w such that the soft-optimal policy under R(s)=w·φ(s)
    matches the expert's feature expectations.

    Returns:
        w: recovered weight vector (5-dim)
    """
    n_states = mdp['n_states']
    n_actions = mdp['n_actions']
    trans_prob = mdp['trans_prob']
    feature_map = mdp['feature_map']
    gamma = mdp['gamma']

    # Compute expert feature expectations from trajectories
    expert_phi = []
    for ep_data in expert_trajectories:
        for pid, traj in ep_data['trajectories'].items():
            for entry in traj:
                expert_phi.append(entry['features'])
    μ_E = np.mean(expert_phi, axis=0) if expert_phi else np.zeros(FEATURE_DIM)

    # Initialize weights
    w = np.ones(FEATURE_DIM, dtype=np.float64) / FEATURE_DIM

    for iteration in range(n_iters):
        # 1. Compute soft value functions
        R = feature_map @ w  # shape: (n_states,)

        # Value iteration for soft optimal policy
        V = np.zeros(n_states, dtype=np.float64)
        for _ in range(500):  # inner value iteration
            # Q(s,a) = R(s) + γ ∑ P(s'|s,a) V(s')
            Q = R[:, None] + gamma * (trans_prob @ V)  # (n_states, n_actions)
            # Softmax: V(s) = log ∑_a exp(Q(s,a))
            V_new = np.zeros(n_states, dtype=np.float64)
            for s in range(n_states):
                max_q = np.max(Q[s])
                V_new[s] = max_q + np.log(np.sum(np.exp(Q[s] - max_q)))

            if np.max(np.abs(V_new - V)) < eps:
                break
            V = V_new

        # 2. Compute expected state visitation frequencies (since no explicit start distribution,
        #    use the empirical one from trajectory data)
        #    π(a|s) ∝ exp(Q(s,a) - V(s))
        policy = np.zeros((n_states, n_actions), dtype=np.float64)
        for s in range(n_states):
            logits = Q[s] - V[s]
            logits -= np.max(logits)
            policy[s] = np.exp(logits)
            policy[s] /= (policy[s].sum() + 1e-10)

        # State visitation: use uniform start distribution weighted by empirical count
        state_count = mdp['state_count'] + 1.0  # Laplace smoothing
        state_dist = state_count / state_count.sum()

        # 3. Compute expected feature counts under policy
        μ_w = np.zeros(FEATURE_DIM, dtype=np.float64)
        for s in range(n_states):
            if mdp['terminal'][s]:
                continue
            μ_w += state_dist[s] * feature_map[s]

        # 4. Gradient and update
        grad = μ_E - μ_w
        w += lr * grad

        # 5. Normalize w (prevent explosion)
        w_norm = np.linalg.norm(w)
        if w_norm > 10.0:
            w *= 10.0 / w_norm

        if (iteration + 1) % 50 == 0:
            print(f'    IRL iter {iteration+1}: ||grad||={np.linalg.norm(grad):.4f}, w={np.round(w, 3)}')

        if np.linalg.norm(grad) < eps:
            print(f'    Converged at iter {iteration+1}')
            break

    return w.astype(np.float32)


# ============================================================
#  Full IRL Pipeline
# ============================================================

def load_trajectories(path: str) -> list:
    """Load trajectories saved by transfer.py."""
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)


def run_full_irl_pipeline(transfer_dir: str = 'transfer_data',
                          random_fe_path: Optional[str] = None,
                          lambda_reg: float = 0.01,
                          use_tabular: bool = True) -> dict:
    """Run both contrastive and tabular IRL, save results."""
    # 1. Find trajectory files
    traj_files = [f for f in os.listdir(transfer_dir) if f.endswith('_trajectories.pkl.gz')]
    summary_files = {f.replace('_summary.json', ''): f
                     for f in os.listdir(transfer_dir) if f.endswith('_summary.json')}

    # 2. Contrastive IRL
    print('\n=== Contrastive IRL ===')
    weights_cont = run_contrastive_irl(transfer_dir, lambda_reg, random_fe_path)

    # 3. Tabular MaxEnt IRL
    weights_tab = {}
    if use_tabular:
        print('\n=== Tabular MaxEnt IRL ===')
        for traj_file in traj_files:
            mode = mode_from_path(traj_file)
            if mode == 'unknown':
                continue
            print(f'\n  Processing {mode}...')
            trajectories = load_trajectories(os.path.join(transfer_dir, traj_file))
            print(f'  Loaded {len(trajectories)} episodes')
            mdp = build_empirical_mdp(trajectories)
            w = tabular_maxent_irl(trajectories, mdp, lr=0.1, n_iters=200)
            weights_tab[mode] = w
            print(f'  [{mode}] Tabular IRL w = {np.round(w, 3)}')

    # 4. Save all weights
    results = {
        'feature_names': FEATURE_NAMES,
        'contrastive_weights': {k: v.tolist() for k, v in weights_cont.items()},
        'tabular_weights': {k: v.tolist() for k, v in weights_tab.items()},
        'parameters': {
            'lambda_reg': lambda_reg,
            'use_tabular': use_tabular,
        }
    }

    save_path = os.path.join(transfer_dir, 'irl_results.json')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nIRL results saved to {save_path}')

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='IRL on transfer data')
    parser.add_argument('--tabular', action='store_true',
                        help='Also run tabular MaxEnt IRL (slower)')
    parser.add_argument('--random-baseline', action='store_true',
                        help='Compute random baseline features first')
    parser.add_argument('--transfer-dir', type=str, default='transfer_data')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Episodes for random baseline')
    args = parser.parse_args()

    if args.random_baseline:
        compute_random_baseline_features(n_episodes=args.episodes)

    run_full_irl_pipeline(transfer_dir=args.transfer_dir,
                          use_tabular=args.tabular)
