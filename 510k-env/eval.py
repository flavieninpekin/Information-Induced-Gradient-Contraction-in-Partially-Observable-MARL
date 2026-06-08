"""
评估已训练的 510K 模型
"""
import os
import time
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from env.env_510k import FiveTenKEnv


def mask_fn(env):
    return env.unwrapped._get_action_mask()


def evaluate(
    model_path: str,
    mode: str = 'single',
    n_episodes: int = 100,
    render: bool = False,
    seed: int = 42,
):
    env = FiveTenKEnv(mode=mode)
    env = ActionMasker(env, mask_fn)

    model = MaskablePPO.load(model_path)

    rewards = []
    episode_lengths = []
    wins = 0

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0
        steps = 0

        while not done:
            mask = info['action_mask']
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
            ep_reward += reward
            steps += 1

        rewards.append(ep_reward)
        episode_lengths.append(steps)
        if ep_reward > 0:
            wins += 1

        if render:
            print(f'Ep {ep + 1}: reward={ep_reward:+.0f}, steps={steps}')

    env.close()

    stats = {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards),
        'mean_steps': np.mean(episode_lengths),
        'win_rate': wins / n_episodes,
    }

    print(f'\n===== Evaluation Results =====')
    print(f'Model: {model_path}')
    print(f'Mode: {mode}')
    print(f'Episodes: {n_episodes}')
    print(f'Win rate: {stats["win_rate"]:.1%}')
    print(f'Mean reward: {stats["mean_reward"]:+.1f} +/- {stats["std_reward"]:.1f}')
    print(f'Reward range: {stats["min_reward"]:+.0f} ~ {stats["max_reward"]:+.0f}')
    print(f'Mean steps/ep: {stats["mean_steps"]:.0f}')
    print(f'===============================')

    return stats


def evaluate_random_baseline(
    mode: str = 'single',
    n_episodes: int = 200,
    seed: int = 42,
):
    """Evaluate the random baseline (agent also uses random action)."""
    env = FiveTenKEnv(mode=mode)
    rewards = []
    wins = 0

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0

        while not done:
            mask = info['action_mask']
            valid = np.where(mask)[0]
            action = int(np.random.choice(valid))
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward

        rewards.append(ep_reward)
        if ep_reward > 0:
            wins += 1

    env.close()

    stats = {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards),
        'win_rate': wins / n_episodes,
    }

    print(f'\n===== Random Baseline =====')
    print(f'Mode: {mode}')
    print(f'Episodes: {n_episodes}')
    print(f'Win rate: {stats["win_rate"]:.1%}')
    print(f'Mean reward: {stats["mean_reward"]:+.1f} +/- {stats["std_reward"]:.1f}')
    print(f'===========================')

    return stats


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='510K Model Evaluation')
    parser.add_argument('--model', type=str, required=False,
                        help='Path to trained model zip file')
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'static', 'dynamic'])
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--baseline', action='store_true',
                        help='Evaluate random baseline instead')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.baseline:
        evaluate_random_baseline(mode=args.mode, n_episodes=args.episodes, seed=args.seed)
    elif args.model:
        evaluate(model_path=args.model, mode=args.mode, n_episodes=args.episodes, seed=args.seed)
    else:
        parser.print_help()
