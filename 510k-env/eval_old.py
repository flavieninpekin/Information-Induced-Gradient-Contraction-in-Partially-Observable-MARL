import os, sys
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(__file__))
from env.env_510k import FiveTenKEnv


def mask_fn(env):
    return env.unwrapped._get_action_mask()


def evaluate(model_path, n_episodes=100, seed=42):
    env = FiveTenKEnv(mode='single', num_players=4)
    env = ActionMasker(env, mask_fn)

    model = MaskablePPO.load(model_path)

    rewards, wins = [], 0
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done, ep_reward = 0.0, 0.0
        while not done:
            mask = info['action_mask']
            action, _ = model.predict(obs[:112], action_masks=mask, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
            ep_reward += reward
        rewards.append(ep_reward)
        if ep_reward > 0:
            wins += 1

    env.close()
    stats = {
        'mean_reward': np.mean(rewards), 'std_reward': np.std(rewards),
        'win_rate': wins / n_episodes,
        'min_reward': np.min(rewards), 'max_reward': np.max(rewards),
    }
    name = os.path.basename(model_path)
    print(f'{name:40s}  win_rate={stats["win_rate"]:.1%}  mean={stats["mean_reward"]:+6.1f} +/- {stats["std_reward"]:.1f}  [{stats["min_reward"]:+}, {stats["max_reward"]:+}]')
    return stats


if __name__ == '__main__':
    model_dir = '../models'
    # test a range of checkpoints
    steps = [16384, 65536, 163840, 327680, 491520, 655360, 819200, 933888, 999424]
    for s in steps:
        f = f'510k_single_{s}_steps.zip'
        p = os.path.join(model_dir, f)
        if os.path.exists(p):
            evaluate(p, n_episodes=100)
    # final model
    p = os.path.join(model_dir, '510k_single_final.zip')
    if os.path.exists(p):
        evaluate(p, n_episodes=100)
