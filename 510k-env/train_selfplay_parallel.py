"""
Parallel self-play training with SubprocVecEnv for GPU utilization.

Each seed uses 8 parallel environments. Seeds run sequentially.

Usage:
  python train_selfplay_parallel.py --mode single --seeds 4 --timesteps 1000000
"""
import os, time, json
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from env.env_510k import FiveTenKEnv


def mask_fn(env):
    return env.unwrapped._get_action_mask()


def _make_env_fn(mode, rank, seed):
    """Factory for SubprocVecEnv."""
    def _init():
        set_random_seed(seed + rank * 1000)
        env = FiveTenKEnv(mode=mode, num_players=4)
        env = ActionMasker(env, mask_fn)
        return env
    return _init


def train_seed_parallel(mode: str, seed: int, total_timesteps: int = 1_000_000,
                        n_envs: int = 8, log_dir: str = 'logs_selfplay',
                        model_dir: str = 'models_selfplay', save_every: int = 100_000):
    """Train one seed with multiple parallel envs."""
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Create parallel envs
    env_fns = [_make_env_fn(mode, i, seed) for i in range(n_envs)]
    env = SubprocVecEnv(env_fns)

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=1,
        seed=seed,
        tensorboard_log=os.path.join(log_dir, mode),
    )

    # Enable self-play on all sub-envs (post-init)
    # SubprocVecEnv wraps multiple envs; set_policy_bot on each via a method
    env.env_method('set_policy_bot', model)

    ckpt_name = f'510k_{mode}_seed{seed}'
    callback = CheckpointCallback(
        save_freq=save_every // n_envs,  # adjust for parallel envs
        save_path=model_dir,
        name_prefix=ckpt_name,
    )

    print(f'\n[{mode}] Seed {seed}: training {total_timesteps} steps '
          f'(self-play, {n_envs} parallel envs)...')
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback,
                progress_bar=False)
    elapsed = time.time() - t0
    print(f'  Done in {elapsed:.0f}s ({elapsed/60:.1f} min)')

    final_path = os.path.join(model_dir, f'{ckpt_name}_final.zip')
    model.save(final_path)
    print(f'  Saved to {final_path}')
    env.close()
    return model


def train_all_parallel(mode: str, start_seed: int, n_seeds: int,
                       timesteps: int = 1_000_000, n_envs: int = 8,
                       save_every: int = 100_000):
    """Train seeds sequentially, each with parallel envs."""
    seeds = [start_seed + i for i in range(n_seeds)]
    results = {}
    for seed in seeds:
        t0 = time.time()
        train_seed_parallel(mode, seed, timesteps, n_envs=n_envs,
                            save_every=save_every)
        elapsed = time.time() - t0
        results[seed] = {'time_s': elapsed, 'timesteps': timesteps}

    summary_path = 'models_selfplay/single_seeds_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nAll {n_seeds} seeds done. Summary: {summary_path}')
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'static', 'dynamic'])
    parser.add_argument('--seeds', type=int, default=4)
    parser.add_argument('--start-seed', type=int, default=51)
    parser.add_argument('--timesteps', type=int, default=1_000_000)
    parser.add_argument('--save-every', type=int, default=100_000)
    parser.add_argument('--envs', type=int, default=8,
                        help='Number of parallel environments')
    args = parser.parse_args()

    train_all_parallel(
        mode=args.mode,
        start_seed=args.start_seed,
        n_seeds=args.seeds,
        timesteps=args.timesteps,
        n_envs=args.envs,
        save_every=args.save_every,
    )
