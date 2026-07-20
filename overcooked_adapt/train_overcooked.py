"""
Overcooked training with hidden partner types.

Usage:
  python train_overcooked.py --mode single --seeds 3 --timesteps 1000000
"""
import os
import time
import json
import argparse
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from overcooked_wrapper import OvercookedHiddenPartner


MODEL_DIR = 'models_overcooked'
LOG_DIR = 'logs_overcooked'
LAYOUT_NAME = 'cramped_room'


def make_env(mode='single', layout_name=LAYOUT_NAME, horizon=400, switch_interval=50):
    return OvercookedHiddenPartner(
        layout_name=layout_name,
        mode=mode,
        horizon=horizon,
        switch_interval=switch_interval,
    )


def train_seed(mode, seed, total_timesteps=1_000_000,
               log_dir=LOG_DIR, model_dir=MODEL_DIR, save_every=100_000):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    env = make_env(mode=mode)

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        tensorboard_log=os.path.join(log_dir, mode),
    )

    ckpt_name = f'overcooked_{mode}_seed{seed}'
    callback = CheckpointCallback(
        save_freq=save_every,
        save_path=model_dir,
        name_prefix=ckpt_name,
    )

    print(f'\n[{mode}] Seed {seed}: training {total_timesteps} steps...')
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    elapsed = time.time() - t0
    print(f'  Done in {elapsed:.0f}s ({elapsed/60:.1f} min)')

    final_path = os.path.join(model_dir, f'{ckpt_name}_final.zip')
    model.save(final_path)
    print(f'  Saved to {final_path}')
    env.close()
    return model


def train_all_seeds(mode, n_seeds=3, timesteps=1_000_000):
    seeds = [41 + i for i in range(n_seeds)]
    results = {}
    for seed in seeds:
        t0 = time.time()
        train_seed(mode, seed, timesteps)
        elapsed = time.time() - t0
        results[seed] = {'time_s': elapsed, 'timesteps': timesteps}

    summary_path = os.path.join(MODEL_DIR, f'{mode}_seeds_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n[mode={mode}] All {n_seeds} seeds done. Summary: {summary_path}')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Overcooked Training')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['single', 'static', 'dynamic'])
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--timesteps', type=int, default=1_000_000)
    parser.add_argument('--save-every', type=int, default=100_000)
    args = parser.parse_args()

    train_all_seeds(
        mode=args.mode,
        n_seeds=args.seeds,
        timesteps=args.timesteps,
    )
