"""
Self-play training: all 4 players use the same policy.

Usage:
  python train_selfplay.py --mode single --seeds 6 --timesteps 1000000
"""
import os, time, sys, json
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.callbacks import CheckpointCallback

from env.env_510k import FiveTenKEnv


def mask_fn(env):
    return env.unwrapped._get_action_mask()


def make_env(mode='single', num_players=4):
    env = FiveTenKEnv(mode=mode, num_players=num_players)
    env = ActionMasker(env, mask_fn)
    return env


def train_seed(mode: str, seed: int, total_timesteps: int = 1_000_000,
               log_dir: str = 'logs_selfplay', model_dir: str = 'models_selfplay',
               save_every: int = 100_000):
    """Train one seed with self-play (all 4 players use the same policy)."""
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    env = make_env(mode=mode, num_players=4)
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

    # Enable self-play: non-P0 players use the same policy
    env.unwrapped.set_policy_bot(model)

    ckpt_name = f'510k_{mode}_seed{seed}'
    # save_freq is in env steps (SB3 calls _on_step after every env step)
    callback = CheckpointCallback(
        save_freq=save_every,
        save_path=model_dir,
        name_prefix=ckpt_name,
    )

    print(f'\n[{mode}] Seed {seed}: training {total_timesteps} steps (self-play)...')
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    elapsed = time.time() - t0
    print(f'  Done in {elapsed:.0f}s ({elapsed/60:.1f} min)')

    final_path = os.path.join(model_dir, f'{ckpt_name}_final.zip')
    model.save(final_path)
    print(f'  Saved to {final_path}')
    env.close()
    return model


def train_all_seeds(mode: str, n_seeds: int = 6, timesteps: int = 1_000_000,
                    log_dir: str = 'logs_selfplay', model_dir: str = 'models_selfplay',
                    save_every: int = 100_000):
    """Train multiple seeds sequentially."""
    seeds = [41 + i for i in range(n_seeds)]
    results = {}
    for seed in seeds:
        t0 = time.time()
        train_seed(mode, seed, timesteps, log_dir, model_dir, save_every)
        elapsed = time.time() - t0
        results[seed] = {'time_s': elapsed, 'timesteps': timesteps}

    summary_path = os.path.join(model_dir, f'{mode}_seeds_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n[mode={mode}] All {n_seeds} seeds done. Summary: {summary_path}')
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='510K Self-Play Training')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['single', 'static', 'dynamic'])
    parser.add_argument('--seeds', type=int, default=6)
    parser.add_argument('--timesteps', type=int, default=1_000_000)
    parser.add_argument('--save-every', type=int, default=100_000)
    args = parser.parse_args()

    train_all_seeds(
        mode=args.mode,
        n_seeds=args.seeds,
        timesteps=args.timesteps,
        save_every=args.save_every,
    )
