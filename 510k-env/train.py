"""
510K RL 训练脚本
支持三种模式 + 自对弈
"""
import os
import random
import time
from typing import Optional, cast

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.callbacks import CheckpointCallback

from env.env_510k import FiveTenKEnv


def mask_fn(env: gym.Env) -> np.ndarray:
    # env.unwrapped is typed as gym.Env, but our custom env implements _get_action_mask
    # Cast to FiveTenKEnv so static checkers recognize the method
    return cast(FiveTenKEnv, env.unwrapped)._get_action_mask()


def make_env(mode: str = 'single', rank: int = 0) -> gym.Env:
    num_players = 3 if mode == '3p' else 4
    env = FiveTenKEnv(mode=mode, num_players=num_players)
    env = ActionMasker(env, mask_fn)
    return env


def train(
    mode: str = 'single',
    total_timesteps: int = 1_000_000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    policy_kwargs: Optional[dict] = None,
    log_dir: str = 'logs',
    model_dir: str = 'models',
    save_every_env_steps: int = 16384,
    seed: int = 42,
):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    env = make_env(mode=mode)

    if policy_kwargs is None:
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        )

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        tensorboard_log=log_dir,
    )

    # save_freq is in environment steps (not rollouts)
    callback = CheckpointCallback(
        save_freq=save_every_env_steps,
        save_path=model_dir,
        name_prefix=f'510k_{mode}',
    )

    # determine number of environments (for VecEnv support); default to 1
    n_envs = getattr(env, 'num_envs', 1)
    print(f'Starting training: mode={mode}, timesteps={total_timesteps}, n_envs={n_envs}')
    start = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=False,
    )
    elapsed = time.time() - start
    print(f'Training done in {elapsed:.1f}s')

    final_path = os.path.join(model_dir, f'510k_{mode}_final.zip')
    model.save(final_path)
    print(f'Model saved to {final_path}')

    env.close()
    return model


def train_self_play(
    mode: str = 'dynamic',
    phases: int = 3,
    timesteps_per_phase: int = 500_000,
    log_dir: str = 'logs',
    model_dir: str = 'models',
    seed: int = 42,
):
    os.makedirs(model_dir, exist_ok=True)

    model_paths = []

    for phase in range(phases):
        print(f'\n===== Self-play Phase {phase + 1}/{phases} =====')

        # Phase 1: vs random bots
        # Phase 2+: vs random + previous checkpoints
        env = FiveTenKEnv(mode=mode) if phase == 0 else make_env(mode=mode)
        env = ActionMasker(env, mask_fn)

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
            seed=seed + phase,
            tensorboard_log=os.path.join(log_dir, f'selfplay_phase_{phase}'),
        )

        ckpt = CheckpointCallback(
            save_freq=16384, save_path=model_dir,
            name_prefix=f'selfplay_phase{phase}',
        )
        model.learn(total_timesteps=timesteps_per_phase, callback=ckpt, progress_bar=True)

        phase_path = os.path.join(model_dir, f'selfplay_phase_{phase}.zip')
        model.save(phase_path)
        model_paths.append(phase_path)
        print(f'Phase {phase + 1} saved to {phase_path}')

        env.close()

    print(f'\nSelf-play done. Checkpoints: {model_paths}')
    return model_paths


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='510K RL Training')
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'static', 'dynamic', '3p'])
    parser.add_argument('--timesteps', type=int, default=1_000_000)
    parser.add_argument('--self-play', action='store_true',
                        help='Enable self-play training')
    parser.add_argument('--phases', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.self_play:
        train_self_play(
            mode=args.mode,
            phases=args.phases,
            timesteps_per_phase=args.timesteps // args.phases,
            seed=args.seed,
        )
    else:
        train(
            mode=args.mode,
            total_timesteps=args.timesteps,
            seed=args.seed,
        )
