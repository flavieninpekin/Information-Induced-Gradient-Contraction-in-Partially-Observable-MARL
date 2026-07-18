"""
MAPPO training: centralized critic, decentralized actors, self-play.
"""
import os, time, json
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from env.mappo_env import MAPPOEnv
from env.mappo_policy import MAPPOCentralizedCriticPolicy


def train_mappo_seed(mode, seed, total_timesteps=1_000_000, save_every=100_000):
    from stable_baselines3.common.callbacks import CheckpointCallback

    def mask_fn(env):
        return env.unwrapped._get_action_mask()

    env = MAPPOEnv(mode=mode, num_players=4)
    env = ActionMasker(env, mask_fn)

    model = MaskablePPO(
        MAPPOCentralizedCriticPolicy,
        env,
        learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
        policy_kwargs=dict(
            net_arch=[],  # handled by custom MLP
        ),
        verbose=1, seed=seed,
        tensorboard_log=f'logs_mappo/{mode}',
    )

    # Self-play
    env.unwrapped.set_policy_bot(model)

    ckpt_name = f'510k_mappo_{mode}_seed{seed}'
    callback = CheckpointCallback(save_freq=save_every, save_path='models_selfplay',
                                  name_prefix=ckpt_name)

    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
    elapsed = time.time() - t0

    final_path = f'models_selfplay/{ckpt_name}_final.zip'
    model.save(final_path)
    env.close()
    print(f'[MAPPO {mode}] seed={seed} done: {elapsed/60:.1f} min', flush=True)
    return {'mode': mode, 'seed': seed, 'time_s': elapsed}


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['single','static','dynamic'], required=True)
    p.add_argument('--seeds', type=int, default=2)
    p.add_argument('--start-seed', type=int, default=91)
    p.add_argument('--timesteps', type=int, default=1_000_000)
    args = p.parse_args()

    seeds = [args.start_seed + i for i in range(args.seeds)]
    for seed in seeds:
        train_mappo_seed(args.mode, seed, args.timesteps)
    print('All MAPPO seeds done.')
