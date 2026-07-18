"""
Train multiple seeds in parallel using multiprocessing.
Supports any mode: single, static, dynamic.
"""
import os, time, sys
import multiprocessing as mp
os.environ.setdefault('OMP_NUM_THREADS', '2')


def train_seed(mode: str, seed: int, total_timesteps: int = 1_000_000,
               save_every: int = 100_000) -> dict:
    import warnings; warnings.filterwarnings('ignore')
    import numpy as np
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
    from stable_baselines3.common.callbacks import CheckpointCallback
    from env.env_510k import FiveTenKEnv

    def mask_fn(e): return e.unwrapped._get_action_mask()

    env = FiveTenKEnv(mode=mode, num_players=4)
    env = ActionMasker(env, mask_fn)
    model = MaskablePPO(
        MaskableActorCriticPolicy, env,
        learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
        policy_kwargs=dict(net_arch=dict(pi=[256,256], vf=[256,256])),
        verbose=0, seed=seed,
    )
    env.unwrapped.set_policy_bot(model)

    ckpt_name = f'510k_{mode}_seed{seed}'
    callback = CheckpointCallback(save_freq=save_every, save_path='models_selfplay',
                                  name_prefix=ckpt_name)
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
    elapsed = time.time() - t0

    final_path = f'models_selfplay/{ckpt_name}_final.zip'
    model.save(final_path)
    env.close()
    print(f'[{mode}] seed={seed} done: {elapsed/60:.1f} min', flush=True)
    return {'mode': mode, 'seed': seed, 'time_s': elapsed}


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--mode', type=str, required=True,
                   choices=['single', 'static', 'dynamic', 'obvious'])
    p.add_argument('--seeds', type=int, default=4)
    p.add_argument('--start-seed', type=int, default=61)
    p.add_argument('--timesteps', type=int, default=1_000_000)
    p.add_argument('--save-every', type=int, default=100_000)
    args = p.parse_args()

    seeds = [args.start_seed + i for i in range(args.seeds)]
    os.makedirs('models_selfplay', exist_ok=True)

    print(f'Training {args.seeds} {args.mode} seeds ({seeds}) in parallel...')

    mp.set_start_method('spawn', force=True)
    with mp.Pool(processes=args.seeds) as pool:
        arglist = [(args.mode, s, args.timesteps, args.save_every) for s in seeds]
        pool.starmap(train_seed, arglist)

    print('All done.')
