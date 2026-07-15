"""
Train SINGLE seeds in parallel using multiprocessing.
Each seed runs independently (own model, own env), sharing the GPU.
"""
import os, time, json, sys
import multiprocessing as mp

os.environ.setdefault('OMP_NUM_THREADS', '2')  # limit CPU threads per process


def train_single_seed(seed: int, total_timesteps: int = 1_000_000,
                      save_every: int = 100_000) -> dict:
    """Train one SINGLE seed (runs in a subprocess)."""
    import warnings
    warnings.filterwarnings('ignore')
    import numpy as np
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
    from stable_baselines3.common.callbacks import CheckpointCallback
    from env.env_510k import FiveTenKEnv

    def mask_fn(e): return e.unwrapped._get_action_mask()

    env = FiveTenKEnv(mode='single', num_players=4)
    env = ActionMasker(env, mask_fn)

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=0, seed=seed,
    )

    # Self-play
    env.unwrapped.set_policy_bot(model)

    ckpt_name = f'510k_single_seed{seed}'
    callback = CheckpointCallback(
        save_freq=save_every, save_path='models_selfplay', name_prefix=ckpt_name)

    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback,
                progress_bar=False)
    elapsed = time.time() - t0

    final_path = f'models_selfplay/{ckpt_name}_final.zip'
    model.save(final_path)
    env.close()

    result = {'seed': seed, 'time_s': elapsed, 'final_path': final_path}
    print(f'[seed={seed}] Done in {elapsed/60:.1f} min -> {final_path}', flush=True)
    return result


def main(n_seeds: int = 4, start_seed: int = 51, timesteps: int = 1_000_000,
         save_every: int = 100_000):
    """Launch n_seeds training processes in parallel."""
    seeds = [start_seed + i for i in range(n_seeds)]
    os.makedirs('models_selfplay', exist_ok=True)

    print(f'Training {n_seeds} SINGLE seeds ({seeds}) in parallel...')
    print(f'Each seed: {timesteps} steps, checkpoints every {save_every}')
    t0 = time.time()

    # Use spawn for Windows compatibility with CUDA
    mp.set_start_method('spawn', force=True)

    with mp.Pool(processes=n_seeds) as pool:
        args = [(s, timesteps, save_every) for s in seeds]
        results = pool.starmap(train_single_seed, args)

    total = time.time() - t0
    print(f'\nAll {n_seeds} seeds done in {total/60:.1f} min')
    for r in results:
        print(f'  seed={r["seed"]}: {r["time_s"]/60:.1f} min')
    return results


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, default=4)
    p.add_argument('--start-seed', type=int, default=51)
    p.add_argument('--timesteps', type=int, default=1_000_000)
    p.add_argument('--save-every', type=int, default=100_000)
    args = p.parse_args()
    main(args.seeds, args.start_seed, args.timesteps, args.save_every)
