"""
Parallel training: SubprocVecEnv for 8x speedup.
"""
import os, sys, time, multiprocessing

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from overcooked_wrapper import OvercookedHiddenPartner
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models_overcooked')
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs_overcooked')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

N_ENVS = 8
MODES = ['single', 'static', 'dynamic']
SEEDS = [41, 42, 43]
TOTAL_STEPS = 1_000_000
SAVE_EVERY = 100_000


def make_env_fn(mode, seed, rank):
    def _init():
        return OvercookedHiddenPartner(
            layout_name='cramped_room', mode=mode, horizon=400,
            seed=seed + rank * 100,
        )
    return _init


if __name__ == '__main__':
    multiprocessing.freeze_support()

    for mode in MODES:
        for seed in SEEDS:
            final_path = os.path.join(MODEL_DIR, f'overcooked_{mode}_seed{seed}_final.zip')
            if os.path.exists(final_path):
                print(f'SKIP (done): {mode} seed{seed}')
                continue

            key = f'{mode}_seed{seed}'
            print(f'\n{"="*60}')
            print(f'TRAINING: {key} ({N_ENVS} parallel envs, {TOTAL_STEPS} steps)')

            env_fns = [make_env_fn(mode, seed, i) for i in range(N_ENVS)]
            env = SubprocVecEnv(env_fns, start_method='spawn')
            env = VecMonitor(env)

            model = PPO(
                "MlpPolicy", env,
                learning_rate=3e-4, n_steps=256,
                batch_size=256,
                n_epochs=10, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.01,
                policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
                verbose=1, seed=seed,
                tensorboard_log=os.path.join(LOG_DIR, mode),
                device='cuda',
            )

            ckpt_name = f'overcooked_{mode}_seed{seed}'
            callback = CheckpointCallback(
                save_freq=SAVE_EVERY,
                save_path=MODEL_DIR,
                name_prefix=ckpt_name,
            )

            t0 = time.time()
            model.learn(total_timesteps=TOTAL_STEPS, callback=callback)
            elapsed = time.time() - t0

            model.save(final_path)
            env.close()
            print(f'  DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)')
            sys.stdout.flush()

    print('\nALL DONE!')
