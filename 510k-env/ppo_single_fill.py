"""Train 510K PPO SINGLE with masked env."""
import os, sys, time, multiprocessing
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '510k-env'))
from env.dqn_wrapper import FiveTenKMaskedEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

MDIR = os.path.join(os.path.dirname(__file__), '..', 'models_510k_ppo_sa')
os.makedirs(MDIR, exist_ok=True)
N_ENVS = 8; TS = 1_000_000

if __name__ == '__main__':
    multiprocessing.freeze_support()
    for seed in [41, 42, 43, 44]:
        fp = os.path.join(MDIR, f'ppo_sa_single_seed{seed}_final.zip')
        if os.path.exists(fp):
            print(f'SKIP single s{seed}')
            continue
        print(f'TRAIN PPO SINGLE s{seed}...')
        fns = [lambda s=seed: FiveTenKMaskedEnv(mode='single') for _ in range(N_ENVS)]
        env = SubprocVecEnv(fns, start_method='spawn')
        env = VecMonitor(env)
        m = PPO('MlpPolicy', env, learning_rate=3e-4, n_steps=256, batch_size=256, n_epochs=10,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                policy_kwargs=dict(net_arch=dict(pi=[256,256], vf=[256,256])),
                verbose=0, seed=seed, device='cuda')
        t0 = time.time(); m.learn(total_timesteps=TS); m.save(fp); env.close()
        print(f'  done {time.time()-t0:.0f}s')
    print('ALL SINGLE PPO DONE')
