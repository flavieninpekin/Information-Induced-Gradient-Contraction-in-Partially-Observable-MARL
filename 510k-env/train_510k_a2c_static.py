"""510K A2C including STATIC mode."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_510k_a2c as t
t.MODES = ['single', 'static', 'dynamic']
t.SEEDS = list(range(41, 49))
import multiprocessing
multiprocessing.freeze_support()
for mode in t.MODES:
    for seed in t.SEEDS:
        fp = os.path.join(t.MODEL_DIR, f'510k_a2c_{mode}_seed{seed}_final.zip')
        if os.path.exists(fp): print(f'SKIP {mode} seed{seed}'); continue
        print(f'TRAIN A2C {mode} seed{seed}...')
        env = t.SubprocVecEnv([t.make_env_fn(mode, seed, i) for i in range(t.N_ENVS)], start_method='spawn')
        env = t.VecMonitor(env)
        model = t.A2C("MlpPolicy", env, learning_rate=3e-4, n_steps=256, gamma=0.99, gae_lambda=0.95, ent_coef=0.01,
                      policy_kwargs=dict(net_arch=dict(pi=[256,256], vf=[256,256])), verbose=0, seed=seed, device='cuda')
        ckpt = t.CheckpointCallback(save_freq=t.SAVE_EVERY, save_path=t.MODEL_DIR,
                                    name_prefix=f'510k_a2c_{mode}_seed{seed}')
        t0 = t.time.time()
        model.learn(total_timesteps=t.TOTAL_STEPS, callback=ckpt)
        model.save(fp); env.close()
        print(f'  done {t.time.time()-t0:.0f}s')
