"""
Resume training from latest checkpoint for each seed.
"""
import os, sys, time, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from overcooked_wrapper import OvercookedHiddenPartner
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models_overcooked')
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs_overcooked')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

MODES = ['single', 'static', 'dynamic']
SEEDS = [41, 42, 43]
TOTAL_STEPS = 1_000_000
SAVE_EVERY = 100_000

for mode in MODES:
    for seed in SEEDS:
        final_path = os.path.join(MODEL_DIR, f'overcooked_{mode}_seed{seed}_final.zip')
        if os.path.exists(final_path):
            print(f'SKIP (done): {mode} seed{seed}')
            continue

        key = f'{mode}_seed{seed}'

        # Find latest checkpoint
        latest_step = 0
        latest_ckpt = None
        for step in range(SAVE_EVERY, TOTAL_STEPS + 1, SAVE_EVERY):
            ckpt = os.path.join(MODEL_DIR, f'overcooked_{mode}_seed{seed}_{step}_steps.zip')
            if os.path.exists(ckpt):
                latest_step = step
                latest_ckpt = ckpt

        remaining = TOTAL_STEPS - latest_step
        print(f'\n{"=" * 60}')
        if latest_ckpt:
            print(f'RESUMING {key}: from step {latest_step}, {remaining} remaining')
        else:
            print(f'FRESH {key}: {TOTAL_STEPS} steps')

        env = OvercookedHiddenPartner(
            layout_name='cramped_room', mode=mode, horizon=400,
        )

        if latest_ckpt:
            model = PPO.load(latest_ckpt, env=env)
            model.tensorboard_log = os.path.join(LOG_DIR, mode)
        else:
            model = PPO(
                "MlpPolicy", env,
                learning_rate=3e-4, n_steps=2048, batch_size=64,
                n_epochs=10, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.01,
                policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
                verbose=1, seed=seed,
                tensorboard_log=os.path.join(LOG_DIR, mode),
            )

        ckpt_name = f'overcooked_{mode}_seed{seed}'
        callback = CheckpointCallback(
            save_freq=SAVE_EVERY,
            save_path=MODEL_DIR,
            name_prefix=ckpt_name,
        )

        t0 = time.time()
        model.learn(
            total_timesteps=remaining,
            callback=callback,
            reset_num_timesteps=False,  # don't reset step counter
        )
        elapsed = time.time() - t0

        model.save(final_path)
        env.close()
        print(f'  DONE in {elapsed:.0f}s')
        sys.stdout.flush()

print('\nALL DONE!')
