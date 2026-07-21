"""
Robust batch training: runs all modes × seeds, skips completed, saves checkpoints.
"""
import os, sys, time, json, traceback

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
TIMESTEPS = 1_000_000
SAVE_EVERY = 100_000

results_log = {}

for mode in MODES:
    for seed in SEEDS:
        final_path = os.path.join(MODEL_DIR, f'overcooked_{mode}_seed{seed}_final.zip')
        if os.path.exists(final_path):
            print(f'SKIP: {final_path} already exists')
            continue

        key = f'{mode}_seed{seed}'
        print(f'\n{"="*60}')
        print(f'TRAINING: {key}')
        print(f'{"="*60}')

        try:
            env = OvercookedHiddenPartner(
                layout_name='cramped_room', mode=mode, horizon=400,
            )

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
            model.learn(total_timesteps=TIMESTEPS, callback=callback)
            elapsed = time.time() - t0

            model.save(final_path)
            env.close()
            results_log[key] = {'status': 'ok', 'time_s': elapsed, 'steps': TIMESTEPS}
            print(f'  COMPLETED in {elapsed:.0f}s')
            sys.stdout.flush()

        except Exception as e:
            results_log[key] = {'status': 'error', 'error': str(e), 'trace': traceback.format_exc()}
            print(f'  ERROR: {e}')
            sys.stdout.flush()

# Save summary
summary_path = os.path.join(MODEL_DIR, '_batch_summary.json')
with open(summary_path, 'w') as f:
    json.dump({k: {kk: str(vv) if not isinstance(vv, (int, float, dict)) else vv
                    for kk, vv in v.items()}
               for k, v in results_log.items()}, f, indent=2)

print(f'\n{"="*60}')
print('BATCH TRAINING COMPLETE')
print(f'{"="*60}')
for k, v in results_log.items():
    print(f'  {k}: {v["status"]}')
