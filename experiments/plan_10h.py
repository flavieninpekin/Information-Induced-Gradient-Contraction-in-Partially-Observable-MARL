"""
10-HOUR MASTER RUN — executes all missing experiments + consolidates results.

What this runs:
  1. 510K PPO selfplay (single/static/dynamic, 8 seeds)       ~2h
  2. 510K A2C STATIC (8 seeds)                                  ~30min
  3. 510K DQN STATIC (8 seeds)                                  ~30min
  4. Overcooked PPO (static/dynamic, 8 seeds)                   ~1h
  5. Overcooked A2C (static/dynamic, 8 seeds)                   ~1h
  6. SAC 510K extended (more seeds + patched patterns)           ~3h
  7. Kappa + path integral consolidation                         ~30min
  Total est: ~8.5h

Output:
  models_*/          — trained models with checkpoints
  *_kappa_*/results.json — per-experiment kappa
  final_results/all_kappa.json — master result table
"""
import os, sys, subprocess, time, json

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(PROJECT, 'final_results', 'run_log.txt')
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def log(msg):
    t = time.strftime('%H:%M:%S')
    line = f'[{t}] {msg}'
    print(line); sys.stdout.flush()
    with open(LOG_FILE, 'a') as f: f.write(line + '\n')

def run(name, cmd, est_min=30):
    log(f'START {name} (est {est_min}min): {cmd}')
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, cwd=PROJECT)
    e = (time.time()-t0)/60
    ok = 'OK' if r.returncode == 0 else f'FAIL({r.returncode})'
    log(f'  {ok} in {e:.1f}min')
    return r.returncode == 0

# ============ PLAN ============
STEPS = [
    ('510K_PPO_SINGLE',  'python 510k-env/train_selfplay.py --mode single  --seeds 8 --timesteps 1000000', 40),
    ('510K_PPO_STATIC',  'python 510k-env/train_selfplay.py --mode static  --seeds 8 --timesteps 1000000', 40),
    ('510K_PPO_DYNAMIC', 'python 510k-env/train_selfplay.py --mode dynamic --seeds 8 --timesteps 1000000', 40),
    ('510K_A2C_STATIC',  'python 510k-env/train_510k_a2c_static.py', 40),
    ('510K_DQN_STATIC',  'python 510k-env/train_510k_dqn_static.py', 40),
    ('OVERCOOKED_PPO',   'python overcooked_adapt/train_v3.py', 60),
    ('OVERCOOKED_A2C',   'python overcooked_adapt/train_overcooked_a2c.py', 60),
    ('510K_SAC_EXTRA',   'python 510k-env/train_510k_sac_all.py', 180),
    ('KAPPA_ALL',        'python experiments/collect_all_kappa.py', 10),
]

if __name__ == '__main__':
    log('=' * 60)
    log('10-HOUR MASTER RUN START')
    log(f'Total {len(STEPS)} steps')
    log('=' * 60)

    total_t0 = time.time()
    ok = 0; fail = 0

    for name, cmd, est in STEPS:
        if run(name, cmd, est): ok += 1
        else: fail += 1
        elapsed = (time.time() - total_t0) / 60
        if elapsed > 540:
            log(f'BUDGET EXCEEDED ({elapsed:.0f}min). Finishing early.')
            break

    log(f'\nCOMPLETE: {ok} OK, {fail} FAILED, {(time.time()-total_t0)/60:.1f}min total')
