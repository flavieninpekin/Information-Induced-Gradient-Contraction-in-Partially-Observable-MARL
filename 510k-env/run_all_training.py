"""
Master training script: runs all 3 modes × 6 seeds sequentially.
Output goes to train_all.log for monitoring.
"""
import sys, os, time, json, subprocess
from datetime import datetime

MODES = ['single', 'static', 'dynamic']
N_SEEDS = 6
TIMESTEPS = 1_000_000
SAVE_EVERY = 100_000
LOG_FILE = 'train_all.log'

def log(msg):
    t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{t}] {msg}'
    print(line)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')

total_start = time.time()
log(f'Starting master training: {len(MODES)} modes × {N_SEEDS} seeds × {TIMESTEPS} steps')

for mode in MODES:
    log(f'=== Mode: {mode} ===')
    cmd = [
        sys.executable, 'train_selfplay.py',
        '--mode', mode,
        '--seeds', str(N_SEEDS),
        '--timesteps', str(TIMESTEPS),
        '--save-every', str(SAVE_EVERY),
    ]
    log(f'Running: {" ".join(cmd)}')
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    log(f'Stdout:\n{result.stdout[-2000:]}')
    if result.stderr:
        severity = 'ERROR' if result.returncode != 0 else 'WARN'
        log(f'[{severity}] Stderr:\n{result.stderr[-2000:]}')
    log(f'Mode {mode} done in {elapsed/60:.1f} min (returncode={result.returncode})')

total_elapsed = time.time() - total_start
log(f'All done! Total time: {total_elapsed/60:.1f} min ({total_elapsed/3600:.1f} hours)')
