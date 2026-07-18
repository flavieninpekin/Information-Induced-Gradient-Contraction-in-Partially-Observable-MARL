"""Compute per-step variance from checkpoint data to support gradient-variance claim."""
import json, numpy as np

with open('path_data/all_paths_7feat.json') as f:
    data = json.load(f)

# For each seed, compute per-checkpoint step sizes and their variance
modes = {'single': [], 'static': [], 'dynamic': []}
all_step_vars = {'single': [], 'static': [], 'dynamic': []}

for key, v in data.items():
    mode = key.split('_')[0]
    if mode not in modes:
        continue
    traj = np.array(v['trajectory'])
    if len(traj) < 3:
        continue
    # Per-checkpoint step sizes (L2 between consecutive checkpoints)
    steps = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    step_var = np.var(steps)
    step_mean = np.mean(steps)
    all_step_vars[mode].append(step_var)
    print(f'{mode} {key}: mean_step={step_mean:.4f} var_step={step_var:.6f}')

print('\n=== Step-size variance by mode ===')
for mode in ['single', 'static', 'dynamic']:
    vals = all_step_vars[mode]
    print(f'{mode:8} (n={len(vals)}): mean={np.mean(vals):.6f} std={np.std(vals):.6f} '
          f'median={np.median(vals):.6f} range=[{min(vals):.6f}, {max(vals):.6f}]')
