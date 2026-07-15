"""Evaluate all self-play trained policies and compare."""
import warnings, json, os, time, numpy as np
warnings.filterwarnings('ignore')
from transfer import transfer_evaluate

MODEL_DIR = 'models_selfplay'
OUTPUT_DIR = 'sp_eval'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Models to evaluate
models = [
    ('single', '510k_single_seed41_final.zip', 'single'),
    ('single_best', '510k_single_seed42_300000_steps.zip', 'single'),  # best available
    ('static', '510k_static_seed41_800000_steps.zip', 'static'),
    ('dynamic_s41', '510k_dynamic_seed41_final.zip', 'dynamic'),
    ('dynamic_s42', '510k_dynamic_seed42_final.zip', 'dynamic'),
    ('dynamic_s43', '510k_dynamic_seed43_final.zip', 'dynamic'),
    ('dynamic_s44', '510k_dynamic_seed44_final.zip', 'dynamic'),
    ('dynamic_s45', '510k_dynamic_seed45_final.zip', 'dynamic'),
    ('dynamic_s46', '510k_dynamic_seed46_final.zip', 'dynamic'),
]

results = {}
for label, model_file, mode in models:
    path = os.path.join(MODEL_DIR, model_file)
    if not os.path.exists(path):
        print(f'SKIP (missing): {path}')
        continue
    print(f'Evaluating {label} ({mode})...', flush=True)
    summary = transfer_evaluate(path, mode, n_episodes=300, output_dir=OUTPUT_DIR)
    fe = summary['feature_expectations']
    results[label] = {'mode': mode, 'fe': fe, 'rewards': summary['avg_rewards_per_player']}
    print(f'  FE: {[round(v,4) for v in fe]}', flush=True)

# Print comparison
FEATURE_NAMES = ['MyScore', 'MyHandSize', 'MyStrength', 'TrickScore', 'PassCount']

print('\n' + '=' * 80)
print('SELF-PLAY TRAINED POLICIES — FEATURE EXPECTATIONS')
print('=' * 80)
header = f'{"Policy":<18} {"Mode":<8} ' + ' '.join(f'{n:>10}' for n in FEATURE_NAMES)
print(header)
print('-' * len(header))
for label, data in sorted(results.items()):
    fe = data['fe']
    print(f'{label:<18} {data["mode"]:<8} ' + ' '.join(f'{v:10.4f}' for v in fe))

# DYNAMIC seed spread
dyn_fe = [results[f'dynamic_s4{s}']['fe'] for s in range(1,7) if f'dynamic_s4{s}' in results]
if len(dyn_fe) >= 2:
    dyn_fe = np.array(dyn_fe)
    print(f'\nDYNAMIC seed spread (mean ± std):')
    for i, name in enumerate(FEATURE_NAMES):
        print(f'  {name:<14}: {np.mean(dyn_fe[:,i]):.4f} ± {np.std(dyn_fe[:,i]):.4f}')

# Pairwise distances
print('\n' + '=' * 80)
print('PAIRWISE L2 DISTANCES')
print('=' * 80)
keys = sorted(results.keys())
for i, k1 in enumerate(keys):
    for k2 in keys[i+1:]:
        d = np.linalg.norm(np.array(results[k1]['fe']) - np.array(results[k2]['fe']))
        print(f'  {k1:<18} vs {k2:<18}: {d:.4f}')
