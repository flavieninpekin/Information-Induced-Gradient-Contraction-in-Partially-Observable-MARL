"""P0: Run transfer evaluation for all 3 modes + random baseline + IRL."""
import sys, warnings, json, time
warnings.filterwarnings('ignore')
from transfer import transfer_evaluate
from irl import compute_random_baseline_features, run_full_irl_pipeline

model_path = '../models/510k_single_final.zip'

for mode in ['single', 'static', 'dynamic']:
    t0 = time.time()
    summary = transfer_evaluate(model_path, mode, n_episodes=500, output_dir='transfer_data')
    t1 = time.time()
    fe_str = json.dumps([round(v, 3) for v in summary['feature_expectations']])
    print(f'[{mode}] {t1-t0:.0f}s  FE={fe_str}')

print('\n=== Random baseline ===')
compute_random_baseline_features(n_episodes=500, output_dir='transfer_data')

print('\n=== IRL ===')
results = run_full_irl_pipeline(transfer_dir='transfer_data', use_tabular=False)
print('\n=== Final weights ===')
for mode, w in results['contrastive_weights'].items():
    w_str = ', '.join([f'{v:.3f}' for v in w])
    print(f'{mode}: [{w_str}]')
