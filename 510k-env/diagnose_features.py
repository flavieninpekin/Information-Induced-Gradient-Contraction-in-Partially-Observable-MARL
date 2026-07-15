"""Feature collinearity diagnosis on all trajectories."""
import json, os, gzip, pickle
import numpy as np

FEATURE_NAMES = ['MyScore', 'MyHandSize', 'MyStrength', 'TrickScore', 'PassCount']

# Collect features from all trajectory files
trajectory_dirs = [
    ('transfer_data', '510k_single_final_single'),
    ('transfer_data_pi_static', '510k_static_1818624_steps_static'),
    ('transfer_data_pi_dynamic', '510k_dynamic_final_dynamic'),
]

all_features = []
sources = []

for traj_dir, prefix in trajectory_dirs:
    traj_path = os.path.join(traj_dir, f'{prefix}_trajectories.pkl.gz')
    try:
        with gzip.open(traj_path, 'rb') as f:
            trajectories = pickle.load(f)
    except FileNotFoundError:
        print(f'Missing: {traj_path}')
        continue

    n_eps = len(trajectories)
    for ep_data in trajectories:
        for pid, traj in ep_data['trajectories'].items():
            for entry in traj:
                all_features.append(entry['features'])
                sources.append(f'{prefix}_p{pid}')

    print(f'{traj_dir}/{prefix}: {n_eps} eps, {len(all_features)} total state samples')

all_features = np.array(all_features, dtype=np.float32)
print(f'\nTotal samples: {all_features.shape[0]} × {all_features.shape[1]}')

# ============================================================
# 1. Correlation matrix
# ============================================================
corr = np.corrcoef(all_features.T)
print('\n=== Feature Correlation Matrix ===')
print(f'{"":<14}', end='')
for name in FEATURE_NAMES:
    print(f'{name[:8]:>8}', end='')
print()
for i, name_i in enumerate(FEATURE_NAMES):
    print(f'{name_i:<14}', end='')
    for j in range(len(FEATURE_NAMES)):
        print(f'{corr[i,j]:>8.3f}', end='')
    print()

# Flag high correlations
print('\n=== High Correlation Pairs (|r| > 0.5) ===')
flagged = False
for i in range(len(FEATURE_NAMES)):
    for j in range(i+1, len(FEATURE_NAMES)):
        if abs(corr[i,j]) > 0.5:
            print(f'  {FEATURE_NAMES[i]} vs {FEATURE_NAMES[j]}: r = {corr[i,j]:.3f}')
            flagged = True
if not flagged:
    print('  None found. Features are well-separated.')

# ============================================================
# 2. Variance Inflation Factor (VIF)
# ============================================================
print('\n=== Variance Inflation Factors (VIF) ===')
# VIF = 1 / (1 - R²) where R² is from regressing feature i on all others
for i in range(len(FEATURE_NAMES)):
    X = np.delete(all_features, i, axis=1)
    y = all_features[:, i]
    # Linear regression using normal equation
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
        beta = XtX_inv @ (X.T @ y)
        y_pred = X @ beta
        residuals = y - y_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-10)
        vif = 1.0 / (1.0 - r2) if r2 < 1.0 else float('inf')
    except np.linalg.LinAlgError:
        vif = float('inf')

    print(f'  {FEATURE_NAMES[i]:<14}: VIF = {vif:.2f}  (VIF > 5 indicates multicollinearity)')

# ============================================================
# 3. Mode-wise distribution statistics
# ============================================================
print('\n=== Feature Distributions by Policy ===')
mode_summaries = {
    'π_single': json.load(open('transfer_data/510k_single_final_single_summary.json')),
    'π_static': json.load(open('transfer_data_pi_static/510k_static_1818624_steps_static_summary.json')),
    'π_dynamic': json.load(open('transfer_data_pi_dynamic/510k_dynamic_final_dynamic_summary.json')),
}
rand_data = json.load(open('transfer_data/random_baseline_features.json'))

print(f'{"Feature":<14} {"π_single":<8} {"π_static":<8} {"π_dynamic":<8} {"Random":<8} {"Range":<8}')
print('-' * 46)
for i, name in enumerate(FEATURE_NAMES):
    vals = [m['feature_expectations'][i] for m in mode_summaries.values()]
    r = rand_data['feature_expectations'][i]
    rng = max(vals) - min(vals)
    print(f'{name:<14} {vals[0]:<8.3f} {vals[1]:<8.3f} {vals[2]:<8.3f} {r:<8.3f} {rng:<8.3f}')
    if rng < 0.02:
                print(f'  [WARN] {name} has very low variance across policies (< 0.02)')

print('\n=== Summary ===')
if not flagged:
    print('OK: No high feature correlations (>0.5) -- features are sufficiently independent.')
else:
    print('[WARN] High correlations detected — consider merging or removing redundant features.')
