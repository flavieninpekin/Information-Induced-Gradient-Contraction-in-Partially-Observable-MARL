import numpy as np
from scipy import stats

single = [0.340, 0.482, 0.561, 0.454, 0.443]
static = [0.253, 0.342, 0.490, 0.281, 0.278]
dynamic = [0.299, 0.231, 0.367, 0.274]

print('=== Descriptive ===')
for name, vals in [('SINGLE',single),('STATIC',static),('DYNAMIC',dynamic)]:
    print(f'{name}: mean={np.mean(vals):.3f} std={np.std(vals):.3f} '
          f'median={np.median(vals):.3f} range={np.min(vals):.3f}-{np.max(vals):.3f}')

print('\n=== Mann-Whitney U (one-sided) ===')
for a, b, label in [(single, dynamic, 'SINGLE > DYNAMIC'),
                     (single, static, 'SINGLE > STATIC'),
                     (static, dynamic, 'STATIC > DYNAMIC')]:
    u, p = stats.mannwhitneyu(a, b, alternative='greater')
    sig = "YES" if p < 0.05 else "NO"
    print(f'{label}: U={u:.0f} p={p:.4f} significant={sig}')

print('\n=== Cohens d ===')
for a, b, label in [(single, dynamic, 'SINGLE vs DYNAMIC'),
                     (single, static, 'SINGLE vs STATIC')]:
    pooled_std = np.sqrt((np.std(a)**2 + np.std(b)**2) / 2)
    d_val = (np.mean(a) - np.mean(b)) / pooled_std
    print(f'{label}: d={d_val:.2f}')

print('\n=== Monotonic trend ===')
all_vals = single + static + dynamic
all_groups = [0]*len(single) + [1]*len(static) + [2]*len(dynamic)
rho, p = stats.spearmanr(all_groups, all_vals)
print(f'Spearman r={rho:.3f} p={p:.4f}')
print(f'Increasing cooperation -> decreasing path length: '
      f'{"CONFIRMED (p<0.05)" if p < 0.05 and rho < 0 else "NOT SIGNIFICANT"}')

# Bootstrap 95% CI for SINGLE-DYNAMIC difference
np.random.seed(42)
diffs = []
for _ in range(10000):
    a_sample = np.random.choice(single, len(single), replace=True)
    b_sample = np.random.choice(dynamic, len(dynamic), replace=True)
    diffs.append(np.mean(a_sample) - np.mean(b_sample))
ci = np.percentile(diffs, [2.5, 97.5])
print(f'\nSINGLE-DYNAMIC diff (bootstrap 95% CI): {ci[0]:.3f} to {ci[1]:.3f}')
print(f'Effect size: {(np.mean(single)-np.mean(dynamic)):.3f} ({ci[0]:.3f}, {ci[1]:.3f})')
