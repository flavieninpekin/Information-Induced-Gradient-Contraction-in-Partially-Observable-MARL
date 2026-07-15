"""Generate scatter plot: path length vs curvature for all seeds."""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Data from compute_path.py (excluding incomplete SINGLE s42)
data = {
    'SINGLE': [
        (0.261, 7.6, 41),
    ],
    'STATIC': [
        (0.145, 2.8, 41),
    ],
    'DYNAMIC': [
        (0.198, 12.7, 41),
        (0.151, 1.5, 42),
        (0.323, 18.9, 43),
        (0.262, 4.1, 44),
    ],
}
colors = {'SINGLE': '#2196F3', 'STATIC': '#4CAF50', 'DYNAMIC': '#FF9800'}

fig, ax = plt.subplots(figsize=(7, 5))

for mode, points in data.items():
    paths = [p[0] for p in points]
    curvs = [p[1] for p in points]
    ax.scatter(paths, curvs, s=120, c=colors[mode], alpha=0.8,
               edgecolors='black', linewidth=0.5, label=mode, zorder=3)
    for pl, cv, seed in points:
        ax.annotate(f's{seed}', (pl, cv), textcoords="offset points",
                     xytext=(6, 4), fontsize=9)

# Add mean crosshairs
for mode, pts in data.items():
    p_mean = np.mean([p[0] for p in pts])
    c_mean = np.mean([p[1] for p in pts])
    ax.axvline(x=p_mean, color=colors[mode], linestyle='--', alpha=0.3, linewidth=1)
    ax.axhline(y=c_mean, color=colors[mode], linestyle='--', alpha=0.3, linewidth=1)
    ax.scatter(p_mean, c_mean, marker='x', s=200, c=colors[mode], linewidths=2, zorder=4)

# Danger zone: high path AND high curvature
ax.axvspan(0.25, 0.40, alpha=0.1, color='red')
ax.axhspan(10, 25, alpha=0.1, color='red')
ax.text(0.33, 22, 'Instability Zone', fontsize=10, color='red', ha='center', fontstyle='italic')

ax.set_xlabel('Path Length (cumulative L2)', fontsize=12)
ax.set_ylabel('Curvature Ratio (path / endpoint)', fontsize=12)
ax.set_title('Training Stability Map', fontsize=14)
ax.legend(fontsize=11)
ax.set_xlim(0, 0.4)
ax.set_ylim(0, 25)
fig.tight_layout()
fig.savefig('../paper/figures/fig_stability_map.pdf', dpi=300)
fig.savefig('../paper/figures/fig_stability_map.png', dpi=200)
print('Saved fig_stability_map')

# Also print a clear table for the paper
print('\n=== For Paper Table ===')
print(f'{"Seed":<8} {"Mode":<8} {"Path":>8} {"Endpt":>8} {"Curv":>8} {"Region":>12}')
for mode, pts in data.items():
    for pl, cv, seed in pts:
        region = 'Stable' if cv < 5 else 'HighPath' if pl < 0.25 else 'Instability'
        print(f's{seed:<7} {mode:<8} {pl:>8.3f} {pl/cv if cv > 0 else 0:>8.3f} {cv:>8.1f}x {region:>12}')

print('\nKey observation:')
dy_curvs = [p[1] for p in data['DYNAMIC']]
all_other = [p[1] for p in data['SINGLE'] + data['STATIC']]
print(f'  DYNAMIC median curvature: {np.median(dy_curvs):.1f}x')
print(f'  Other modes median: {np.median(all_other):.1f}x')
print(f'  DYNAMIC seeds in instability zone (curv>10): {sum(1 for c in dy_curvs if c>10)}/4')
print(f'  Other seeds in instability zone: {sum(1 for c in all_other if c>10)}/2')
print(f'  DYNAMIC standard deviation: {np.std(dy_curvs):.1f}x')
print(f'  Other modes std: {np.std(all_other):.1f}x')
