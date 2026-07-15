"""Final stability map with all 14 seeds."""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

data = {
    'SINGLE': [
        (0.261, 7.6, 41), (0.384, 6.5, 51), (0.486, 23.4, 52),
        (0.337, 4.6, 53), (0.352, 7.3, 54),
    ],
    'STATIC': [
        (0.145, 2.8, 41), (0.253, 3.1, 61), (0.421, 7.2, 62),
        (0.244, 5.3, 63), (0.203, 11.5, 64),
    ],
    'DYNAMIC': [
        (0.198, 12.7, 41), (0.151, 1.5, 42),
        (0.323, 18.9, 43), (0.262, 4.1, 44),
    ],
}
colors = {'SINGLE': '#2196F3', 'STATIC': '#4CAF50', 'DYNAMIC': '#FF9800'}

fig, ax = plt.subplots(figsize=(7.5, 5.5))

for mode, pts in data.items():
    paths = [p[0] for p in pts]
    curvs = [p[1] for p in pts]
    ax.scatter(paths, curvs, s=100, c=colors[mode], alpha=0.8,
               edgecolors='black', linewidth=0.5, label=mode, zorder=3)
    for pl, cv, seed in pts:
        ax.annotate(f's{seed}', (pl, cv), textcoords="offset points",
                     xytext=(5, 3), fontsize=7)
    p_mean = np.mean(paths)
    c_mean = np.mean(curvs)
    ax.scatter(p_mean, c_mean, marker='X', s=150, c=colors[mode],
               edgecolors='black', linewidths=1.5, zorder=4)

# Stats annotation
ax.text(0.02, 26, f'SINGLE mean: path=0.364 curv=9.9x (n=5)', fontsize=8, color=colors['SINGLE'])
ax.text(0.02, 24, f'STATIC mean: path=0.253 curv=6.0x (n=5)', fontsize=8, color=colors['STATIC'])
ax.text(0.02, 22, f'DYNAMIC mean: path=0.234 curv=9.3x (n=4)', fontsize=8, color=colors['DYNAMIC'])

ax.set_xlabel('Path Length (cumulative L2)', fontsize=12)
ax.set_ylabel('Curvature Ratio (path / endpoint)', fontsize=12)
ax.set_title('Training Stability Across Cooperation Structures (All 14 Seeds)', fontsize=13)
ax.legend(fontsize=10, loc='lower right')
ax.set_xlim(0, 0.55)
ax.set_ylim(0, 28)
fig.tight_layout()
fig.savefig('../paper/figures/fig_stability_map.pdf', dpi=300)
fig.savefig('../paper/figures/fig_stability_map.png', dpi=200)
print('Saved fig_stability_map (final)')

# Print stats
print('\n=== Final Stats ===')
for mode in ['SINGLE', 'STATIC', 'DYNAMIC']:
    pts = data[mode]
    paths = [p[0] for p in pts]
    curvs = [p[1] for p in pts]
    print(f'{mode:8} (n={len(pts)}): path={np.mean(paths):.3f}+-{np.std(paths):.3f}  '
          f'curv median={np.median(curvs):.1f}x  range=[{min(curvs):.1f}-{max(curvs):.1f}]x')
