"""Updated stability map with all 5 SINGLE seeds + DYNAMIC + STATIC."""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

data = {
    'SINGLE': [
        (0.261, 7.6, 41),
        (0.384, 6.5, 51),
        (0.486, 23.4, 52),
        (0.337, 4.6, 53),
        (0.352, 7.3, 54),
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

fig, ax = plt.subplots(figsize=(7.5, 5.5))

for mode, pts in data.items():
    paths = [p[0] for p in pts]
    curvs = [p[1] for p in pts]
    ax.scatter(paths, curvs, s=120, c=colors[mode], alpha=0.8,
               edgecolors='black', linewidth=0.5, label=mode, zorder=3)
    for pl, cv, seed in pts:
        ax.annotate(f's{seed}', (pl, cv), textcoords="offset points",
                     xytext=(6, 4), fontsize=8)

    # Mean crosshairs
    p_mean = np.mean(paths)
    c_mean = np.mean(curvs)
    ax.scatter(p_mean, c_mean, marker='x', s=200, c=colors[mode],
               linewidths=3, zorder=4, label=f'{mode} mean')

ax.set_xlabel('Path Length (cumulative L2)', fontsize=12)
ax.set_ylabel('Curvature Ratio (path / endpoint)', fontsize=12)
ax.set_title('Training Stability Map (All Seeds)', fontsize=14)
# Only show one legend entry per mode
handles, labels = ax.get_legend_handles_labels()
by_label = {}
for h, l in zip(handles, labels):
    if not l.endswith('mean'):
        by_label[l] = h
ax.legend(by_label.values(), by_label.keys(), fontsize=10)
ax.set_xlim(0, 0.55)
ax.set_ylim(0, 28)
fig.tight_layout()
fig.savefig('../paper/figures/fig_stability_map.pdf', dpi=300)
fig.savefig('../paper/figures/fig_stability_map.png', dpi=200)
print('Saved fig_stability_map (updated)')

# Stats table
print('\n=== Mode Statistics ===')
for mode in ['SINGLE', 'STATIC', 'DYNAMIC']:
    pts = data[mode]
    paths = [p[0] for p in pts]
    curvs = [p[1] for p in pts]
    print(f'{mode:8} (n={len(pts)}): path={np.mean(paths):.3f}±{np.std(paths):.3f}  '
          f'curv={np.median(curvs):.1f}x (median)  range=[{np.min(curvs):.1f}-{np.max(curvs):.1f}]x')

print('\n=== Cross-mode comparison ===')
sing_curv = sorted([p[1] for p in data['SINGLE']])
dyn_curv = sorted([p[1] for p in data['DYNAMIC']])
print(f'SINGLE curvatures: {sing_curv}')
print(f'DYNAMIC curvatures: {dyn_curv}')
print(f'Seeds with curv > 15: SINGLE={sum(1 for c in sing_curv if c>15)} DYNAMIC={sum(1 for c in dyn_curv if c>15)}')
print(f'Seeds with curv < 5:  SINGLE={sum(1 for c in sing_curv if c<5)} DYNAMIC={sum(1 for c in dyn_curv if c<5)}')
