"""Compute path integrals from logged checkpoint evaluation data."""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Data extracted from path_integral.py run output
data = {
    ('single', 41): [
        (100000,  [0.0679, 0.3653, 0.4754, 0.0558, 0.2101]),
        (200000,  [0.0634, 0.3307, 0.5135, 0.0581, 0.2127]),
        (300000,  [0.0632, 0.3241, 0.5122, 0.0544, 0.2135]),
        (400000,  [0.0667, 0.3734, 0.4984, 0.0669, 0.1984]),
        (500000,  [0.0663, 0.3692, 0.5000, 0.0644, 0.1956]),
        (600000,  [0.0686, 0.3668, 0.5017, 0.0599, 0.2062]),
        (700000,  [0.0718, 0.3934, 0.4879, 0.0676, 0.1953]),
        (800000,  [0.0711, 0.3613, 0.4692, 0.0505, 0.2180]),
        (900000,  [0.0679, 0.3689, 0.4823, 0.0573, 0.2066]),
        (1000000, [0.0700, 0.3913, 0.4825, 0.0645, 0.1949]),
        ('final',  [0.0707, 0.3928, 0.4819, 0.0653, 0.1934]),
    ],
    ('single', 42): [
        (100000,  [0.0699, 0.3753, 0.4834, 0.0568, 0.1972]),
        (200000,  [0.0715, 0.3742, 0.4825, 0.0548, 0.2108]),
        (300000,  [0.0730, 0.3821, 0.4802, 0.0564, 0.2101]),
    ],
    ('static', 41): [
        (100000,  [0.0561, 0.3140, 0.5167, 0.0344, 0.1712]),
        (200000,  [0.0564, 0.3215, 0.5259, 0.0360, 0.1709]),
        (300000,  [0.0576, 0.3317, 0.5199, 0.0385, 0.1728]),
        (400000,  [0.0585, 0.3479, 0.5198, 0.0400, 0.1716]),
        (500000,  [0.0599, 0.3538, 0.5112, 0.0407, 0.1754]),
        (600000,  [0.0704, 0.4001, 0.4996, 0.0466, 0.1736]),
        (700000,  [0.0678, 0.3789, 0.4939, 0.0436, 0.1770]),
        (800000,  [0.0633, 0.3634, 0.5073, 0.0415, 0.1747]),
    ],
    ('dynamic', 41): [
        (100000,  [0.0554, 0.3054, 0.4884, 0.0264, 0.1939]),
        (200000,  [0.0642, 0.3637, 0.4727, 0.0370, 0.1955]),
        (300000,  [0.0640, 0.3552, 0.4651, 0.0343, 0.1920]),
        (400000,  [0.0599, 0.3300, 0.4804, 0.0314, 0.1909]),
        (500000,  [0.0585, 0.3240, 0.4869, 0.0302, 0.1890]),
        (600000,  [0.0576, 0.3185, 0.4838, 0.0287, 0.1907]),
        (700000,  [0.0598, 0.3369, 0.4724, 0.0300, 0.1911]),
        (800000,  [0.0591, 0.3382, 0.4813, 0.0315, 0.1885]),
        (900000,  [0.0605, 0.3350, 0.4866, 0.0328, 0.1910]),
        (1000000, [0.0530, 0.3016, 0.5016, 0.0293, 0.1915]),
        ('final',  [0.0524, 0.3004, 0.5020, 0.0302, 0.1912]),
    ],
    ('dynamic', 42): [
        (100000,  [0.0413, 0.2280, 0.5231, 0.0256, 0.1776]),
        (200000,  [0.0418, 0.2296, 0.5358, 0.0248, 0.1791]),
        (300000,  [0.0532, 0.2879, 0.5361, 0.0336, 0.1863]),
        (400000,  [0.0529, 0.2877, 0.5375, 0.0338, 0.1892]),
        (500000,  [0.0552, 0.2995, 0.5253, 0.0342, 0.1872]),
        (600000,  [0.0588, 0.3240, 0.5220, 0.0383, 0.1876]),
        (700000,  [0.0598, 0.3285, 0.5249, 0.0382, 0.1869]),
        (800000,  [0.0594, 0.3294, 0.5229, 0.0380, 0.1869]),
        (900000,  [0.0579, 0.3213, 0.5238, 0.0363, 0.1916]),
        (1000000, [0.0598, 0.3299, 0.5178, 0.0382, 0.1870]),
        ('final',  [0.0589, 0.3283, 0.5194, 0.0380, 0.1852]),
    ],
    ('dynamic', 43): [
        (100000,  [0.0553, 0.3153, 0.4998, 0.0357, 0.1726]),
        (200000,  [0.0503, 0.2852, 0.5183, 0.0329, 0.1723]),
        (300000,  [0.0564, 0.3232, 0.5100, 0.0405, 0.1706]),
        (400000,  [0.0598, 0.3549, 0.4999, 0.0476, 0.1697]),
        (500000,  [0.0519, 0.2930, 0.5203, 0.0351, 0.1740]),
        (600000,  [0.0449, 0.2619, 0.5304, 0.0282, 0.1747]),
        (700000,  [0.0532, 0.3046, 0.5209, 0.0354, 0.1815]),
        (800000,  [0.0497, 0.2810, 0.5262, 0.0321, 0.1775]),
        (900000,  [0.0509, 0.2917, 0.5243, 0.0359, 0.1775]),
        (1000000, [0.0534, 0.3157, 0.5172, 0.0398, 0.1756]),
        ('final',  [0.0539, 0.3123, 0.5162, 0.0388, 0.1744]),
    ],
    ('dynamic', 44): [
        (100000,  [0.0425, 0.2450, 0.5387, 0.0273, 0.1738]),
        (200000,  [0.0450, 0.2591, 0.5326, 0.0289, 0.1730]),
        (300000,  [0.0477, 0.2739, 0.5347, 0.0298, 0.1769]),
        (400000,  [0.0464, 0.2685, 0.5235, 0.0290, 0.1798]),
        (500000,  [0.0573, 0.3419, 0.5024, 0.0393, 0.1747]),
        (600000,  [0.0552, 0.3456, 0.5046, 0.0395, 0.1718]),
        (700000,  [0.0481, 0.2786, 0.5187, 0.0315, 0.1807]),
        (800000,  [0.0499, 0.3117, 0.5199, 0.0350, 0.1779]),
        (900000,  [0.0486, 0.2955, 0.5251, 0.0321, 0.1761]),
        (1000000, [0.0521, 0.3041, 0.5205, 0.0350, 0.1762]),
        ('final',  [0.0520, 0.3048, 0.5182, 0.0343, 0.1754]),
    ],
}

FEATURE_NAMES = ['MyScore', 'MyHandSize', 'MyStrength', 'TrickScore', 'PassCount']

# ============================================================
print('=' * 65)
print('PATH INTEGRAL ANALYSIS')
print('=' * 65)

all_results = {}
for (mode, seed), entries in data.items():
    mus = np.array([e[1] for e in entries])
    steps = [e[0] for e in entries]

    if len(mus) < 2:
        continue

    # Path length
    path_len = np.sum(np.linalg.norm(np.diff(mus, axis=0), axis=1))
    # Endpoint distance
    endpoint_dist = np.linalg.norm(mus[-1] - mus[0])
    # Curvature
    curvature = path_len / max(endpoint_dist, 1e-6)
    # Directness: how straight was the path? 1.0 = perfectly straight
    directness = endpoint_dist / max(path_len, 1e-6)

    all_results[(mode, seed)] = {
        'path_len': path_len, 'endpoint_dist': endpoint_dist,
        'curvature': curvature, 'directness': directness,
    }
    mark = '***' if curvature > 5 else ''
    print(f'  {mode:8} s{seed}: path={path_len:.4f}  endpt={endpoint_dist:.4f}  '
          f'curve={curvature:.1f}x  direct={directness:.3f}  {mark}')

# Aggregate by mode
print('\n--- By Mode ---')
for mode in ['single', 'static', 'dynamic']:
    vals = [v for k, v in all_results.items() if k[0] == mode]
    if vals:
        paths = [v['path_len'] for v in vals]
        endps = [v['endpoint_dist'] for v in vals]
        curvs = [v['curvature'] for v in vals]
        print(f'  {mode:8}: path={np.mean(paths):.4f}±{np.std(paths):.4f}  '
              f'endpt={np.mean(endps):.4f}±{np.std(endps):.4f}  '
              f'curve={np.mean(curvs):.1f}x  (n={len(vals)})')

# ============================================================
# Visualization
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(5.5, 9), sharex=True, sharey=True)
colors_map = {'single': '#2196F3', 'static': '#4CAF50', 'dynamic': '#FF9800'}
titles = {'single': 'SINGLE', 'static': 'STATIC', 'dynamic': 'DYNAMIC'}

all_x = []; all_y = []
for (m, seed), entries in data.items():
    mus = np.array([e[1] for e in entries])
    if len(mus) >= 2:
        all_x.extend(mus[:, 1].tolist())
        all_y.extend(mus[:, 2].tolist())
x_min, x_max = min(all_x), max(all_x)
y_min, y_max = min(all_y), max(all_y)
x_margin = (x_max - x_min) * 0.08 or 0.02
y_margin = (y_max - y_min) * 0.08 or 0.02

for ax_idx, mode in enumerate(['single', 'static', 'dynamic']):
    ax = axes[ax_idx]
    for (m, seed), entries in data.items():
        if m != mode:
            continue
        mus = np.array([e[1] for e in entries])
        if len(mus) < 2:
            continue
        x = mus[:, 1]  # MyHandSize
        y = mus[:, 2]  # MyStrength
        ax.plot(x, y, 'o-', alpha=0.7, markersize=4, linewidth=1.2, label=f's{seed}',
                color=colors_map[mode] if len([k for k in data if k[0]==mode]) == 1 else None)
        ax.scatter(x[0], y[0], s=80, marker='s', zorder=5, edgecolors='black', linewidth=0.5,
                    color=colors_map[mode])
        ax.scatter(x[-1], y[-1], s=100, marker='*', zorder=5, edgecolors='black', linewidth=0.8,
                    color=colors_map[mode])
    ax.set_title(titles[mode], fontsize=12, fontweight='bold', pad=6)
    if ax_idx == 2:
        ax.set_xlabel('MyHandSize', fontsize=10)
    ax.set_ylabel('MyStrength', fontsize=10)
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.legend(fontsize=7, ncol=2, loc='lower right')

fig.suptitle('Training Trajectories in Feature Space  (square=start, star=endpoint)',
             fontsize=11, y=1.01)
fig.tight_layout()
fig.savefig('../paper/figures/fig_trajectories.png', dpi=200)
fig.savefig('../paper/figures/fig_trajectories.pdf', dpi=300)
print('\nSaved: fig_trajectories.png/pdf')

# Path length bar chart
fig2, ax2 = plt.subplots(figsize=(7, 4))
x = np.arange(3)
w = 0.25
for i, mode in enumerate(['single', 'static', 'dynamic']):
    vals = [v for k, v in all_results.items() if k[0] == mode]
    if vals:
        paths = np.array([v['path_len'] for v in vals])
        endps = np.array([v['endpoint_dist'] for v in vals])
        # Path length bars
        ax2.bar(x[i] - w/2, np.mean(paths), w, yerr=np.std(paths) if len(paths)>1 else 0,
                color=colors_map[mode], alpha=0.7, label=f'{titles[mode]} path')
        # Endpoint distance bars
        ax2.bar(x[i] + w/2, np.mean(endps), w, yerr=np.std(endps) if len(endps)>1 else 0,
                color=colors_map[mode], alpha=0.3, hatch='//', label=f'{titles[mode]} endpt' if i==0 else None)
        # Annotate
        ax2.text(x[i], max(np.mean(paths), np.mean(endps)) + 0.005,
                 f'{np.mean(paths):.3f}\n[{np.mean(endps):.3f}]',
                 ha='center', fontsize=8)

ax2.set_xticks(x)
ax2.set_xticklabels(['SINGLE', 'STATIC', 'DYNAMIC'], fontsize=12)
ax2.set_ylabel('L2 Distance', fontsize=11)
ax2.set_title('Path Integral (filled) vs Endpoint Distance (hatched)', fontsize=13)
# Legend: only show once
handles, labels = ax2.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax2.legend(by_label.values(), by_label.keys(), fontsize=9)
fig2.tight_layout()
fig2.savefig('../paper/figures/fig_path_lengths.png', dpi=200)
fig2.savefig('../paper/figures/fig_path_lengths.pdf', dpi=300)
print('Saved: fig_path_lengths.png/pdf')

print('\nDone.')
