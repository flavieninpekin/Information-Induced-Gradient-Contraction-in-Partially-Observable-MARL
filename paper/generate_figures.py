"""Generate paper figures: path lengths + continuous reveal."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

out_dir = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures')
os.makedirs(out_dir, exist_ok=True)

# ======== Figure 1: Path Lengths across 510K modes ========
fig, ax = plt.subplots(figsize=(6, 3.5))

modes = ['SINGLE', 'STATIC', 'OBVIOUS', 'DYNAMIC']
means = [0.456, 0.329, 0.328, 0.293]
stds  = [0.071, 0.086, 0.062, 0.049]
n_seeds = [5, 5, 8, 4]
colors = ['#1565C0', '#4CAF50', '#FF9800', '#F44336']

x = np.arange(len(modes))
bars = ax.bar(x, means, yerr=stds, color=colors, capsize=6, edgecolor='white', linewidth=0.8)

for i, (m, s, n) in enumerate(zip(means, stds, n_seeds)):
    ax.text(i, m + s + 0.015, f'{m:.3f}', ha='center', fontsize=8, fontweight='bold')
    ax.text(i, m - s - 0.025, f'n={n}', ha='center', fontsize=7, color='gray')

ax.set_xticks(x)
ax.set_xticklabels(modes, fontsize=11)
ax.set_ylabel('Path Integral $\\mathcal{P}$', fontsize=12)
ax.set_title('Training Trajectory Length by Cooperation Mode (510K)', fontsize=12, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, 0.58)

# Add OBVIOUS annotation
ax.annotate('OBVIOUS = DYNAMIC rules\n+ visible teams\n\nMatches STATIC (\u0394=0.001)\n\u2192 Team info is causal driver',
            xy=(2, 0.328), xytext=(2.8, 0.48),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.2),
            fontsize=8, color='#E65100', bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0', alpha=0.8))

fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig_path_lengths.pdf'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(out_dir, 'fig_path_lengths.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Saved fig_path_lengths.pdf')

# ======== Figure 2: Continuous Reveal W-Curve ========
fig, ax = plt.subplots(figsize=(6, 3.2))

reveal = [0.0, 0.25, 0.50, 0.75, 1.0]
kappas = [0.5342, 0.4455, 0.5144, 0.3242, 0.6159]
kstd   = [0.0376, 0.0847, 0.0238, 0.0786, 0.0387]

ax.errorbar(reveal, kappas, yerr=kstd, marker='o', markersize=10, linewidth=2.5,
            color='#1565C0', capsize=6, markerfacecolor='white', markeredgewidth=2.5)

# Key annotations
ax.annotate('"No info is better\nthan noisy info"',
            xy=(0.25, 0.446), xytext=(0.08, 0.35),
            arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.2),
            fontsize=9, color='#E65100', fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0', alpha=0.8))

ax.annotate('Maximum penalty:\n"Almost right" is\nmost dangerous',
            xy=(0.75, 0.324), xytext=(0.92, 0.22),
            arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.2),
            fontsize=9, color='#C62828', fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', alpha=0.8))

ax.annotate('Best: full,\nconsistent info',
            xy=(1.0, 0.616), xytext=(0.85, 0.75),
            arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.2),
            fontsize=9, color='#2E7D32', fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', alpha=0.8))

ax.set_xlabel('Team Information Revealed (fraction)', fontsize=12)
ax.set_ylabel('$\\kappa$', fontsize=14)
ax.set_title('Continuous Information Reveal (510K, PPO)', fontsize=12, fontweight='bold')
ax.set_xticks(reveal)
ax.set_xticklabels([f'{int(r*100)}%' for r in reveal], fontsize=11)
ax.set_ylim(0.15, 0.75)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.2)

# Baseline line
ax.axhline(y=0.534, color='gray', linestyle='--', alpha=0.4, linewidth=1)
ax.text(0.02, 0.538, '0% baseline', fontsize=8, color='gray')

fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig_reveal.pdf'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(out_dir, 'fig_reveal.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Saved fig_reveal.pdf')

print('\nAll figures in paper/figures/')
