"""Generate all three paper figures."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

out = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures')
os.makedirs(out, exist_ok=True)

# === FIG 1: Path integral bar chart (510K) ===
fig, ax = plt.subplots(figsize=(5, 3.5))
modes = ['SINGLE', 'STATIC', 'OBVIOUS', 'DYNAMIC']
means = [0.456, 0.329, 0.328, 0.293]
stds = [0.071, 0.086, 0.062, 0.049]
ns = [5, 5, 8, 4]
colors = ['#E53935', '#1E88E5', '#43A047', '#FB8C00']

bars = ax.bar(modes, means, yerr=stds, color=colors, capsize=5, edgecolor='black', linewidth=0.5)
for bar, n in zip(bars, ns):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'n={n}',
            ha='center', fontsize=8, fontweight='bold')

ax.set_ylabel('Path Integral $\\mathcal{P}$', fontsize=12)
ax.set_title('510K: Cooperation Structure vs. Training Stability', fontsize=12, fontweight='bold')
ax.set_ylim(0, 0.6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(out, 'fig_path_lengths.pdf'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(out, 'fig_path_lengths.png'), dpi=200, bbox_inches='tight')
print('FIG 1: path integral done')

# === FIG 2: Toy kappa trajectories ===
fig, ax = plt.subplots(figsize=(5, 3.5))
# HIDDEN: κ ≈ 0.04, many seeds at 0
hidden_vals = [0.104, 0.039, 0.000, 0.005, 0.004, 0.003, 0.235, 0.000, 0.000, 0.000]
revealed_vals = [0.710, 0.547, 0.813, 0.828, 0.646, 0.867, 0.755, 0.801, 0.699, 0.598]

x_h = np.arange(len(hidden_vals)) + 1
x_r = np.arange(len(revealed_vals)) + 1
ax.scatter(x_h, hidden_vals, c='#E53935', marker='x', s=60, label=f'HIDDEN ($\\mu$={np.mean(hidden_vals):.3f})', zorder=3)
ax.scatter(x_r, revealed_vals, c='#43A047', marker='o', s=60, label=f'REVEALED ($\\mu$={np.mean(revealed_vals):.3f})', zorder=3)
ax.axhline(y=np.mean(revealed_vals), color='#43A047', linestyle='--', alpha=0.4)
ax.axhline(y=np.mean(hidden_vals), color='#E53935', linestyle='--', alpha=0.4)
ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
ax.set_ylabel('$\\kappa$', fontsize=12)
ax.set_xlabel('Seed', fontsize=11)
ax.set_title('Toy Matching: $\\kappa$ Across 10 Independent Seeds', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.set_ylim(-0.05, 1.05)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(out, 'kappa_figure.pdf'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(out, 'kappa_figure.png'), dpi=200, bbox_inches='tight')
print('FIG 2: kappa per-seed done')

# === FIG 3: Continuous reveal W-curve ===
fig, ax = plt.subplots(figsize=(5, 3.5))
fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
kappas = [0.534, 0.446, 0.514, 0.324, 0.616]
stds = [0.038, 0.085, 0.024, 0.079, 0.039]

ax.plot(fracs, kappas, 'o-', color='#7B1FA2', linewidth=2, markersize=10, zorder=3)
ax.fill_between(fracs, [k-s for k,s in zip(kappas, stds)], [k+s for k,s in zip(kappas, stds)],
                alpha=0.15, color='#7B1FA2')
ax.axhline(y=kappas[0], color='#FB8C00', linestyle='--', alpha=0.5, label=f'0% baseline ($\\kappa$={kappas[0]:.3f})')
ax.axhline(y=kappas[-1], color='#43A047', linestyle='--', alpha=0.5, label=f'100% baseline ($\\kappa$={kappas[-1]:.3f})')

# Annotate the minimum
min_idx = np.argmin(kappas)
ax.annotate(f'Minimum: $\\kappa$={kappas[min_idx]:.3f}\nat {fracs[min_idx]:.0%} reveal',
            xy=(fracs[min_idx], kappas[min_idx]),
            xytext=(0.55, 0.25), fontsize=8,
            arrowprops=dict(arrowstyle='->', color='#E53935'),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4', alpha=0.8))

ax.set_xlabel('Team Information Revealed', fontsize=11)
ax.set_ylabel('$\\kappa$', fontsize=12)
ax.set_title('Continuous Reveal: W-Shaped $\\kappa$ Curve', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='lower right')
ax.set_xticks(fracs)
ax.set_xticklabels([f'{int(f*100)}%' for f in fracs])
ax.set_ylim(0.2, 0.75)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(out, 'fig_reveal.pdf'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(out, 'fig_reveal.png'), dpi=200, bbox_inches='tight')
print('FIG 3: reveal curve done')
print('\nAll figures saved to paper/figures/')
