"""Generate kappa comparison figure."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: PG methods on Toy
envs = ['Toy', '510K', 'Overcooked']
algos = ['PPO', 'A2C']
# (revealed_k, hidden_k, revealed_std, hidden_std)
pg_data = {
    ('Toy', 'PPO'): (0.746, 0.049, 0.100, 0.078),
    ('Toy', 'A2C'): (0.839, 0.243, 0.025, 0.378),
    ('510K', 'PPO'): (0.569, 0.386, 0.000, 0.025),
    ('510K', 'A2C'): (0.644, 0.519, 0.201, 0.060),
    ('Overcooked', 'PPO'): (0.497, 0.000, 0.006, 0.000),
}
colors_pg = {'PPO': '#2196F3', 'A2C': '#4CAF50'}

x = np.arange(len(envs))
w = 0.25

for i, algo in enumerate(algos):
    rv, hv, rs, hs = [], [], [], []
    for env in envs:
        if (env, algo) in pg_data:
            d = pg_data[(env, algo)]
            rv.append(d[0]); hv.append(d[1]); rs.append(d[2]); hs.append(d[3])
        else:
            rv.append(0); hv.append(0); rs.append(0); hs.append(0)

    offset = (i - 0.5) * w
    ax1.bar(x + offset - w*0.2, rv, w*0.4, yerr=rs, color=colors_pg[algo], alpha=0.9,
            label=f'{algo} Revealed' if i == 0 else '')
    ax1.bar(x + offset + w*0.2, hv, w*0.4, yerr=hs, color=colors_pg[algo], alpha=0.3,
            label=f'{algo} Hidden' if i == 0 else '', hatch='///')

ax1.set_xticks(x)
ax1.set_xticklabels(envs, fontsize=11)
ax1.set_ylabel(r'$\kappa$', fontsize=13)
ax1.set_title('Policy-Gradient Methods\n' + r'$\kappa_\mathrm{revealed} > \kappa_\mathrm{hidden}$', fontsize=11)
ax1.legend(fontsize=8, loc='upper right')
ax1.set_ylim(0, 1.05)
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)

# Panel B: Cross-family comparison
comparisons = [
    ("PPO\n(Toy)", 0.746, 0.049, 0.100, 0.078, '#2196F3', 'PG'),
    ("A2C\n(510K)", 0.644, 0.519, 0.201, 0.060, '#4CAF50', 'PG'),
    ("PPO\n(Overcooked)", 0.497, 0.000, 0.006, 0.000, '#1565C0', 'PG'),
    ("DQN\n(510K)", 0.797, 0.917, 0.123, 0.063, '#FF5722', 'Value'),
    ("SAC\n(510K)", 0.504, 0.540, 0.038, 0.069, '#9C27B0', 'AC'),
    ("REINFORCE\n(510K)", 0.487, 0.605, 0.312, 0.238, '#795548', 'PG-v'),
]

labels = [c[0] for c in comparisons]
x = np.arange(len(labels))

for i, (label, rk, hk, rs, hs, color, fam) in enumerate(comparisons):
    ax2.bar(i - 0.18, rk, 0.33, color=color, alpha=0.9)
    ax2.bar(i + 0.18, hk, 0.33, color=color, alpha=0.25, hatch='///')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='gray', alpha=0.9, label='Revealed/Static'),
    Patch(facecolor='gray', alpha=0.25, hatch='///', label='Hidden/Dynamic'),
]
ax2.legend(handles=legend_elements, fontsize=8, loc='upper right')

# Family labels
for i, (label, rk, hk, rs, hs, color, fam) in enumerate(comparisons):
    if fam == 'PG':
        ax2.text(i, 1.02, 'PG', ha='center', fontsize=7, fontweight='bold', color='#1565C0')
    elif fam == 'Value':
        ax2.text(i, 1.02, 'Value', ha='center', fontsize=7, fontweight='bold', color='#E65100')
    elif fam == 'AC':
        ax2.text(i, 1.02, 'AC', ha='center', fontsize=7, fontweight='bold', color='#7B1FA2')
    elif fam == 'PG-v':
        ax2.text(i, 1.02, 'PG-v', ha='center', fontsize=7, fontweight='bold', color='#5D4037')

ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=8)
ax2.set_ylabel(r'$\kappa$', fontsize=13)
ax2.set_title('Cross-Family Comparison', fontsize=11)
ax2.set_ylim(0, 1.15)
ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)

fig.suptitle(r'$\kappa$: Diagnosing Gradient Contraction from Hidden Information',
             fontsize=13, fontweight='bold')
fig.tight_layout()

out_dir = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures')
os.makedirs(out_dir, exist_ok=True)
fig.savefig(os.path.join(out_dir, 'kappa_figure.pdf'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(out_dir, 'kappa_figure.png'), dpi=200, bbox_inches='tight')
print('Figure saved to paper/figures/kappa_figure.{pdf,png}')

