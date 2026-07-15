"""Generate paper figures from self-play evaluation results."""
import json, os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = 'paper/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_NAMES = ['MyScore', 'MyHandSize', 'MyStrength', 'TrickScore', 'PassCount']
SHORT_NAMES = ['Score', 'HandSize', 'Strength', 'TrickStake', 'PassCount']
COLORS = {'SINGLE': '#2196F3', 'STATIC': '#4CAF50', 'DYNAMIC_A': '#FF9800', 'DYNAMIC_B': '#FF5722'}

# Data from eval_all_sp.py run
data = {
    ('SINGLE', 'seed41'):      [0.0669, 0.3825, 0.4929, 0.0630, 0.1922],
    ('STATIC', 'seed41'):      [0.0639, 0.3692, 0.4996, 0.0406, 0.1762],
    ('DYNAMIC', 'seed41'):     [0.0512, 0.2977, 0.5047, 0.0291, 0.1896],
    ('DYNAMIC', 'seed42'):     [0.0577, 0.3330, 0.5219, 0.0365, 0.1826],
    ('DYNAMIC', 'seed43'):     [0.0544, 0.3152, 0.5146, 0.0388, 0.1772],
    ('DYNAMIC', 'seed44'):     [0.0522, 0.3019, 0.5193, 0.0340, 0.1764],
    ('DYNAMIC', 'seed45'):     [0.0655, 0.3664, 0.5017, 0.0433, 0.1879],
    ('DYNAMIC', 'seed46'):     [0.0647, 0.3664, 0.5060, 0.0436, 0.1762],
}

# ============================================================
# Figure 1: Grouped bar chart comparing SINGLE, STATIC, DYNAMIC
# ============================================================
fig, ax = plt.subplots(figsize=(7.5, 3.5))

# Use averages for clusters
dyn_a = np.mean([data[('DYNAMIC', f'seed{s}')] for s in ['41','42','43','44']], axis=0)
dyn_b = np.mean([data[('DYNAMIC', f'seed{s}')] for s in ['45','46']], axis=0)

groups = {
    'SINGLE': data[('SINGLE','seed41')],
    'STATIC': data[('STATIC','seed41')],
    'DYN-A': dyn_a,
    'DYN-B': dyn_b,
}

x = np.arange(len(FEATURE_NAMES))
w = 0.2
for i, (label, vals) in enumerate(groups.items()):
    color = COLORS.get(label, '#9E9E9E')
    bars = ax.bar(x + (i-1.5)*w, vals, w, label=label, color=color, alpha=0.85)

ax.set_ylabel('Feature Expectation', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(SHORT_NAMES, fontsize=9)
ax.legend(fontsize=8, ncol=2)
ax.set_title('Feature Profiles of Self-Play Trained Policies', fontsize=11)
ax.set_ylim(0, 0.65)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig_feature_profiles.pdf'), dpi=300)
fig.savefig(os.path.join(OUTPUT_DIR, 'fig_feature_profiles.png'), dpi=200)
print('Saved fig_feature_profiles')

# ============================================================
# Figure 2: DYNAMIC seed scatter (all 6 seeds)
# ============================================================
fig2, ax2 = plt.subplots(figsize=(5, 4))
dyn_seeds = [('seed41','A'), ('seed42','A'), ('seed43','A'), ('seed44','A'),
             ('seed45','B'), ('seed46','B')]

for seed, cluster in dyn_seeds:
    v = data[('DYNAMIC', seed)]
    c = COLORS['DYNAMIC_A'] if cluster == 'A' else COLORS['DYNAMIC_B']
    ax2.scatter(v[1], v[2], s=120, c=c, alpha=0.8, edgecolors='black', linewidth=0.5, zorder=3)
    ax2.annotate(seed.replace('seed','s'), (v[1], v[2]), textcoords="offset points", xytext=(5,5), fontsize=8)

# Add SINGLE and STATIC references
sv = data[('SINGLE','seed41')]
tv = data[('STATIC','seed41')]
ax2.scatter(sv[1], sv[2], marker='s', s=120, c=COLORS['SINGLE'], alpha=0.8, edgecolors='black', linewidth=0.5, zorder=3, label='SINGLE')
ax2.scatter(tv[1], tv[2], marker='^', s=120, c=COLORS['STATIC'], alpha=0.8, edgecolors='black', linewidth=0.5, zorder=3, label='STATIC')
ax2.annotate('SINGLE', (sv[1], sv[2]), textcoords="offset points", xytext=(-40,-15), fontsize=9)
ax2.annotate('STATIC', (tv[1], tv[2]), textcoords="offset points", xytext=(5,-15), fontsize=9)

ax2.set_xlabel('MyHandSize', fontsize=11)
ax2.set_ylabel('MyStrength', fontsize=11)
ax2.set_title('DYNAMIC Seeds: Bimodal Distribution', fontsize=11)
ax2.legend(fontsize=9)
fig2.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, 'fig_dynamic_bimodal.pdf'), dpi=300)
fig2.savefig(os.path.join(OUTPUT_DIR, 'fig_dynamic_bimodal.png'), dpi=200)
print('Saved fig_dynamic_bimodal')

# ============================================================
# Figure 3: Radar chart (3 policies)
# ============================================================
fig3, ax3 = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))
angles = np.linspace(0, 2*np.pi, len(FEATURE_NAMES), endpoint=False).tolist()
angles += angles[:1]

for label, color in [('SINGLE', COLORS['SINGLE']), ('STATIC', COLORS['STATIC'])]:
    vals = data[(label,'seed41')] + [data[(label,'seed41')][0]]
    ax3.plot(angles, vals, 'o-', label=label, color=color, linewidth=2)
    ax3.fill(angles, vals, alpha=0.1, color=color)

vals = dyn_a.tolist() + [dyn_a[0]]
ax3.plot(angles, vals, 'o-', label='DYN-A', color=COLORS['DYNAMIC_A'], linewidth=2)
ax3.fill(angles, vals, alpha=0.1, color=COLORS['DYNAMIC_A'])
vals = dyn_b.tolist() + [dyn_b[0]]
ax3.plot(angles, vals, 'o--', label='DYN-B', color=COLORS['DYNAMIC_B'], linewidth=2)

ax3.set_xticks(angles[:-1])
ax3.set_xticklabels(SHORT_NAMES, fontsize=9)
ax3.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.35, 1.1))
ax3.set_ylim(0, 0.6)
ax3.set_title('Policy Feature Profiles (Radar)', fontsize=11, pad=20)
fig3.tight_layout()
fig3.savefig(os.path.join(OUTPUT_DIR, 'fig_radar.pdf'), dpi=300)
fig3.savefig(os.path.join(OUTPUT_DIR, 'fig_radar.png'), dpi=200)
print('Saved fig_radar')

print('\nAll figures saved to paper/figures/')
