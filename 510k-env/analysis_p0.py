"""Final analysis: 6-feature results for AAAI-27 paper."""
import json, os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

TRANSFER_DIR = 'transfer_data'
FEATURE_NAMES = ['MyScore', 'MyProgress', 'MyStrength', 'GameLength', 'OnRedATeam', 'RedARevealed']
FEATURE_DESC = [
    '510K score\n(accumulated)',
    'Hand progress\n(fraction played)',
    'Card strength\n(avg rank)',
    'Game length\n(finished players)',
    'Red-A team\n(membership)',
    'Red-A revealed\n(information)',
]
MODES = ['single', 'static', 'dynamic']
COLORS = {'single': '#2196F3', 'static': '#4CAF50', 'dynamic': '#FF9800', 'random': '#9E9E9E'}
SHORT = {'single': 'SINGLE', 'static': 'STATIC', 'dynamic': 'DYNAMIC'}

# Load
summaries = {}
for m in MODES:
    with open(os.path.join(TRANSFER_DIR, f'510k_single_final_{m}_summary.json')) as f:
        summaries[m] = json.load(f)
with open(os.path.join(TRANSFER_DIR, 'random_baseline_features.json')) as f:
    rand_data = json.load(f)

mu_rand = np.array(rand_data['feature_expectations'])
mu_modes = {m: np.array(summaries[m]['feature_expectations']) for m in MODES}

# ============================================================
# TABLE: print raw values
# ============================================================
print('=' * 80)
print('TABLE 1: Feature Expectations (6-dim, 500 episodes per mode)')
print('=' * 80)
header = f'{"Feature":<16}' + ''.join(f'{SHORT[m]:>10}' for m in MODES) + f'{"Random":>10}'
print(header)
print('-' * len(header))
for i, name in enumerate(FEATURE_NAMES):
    vals = '  '.join(f'{mu_modes[m][i]:.3f}' for m in MODES)
    print(f'{name:<16}  {vals}  {mu_rand[i]:.3f}')

print()
print('Pairwise L2 distances:')
for m1, m2 in [('single', 'static'), ('single', 'dynamic'), ('static', 'dynamic')]:
    d = np.linalg.norm(mu_modes[m1] - mu_modes[m2])
    print(f'  {SHORT[m1]} vs {SHORT[m2]}: {d:.4f}')

print('\nFeature ranking by max pairwise difference:')
for i, name in enumerate(FEATURE_NAMES):
    diffs = [abs(mu_modes[m1][i] - mu_modes[m2][i]) for m1 in MODES for m2 in MODES if m1 < m2]
    print(f'  {name:<16}  max diff = {max(diffs):.4f}')

# ============================================================
# Figure 1: Grouped bar chart
# ============================================================
fig, ax = plt.subplots(figsize=(9, 4.2))
x = np.arange(len(FEATURE_NAMES))
w = 0.22
all_modes = MODES + ['random']
for i, m in enumerate(all_modes):
    vals = mu_rand if m == 'random' else mu_modes[m]
    offset = (i - 1.5) * w
    label = 'Random' if m == 'random' else SHORT[m]
    c = COLORS[m]
    bars = ax.bar(x + offset, vals, w, label=label, color=c, alpha=0.85)
    for bar, v in zip(bars, vals):
        if v > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=6.5)

ax.set_ylabel('Feature Expectation (normalized)', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(FEATURE_DESC, fontsize=9)
ax.legend(fontsize=9)
ax.set_title('State Feature Distributions of a Fixed Policy Under Three Rule Regimes', fontsize=11)
ax.set_ylim(0, 0.85)
fig.tight_layout()
fig.savefig(os.path.join(TRANSFER_DIR, 'fig1_6feat.png'), dpi=200)
print('\nSaved fig1_6feat.png')

# ============================================================
# Figure 2: Delta from random
# ============================================================
fig2, ax2 = plt.subplots(figsize=(9, 4))
for i, m in enumerate(MODES):
    offset = (i - 1) * w
    delta = mu_modes[m] - mu_rand
    bars = ax2.bar(x + offset, delta, w, label=SHORT[m], color=COLORS[m], alpha=0.85)
    for bar, v in zip(bars, delta):
        if abs(v) > 0.01:
            ypos = bar.get_height() + 0.006 if v >= 0 else bar.get_height() - 0.02
            ax2.text(bar.get_x() + bar.get_width()/2, ypos,
                    f'{v:+.2f}', ha='center', va='bottom', fontsize=6.5)

ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
ax2.set_ylabel('Deviation from Random Baseline', fontsize=11)
ax2.set_xticks(x)
ax2.set_xticklabels(FEATURE_DESC, fontsize=9)
ax2.legend(fontsize=9)
ax2.set_title('How Rule Changes Shift the Same Policy\'s State Distribution', fontsize=11)
fig2.tight_layout()
fig2.savefig(os.path.join(TRANSFER_DIR, 'fig2_delta_6feat.png'), dpi=200)
print('Saved fig2_delta_6feat.png')

# ============================================================
# Figure 3: Radar (4 key features)
# ============================================================
KEY_FEATS = ['MyStrength', 'GameLength', 'OnRedATeam', 'RedARevealed']
key_idx = [FEATURE_NAMES.index(f) for f in KEY_FEATS]

fig3, ax3 = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
angles = np.linspace(0, 2*np.pi, len(KEY_FEATS), endpoint=False).tolist()
angles += angles[:1]

for m in MODES:
    vals = [mu_modes[m][i] for i in key_idx] + [mu_modes[m][key_idx[0]]]
    ax3.plot(angles, vals, 'o-', label=SHORT[m], color=COLORS[m], linewidth=2)
    ax3.fill(angles, vals, alpha=0.1, color=COLORS[m])

ax3.set_xticks(angles[:-1])
ax3.set_xticklabels(KEY_FEATS, fontsize=10)
ax3.set_title('Feature Profiles by Mode\n(4 discriminating features)', fontsize=11, pad=20)
ax3.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))
ax3.set_ylim(0, 0.7)
fig3.tight_layout()
fig3.savefig(os.path.join(TRANSFER_DIR, 'fig3_radar_6feat.png'), dpi=200)
print('Saved fig3_radar_6feat.png')

# Save a compact numeric summary for the proposal
summary = {
    'feature_names': FEATURE_NAMES,
    'feature_expectations': {m: [round(v,4) for v in mu_modes[m]] for m in MODES},
    'random_baseline': [round(v,4) for v in mu_rand],
    'pairwise_L2': {
        'single_vs_static': round(np.linalg.norm(mu_modes['single'] - mu_modes['static']), 4),
        'single_vs_dynamic': round(np.linalg.norm(mu_modes['single'] - mu_modes['dynamic']), 4),
        'static_vs_dynamic': round(np.linalg.norm(mu_modes['static'] - mu_modes['dynamic']), 4),
    }
}
with open(os.path.join(TRANSFER_DIR, 'results_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print('\nSaved results_summary.json')
print('Done.')
