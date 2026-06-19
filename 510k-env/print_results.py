"""Print honest 4-feature results (decision-relevant features)."""
import json, numpy as np, os

TRANSFER_DIR = 'transfer_data'
FEATURE_NAMES = ['MyScore', 'MyHandSize', 'MyStrength', 'TrickScore']
MODES = ['single', 'static', 'dynamic']

summaries = {}
for m in MODES:
    with open(os.path.join(TRANSFER_DIR, f'510k_single_final_{m}_summary.json')) as f:
        summaries[m] = json.load(f)

mu = {m: np.array(summaries[m]['feature_expectations']) for m in MODES}
with open(os.path.join(TRANSFER_DIR, 'random_baseline_features.json')) as f:
    rand = json.load(f)
mu_rand = np.array(rand['feature_expectations'])

print('=' * 70)
print('Decision-Relevant Features (mode-independent computation)')
print('=' * 70)
header = f"{'Feature':<14} {'Random':<8} {'SINGLE':<8} {'STATIC':<8} {'DYNAMIC':<8} {'ΔS-T':<8}"
print(header)
print('-' * len(header))
for i, name in enumerate(FEATURE_NAMES):
    r = mu_rand[i]
    s = mu['single'][i]
    t = mu['static'][i]
    d = mu['dynamic'][i]
    print(f'{name:<14} {r:<8.3f} {s:<8.3f} {t:<8.3f} {d:<8.3f} {s-t:<+8.3f}')

print()
print('Pairwise L2 distances:')
for m1, m2 in [('single', 'static'), ('single', 'dynamic'), ('static', 'dynamic')]:
    d = np.linalg.norm(mu[m1] - mu[m2])
    print(f'  {m1} vs {m2}: L2 = {d:.4f}')

print()
print('=' * 70)
print('INTERPRETATION')
print('=' * 70)
print('''
1. MyScore (0.07-0.08 vs random 0.095):
   The policy gets FEWER 510K points than random play.
   → It prioritizes winning (finishing first) over eating points.
   This is the same in ALL modes (π is fixed, same behavior).

2. MyHandSize (SINGLE=0.369 vs TEAMS=0.442-0.445):
   Player has fewer cards in SINGLE mode.
   → SINGLE games end faster (1 player finishes → game over).
   → TEAM games last longer (2 from same team must finish).
   This is an ENVIRONMENT DYNAMICS effect, not policy adaptation.

3. MyStrength (RL=0.449-0.482 vs Random=0.382):
   The policy keeps significantly stronger cards.
   → This is the core strategic behavior the policy learned.
   Very similar across all modes (0.449-0.482).

4. TrickScore (~0.05 in all modes):
   Identical across modes and vs random.
   → The policy doesn't specially manage trick risk.

CONCLUSION:
  - SINGLE vs TEAM: small difference (L2≈0.12), driven by game length
  - STATIC vs DYNAMIC: nearly identical (L2≈0.008)
  - The policy's decision-making is STABLE across rule changes
  - IRL recovers similar w across modes (policy is the same π)
  - Rule changes affect STATE DISTRIBUTION, not the REWARD FUNCTION
''')
