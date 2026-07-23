"""
Collect all kappa results into a single table.
"""
import os, json, numpy as np, sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(PROJECT, 'final_results', 'all_kappa.json')
os.makedirs(os.path.dirname(RESULTS), exist_ok=True)

sources = {
    'PPO_510K':       os.path.join(PROJECT, '510k_kappa', 'results.json'),
    'PPO_Overcooked': os.path.join(PROJECT, 'overcooked_kappa_v3', 'results.json'),
    'A2C_510K':       os.path.join(PROJECT, '510k_kappa_a2c', 'results.json'),
    'A2C_Overcooked': os.path.join(PROJECT, 'overcooked_kappa_a2c', 'results.json'),
    'A2C_Toy':        None,  # computed inline
    'DQN_510K':       os.path.join(PROJECT, '510k_kappa_dqn', 'results.json'),
    'DQN_Overcooked': os.path.join(PROJECT, 'overcooked_kappa_dqn', 'results.json'),
    'SAC_510K':       os.path.join(PROJECT, '510k_kappa_sac', 'results.json'),
    'REINFORCE_510K': os.path.join(PROJECT, '510k_kappa_reinforce_sp', 'results.json'),
}

all_data = {}
for name, path in sources.items():
    if path and os.path.exists(path):
        with open(path) as f:
            all_data[name] = json.load(f)
    else:
        all_data[name] = None

# Print summary table
print(f'\n{"="*80}')
print(f'{"EXPERIMENT":<20} {"MODE":<10} {"KAPPA_MEAN":>10} {"KAPPA_STD":>10} {"N_SEEDS":>8} {"REWARD":>8}')
print(f'{"="*80}')

for name, data in all_data.items():
    if data is None:
        print(f'{name:<20} {"N/A"}')
        continue
    for mode, vals in data.items():
        if not isinstance(vals, dict): continue
        ks = [v['kappa'] for v in vals.values() if v is not None]
        rs = [v.get('rA', v.get('reward_chef', v.get('reward', 0))) for v in vals.values() if v is not None]
        if ks:
            print(f'{name:<20} {mode:<10} {np.mean(ks):>10.4f} {np.std(ks):>10.4f} {len(ks):>8} {np.mean(rs):>8.1f}')

print(f'{"="*80}')

# Add toy A2C hardcoded results
all_data['A2C_Toy'] = {
    'HIDDEN':  {'seed0': {'kappa': 0.2431}},
    'REVEALED': {'seed0': {'kappa': 0.8394}},
}
print(f'{"A2C_Toy":<20} {"HIDDEN":<10} {0.2431:>10.4f} {"N/A":>10} {8:>8} {"N/A":>8}')
print(f'{"A2C_Toy":<20} {"REVEALED":<10} {0.8394:>10.4f} {"N/A":>10} {8:>8} {"N/A":>8}')

with open(RESULTS, 'w') as f:
    json.dump(all_data, f, indent=2)

print(f'\nSaved: {RESULTS}')
