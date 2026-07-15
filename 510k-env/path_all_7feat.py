"""Re-run path integrals with new 7-dim cooperative features."""
import json, os, time, numpy as np
from warnings import filterwarnings
filterwarnings('ignore')

from sb3_contrib import MaskablePPO
from env.game import Game, GameMode
from env.obs_utils import obs_for_player, action_mask_for_player
from env.features import extract_features, FEATURE_NAMES, FEATURE_DIM
import random

MODEL_DIR = 'models_selfplay'
CKPT_STEPS = list(range(100000, 1100000, 100000)) + ['final']
N_EPS = 50

JOBS = {
    'single': [41, 51, 52, 53, 54],
    'static': [41, 61, 62, 63, 64],
    'dynamic': [41, 42, 43, 44],
}

all_results = {}
total_evals = 0
for mode, seeds in JOBS.items():
    for seed in seeds:
        for step in CKPT_STEPS:
            path = os.path.join(MODEL_DIR, f'510k_{mode}_seed{seed}_{step}_steps.zip')
            if step == 'final':
                path = os.path.join(MODEL_DIR, f'510k_{mode}_seed{seed}_final.zip')
            if os.path.exists(path):
                total_evals += 1

print(f'Total evaluations: ~{total_evals}')

for mode, seeds in JOBS.items():
    for seed in seeds:
        mus = []
        print(f'\n{mode} seed{seed}:', flush=True)
        for step in CKPT_STEPS:
            path = os.path.join(MODEL_DIR, f'510k_{mode}_seed{seed}_{step}_steps.zip')
            step_num = 1100000 if step == 'final' else step
            if not os.path.exists(path):
                continue
            model = MaskablePPO.load(path)
            feats = []
            for ep in range(N_EPS):
                random.seed(seed * 100000 + ep + step_num)
                np.random.seed(seed * 100000 + ep + step_num)
                game = Game(mode=GameMode(mode), num_players=4)
                while not game.is_over:
                    pid = game.current_player
                    obs = obs_for_player(game, pid)
                    mask = action_mask_for_player(game, pid)
                    feats.append(extract_features(game, pid))
                    action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                    valid = game.get_valid_actions(pid)
                    valid_cards = [p.cards for p in valid]
                    if int(action) == 0 and game.can_pass(pid):
                        game.pass_turn(pid)
                    else:
                        idx = int(action) - 1
                        if 0 <= idx < len(valid_cards):
                            game.play_cards(pid, valid_cards[idx])
                        elif valid:
                            game.play_cards(pid, random.choice(valid).cards)
                        elif game.can_pass(pid):
                            game.pass_turn(pid)
            mu = np.mean(np.array(feats, dtype=np.float32), axis=0)
            mus.append((step, mu))
            last_two = [f'{v:.3f}' for v in mu[-2:]]
            print(f'  {step}: {" ".join(f"{v:.3f}" for v in mu[:5])} | {" ".join(last_two)}', flush=True)
        if len(mus) >= 2:
            arr = np.array([m[1] for m in mus])
            path_len = np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1))
            endpt = np.linalg.norm(arr[-1] - arr[0])
            curv = path_len / max(endpt, 1e-6)
            print(f'  → PATH={path_len:.4f} ENDPT={endpt:.4f} CURV={curv:.1f}x', flush=True)
            all_results[(mode, seed)] = {
                'path_len': path_len, 'endpt': endpt, 'curv': curv,
                'start': arr[0].tolist(), 'end': arr[-1].tolist(),
                'trajectory': [m[1].tolist() for m in mus],
            }

# Save
with open('path_data/all_paths_7feat.json', 'w') as f:
    serial = {}
    for (m, s), v in all_results.items():
        serial[f'{m}_{s}'] = {
            'path_len': float(v['path_len']), 'endpt': float(v['endpt']),
            'curv': float(v['curv']),
            'start': v['start'], 'end': v['end'],
            'trajectory': v['trajectory'],
        }
    json.dump(serial, f, indent=2)

# Print summary
print('\n' + '=' * 65)
print(f'FEATURE SET: {FEATURE_NAMES} (n={FEATURE_DIM})')
print('=' * 65)
for mode in ['single', 'static', 'dynamic']:
    vals = [(v['path_len'], v['curv']) for (m, s), v in all_results.items() if m == mode]
    if vals:
        paths = [v[0] for v in vals]
        curvs = [v[1] for v in vals]
        print(f'{mode:8} (n={len(vals)}): path={np.mean(paths):.3f}+-{np.std(paths):.3f}  '
              f'curv median={np.median(curvs):.1f}x  range=[{min(curvs):.1f}-{max(curvs):.1f}]x')

# Feature-specific analysis: new cooperative features
print('\n--- New Cooperative Feature Analysis ---')
for mode in ['single', 'static', 'dynamic']:
    vals = [v for (m, s), v in all_results.items() if m == mode]
    if vals:
        score_spreads = [v['end'][5] for v in vals]  # φ₆ index 5
        sup_gaps = [v['end'][6] for v in vals]  # φ₇ index 6
        print(f'{mode:8}: ScoreSpread={np.mean(score_spreads):.3f}+-{np.std(score_spreads):.3f}  '
              f'SuppressionGap={np.mean(sup_gaps):.3f}+-{np.std(sup_gaps):.3f}')

print('\nDone.')
