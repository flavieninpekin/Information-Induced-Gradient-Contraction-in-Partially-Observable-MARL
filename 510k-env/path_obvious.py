"""Path integral for OBVIOUS seeds 71-74."""
import json, os, time, numpy as np
from warnings import filterwarnings
filterwarnings('ignore')

from sb3_contrib import MaskablePPO
from env.game import Game, GameMode
from env.obs_utils import obs_for_player, action_mask_for_player
from env.features import extract_features
import random

MODEL_DIR = 'models_selfplay'
MODE = 'obvious'
SEEDS = [71, 72, 73, 74]
CKPT_STEPS = list(range(100000, 1100000, 100000)) + ['final']
N_EPS = 50

results = {}
for seed in SEEDS:
    mus = []
    print(f'\nOBVIOUS seed{seed}:')
    for step in CKPT_STEPS:
        path = os.path.join(MODEL_DIR, f'510k_{MODE}_seed{seed}_{step}_steps.zip')
        step_num = 1100000 if step == 'final' else step
        if not os.path.exists(path):
            continue
        model = MaskablePPO.load(path)
        feats = []
        for ep in range(N_EPS):
            random.seed(seed * 100000 + ep + step_num)
            np.random.seed(seed * 100000 + ep + step_num)
            game = Game(mode=GameMode(MODE), num_players=4)
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
        mus.append(mu)
        print(f'  {step}: {" ".join(f"{v:.3f}" for v in mu[:5])} | {" ".join(f"{v:.3f}" for v in mu[5:])}' if len(mu)>5 else f'  {step}: {np.round(mu,4)}', flush=True)
    mus_arr = np.array(mus)
    path_len = np.sum(np.linalg.norm(np.diff(mus_arr, axis=0), axis=1))
    endpt = np.linalg.norm(mus_arr[-1] - mus_arr[0])
    curv = path_len / max(endpt, 1e-6)
    print(f'  PATH={path_len:.4f} ENDPT={endpt:.4f} CURV={curv:.1f}x', flush=True)
    results[seed] = {'path_len': path_len, 'endpt': endpt, 'curv': curv}

print('\n=== OBVIOUS Summary ===')
paths = [v['path_len'] for v in results.values()]
endps = [v['endpt'] for v in results.values()]
curvs = [v['curv'] for v in results.values()]
print(f'OBVIOUS (n=4): path={np.mean(paths):.3f}+-{np.std(paths):.3f}  '
      f'endpt={np.mean(endps):.3f}  curv median={np.median(curvs):.1f}x')

# Compare with previous data
print('\n=== Comparison ===')
prev = {
    'SINGLE':  (0.456, 0.071),
    'STATIC':  (0.329, 0.086),
    'DYNAMIC': (0.293, 0.049),
}
for name, (m, s) in prev.items():
    print(f'{name:10}: path={m:.3f}+-{s:.3f}')
print(f'{"OBVIOUS":10}: path={np.mean(paths):.3f}+-{np.std(paths):.3f}')
