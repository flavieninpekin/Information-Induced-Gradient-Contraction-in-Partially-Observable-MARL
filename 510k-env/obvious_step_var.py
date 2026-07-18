"""Compute step-variance and curvature stats for OBVIOUS seeds."""
import os, numpy as np
from sb3_contrib import MaskablePPO
from env.game import Game, GameMode
from env.obs_utils import obs_for_player, action_mask_for_player
from env.features import extract_features
import random

SEEDS = [71, 72, 73, 74]
CKPT_STEPS = list(range(100000, 1100000, 100000))
N_EPS = 50

for seed in SEEDS:
    mus = []
    for step in CKPT_STEPS:
        path = f'models_selfplay/510k_obvious_seed{seed}_{step}_steps.zip'
        if not os.path.exists(path):
            continue
        model = MaskablePPO.load(path)
        feats = []
        for ep in range(N_EPS):
            random.seed(seed * 100000 + ep + step)
            np.random.seed(seed * 100000 + ep + step)
            game = Game(mode=GameMode('obvious'), num_players=4)
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
    mus = np.array(mus)
    steps = np.linalg.norm(np.diff(mus, axis=0), axis=1)
    step_var = np.var(steps)
    path_len = np.sum(steps)
    endpt = np.linalg.norm(mus[-1] - mus[0])
    curv = path_len / max(endpt, 1e-6)
    print(f'OBVIOUS seed{seed}: path={path_len:.4f} endpt={endpt:.4f} '
          f'curv={curv:.1f}x step_var={step_var:.6f} step_mean={np.mean(steps):.4f}')

# Compare with previous data
prev_step_vars = {'single': [0.000299,0.000751,0.001150,0.000597,0.000239],
                  'static': [0.000310,0.000111,0.000806,0.000146,0.000150],
                  'dynamic': [0.000539,0.000134,0.000267,0.000175]}
prev_curvs = {'single': [7.3, 7.1, 16.6, 4.2, 8.8],
              'static': [3.6, 4.0, 6.7, 5.9, 13.5],
              'dynamic': [6.5, 2.1, 14.3, 8.9]}

# Note: these are 7-dim curv values, the OBVIOUS values above are also 7-dim
