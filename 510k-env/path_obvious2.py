"""Path integral for OBVIOUS seeds 75-78."""
import os, numpy as np, random
from sb3_contrib import MaskablePPO
from env.game import Game, GameMode
from env.obs_utils import obs_for_player, action_mask_for_player
from env.features import extract_features

SEEDS = [75, 76, 77, 78]
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
    path_len = np.sum(np.linalg.norm(np.diff(mus, axis=0), axis=1))
    endpt = np.linalg.norm(mus[-1] - mus[0])
    curv = path_len / max(endpt, 1e-6)
    steps = np.linalg.norm(np.diff(mus, axis=0), axis=1)
    step_var = np.var(steps)
    print(f'OBVIOUS seed{seed}: path={path_len:.4f} curv={curv:.1f}x step_var={step_var:.6f}')

# Combined stats
all_obv_paths = [0.203, 0.355, 0.366, 0.302]  # from previous run
# will add new ones after this run
