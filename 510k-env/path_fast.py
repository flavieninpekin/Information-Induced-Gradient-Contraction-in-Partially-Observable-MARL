"""Fast path integral: eval checkpoints, compute immediately, save incrementally."""
import json, os, time, numpy as np
from warnings import filterwarnings
filterwarnings('ignore')

from sb3_contrib import MaskablePPO
from env.game import Game, GameMode
from env.obs_utils import obs_for_player, action_mask_for_player
from env.features import extract_features
import random

MODEL_DIR = 'models_selfplay'
OUTPUT_DIR = 'path_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Only evaluate these (known available)
jobs = {
    ('single', 41): list(range(100000, 1100000, 100000)) + ['final'],
    ('single', 42): [100000, 200000, 300000],
    ('static', 41): list(range(100000, 900000, 100000)),
    ('dynamic', 41): list(range(100000, 1100000, 100000)) + ['final'],
    ('dynamic', 42): list(range(100000, 1100000, 100000)) + ['final'],
    ('dynamic', 43): list(range(100000, 1100000, 100000)) + ['final'],
    ('dynamic', 44): list(range(100000, 1100000, 100000)) + ['final'],
    ('dynamic', 45): list(range(100000, 1100000, 100000)) + ['final'],
    ('dynamic', 46): list(range(100000, 1100000, 100000)) + ['final'],
}

N_EPS = 50

results = {}  # (mode, seed) -> {'mus': [...], 'steps': [...]}

for (mode, seed), steps_list in jobs.items():
    print(f'\n{mode} seed{seed}:', flush=True)
    mus = []
    steps_out = []
    for step in steps_list:
        step_num = 1100000 if step == 'final' else step
        path = os.path.join(MODEL_DIR, f'510k_{mode}_seed{seed}_{step}_steps.zip')
        if not os.path.exists(path):
            if step == 'final':
                path = os.path.join(MODEL_DIR, f'510k_{mode}_seed{seed}_final.zip')
                if not os.path.exists(path):
                    continue
            else:
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
        mus.append(mu)
        steps_out.append(str(step))
        print(f'  {step}: {np.round(mu, 4)}', flush=True)

    results[f'{mode}_{seed}'] = {'mus': [m.tolist() for m in mus], 'steps': steps_out}

    # Save incrementally
    with open(os.path.join(OUTPUT_DIR, 'trajectories.json'), 'w') as f:
        json.dump(results, f, indent=1)

print('\nAll data saved to path_data/trajectories.json')
