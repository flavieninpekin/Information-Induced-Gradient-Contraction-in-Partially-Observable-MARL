"""
Path Integral Analysis: compute feature expectations at every 100K checkpoint,
then compute path integrals through feature space during training.

For each seed × mode, we have checkpoints at 100K, 200K, ..., 1M (or until final).
The path integral = sum of L2 distances between consecutive checkpoint feature expectations.

This captures how much the policy "wandered" during training,
not just where it ended up.
"""
import json, os, time, gzip, pickle, glob
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings('ignore')

from sb3_contrib import MaskablePPO
from env.game import Game, GameMode
from env.obs_utils import obs_for_player, action_mask_for_player
from env.features import extract_features, FEATURE_NAMES, FEATURE_DIM
from env.scorer import Scorer
import random

MODEL_DIR = 'models_selfplay'
OUTPUT_DIR = 'path_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Checkpoints to evaluate per seed
CKPT_STEPS = list(range(100000, 1100000, 100000))

MODES = ['single', 'static', 'dynamic']
SEEDS = list(range(41, 47))
N_EVAL_EPS = 100  # episodes per checkpoint for feature expectations


def eval_checkpoint(model_path: str, mode: str, seed: int, n_eps: int = N_EVAL_EPS):
    """Extract feature expectations for one checkpoint."""
    if not os.path.exists(model_path):
        return None

    model = MaskablePPO.load(model_path)
    all_features = []

    for ep in range(n_eps):
        random.seed(seed * 100000 + ep)
        np.random.seed(seed * 100000 + ep)
        game = Game(mode=GameMode(mode), num_players=4)

        while not game.is_over:
            pid = game.current_player
            obs = obs_for_player(game, pid)
            mask = action_mask_for_player(game, pid)
            feats = extract_features(game, pid)

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

            all_features.append(feats)

    feats = np.array(all_features, dtype=np.float32)
    mu = np.mean(feats, axis=0)
    return mu


# Collect all available checkpoints
print('Scanning checkpoints...')
available = {}
for mode in MODES:
    for seed in SEEDS:
        ckpts = []
        for step in CKPT_STEPS:
            path = os.path.join(MODEL_DIR, f'510k_{mode}_seed{seed}_{step}_steps.zip')
            if os.path.exists(path):
                ckpts.append((step, path))
            else:
                break  # assume consecutive up to final
        # Also check final
        final_path = os.path.join(MODEL_DIR, f'510k_{mode}_seed{seed}_final.zip')
        if os.path.exists(final_path):
            ckpts.append(('final', final_path))
        if ckpts:
            available[(mode, seed)] = ckpts
            print(f'  {mode} seed{seed}: {len(ckpts)} checkpoints ({ckpts[0][0]} -> {ckpts[-1][0]})')


# Evaluate all checkpoints (this takes a while — 100 eps per checkpoint)
print(f'\nEvaluating {sum(len(v) for v in available.values())} checkpoints...')
trajectories = {}  # (mode, seed) -> [[mu_0, mu_1, ...], [step_0, step_1, ...]]

for (mode, seed), ckpts in available.items():
    print(f'\n  {mode} seed{seed}:', flush=True)
    mus = []
    steps = []
    for step, path in ckpts:
        t0 = time.time()
        mu = eval_checkpoint(path, mode, seed)
        if mu is not None:
            mus.append(mu)
            steps.append(step)
            print(f'    {step}: {[round(v,4) for v in mu]} ({time.time()-t0:.0f}s)', flush=True)
    trajectories[(mode, seed)] = (np.array(mus), steps)

# Save for reuse
with open(os.path.join(OUTPUT_DIR, 'trajectories.json'), 'w') as f:
    serializable = {}
    for key, (mus, steps) in trajectories.items():
        serializable[f'{key[0]}_{key[1]}'] = {'mus': mus.tolist(), 'steps': [str(s) for s in steps]}
    json.dump(serializable, f, indent=2)

# ============================================================
# Path Integral Computation
# ============================================================
print('\n' + '=' * 70)
print('PATH INTEGRALS')
print('=' * 70)

path_lengths = {}
for (mode, seed), (mus, steps) in trajectories.items():
    if len(mus) < 2:
        continue
    # Path length: sum of L2 distances between consecutive checkpoints
    path_len = np.sum(np.linalg.norm(np.diff(mus, axis=0), axis=1))
    # Endpoint distance: L2 between first and last
    endpoint_dist = np.linalg.norm(mus[-1] - mus[0])
    # Curvature: path_length / endpoint_distance (how much did it wander?)
    curvature = path_len / max(endpoint_dist, 1e-6)
    path_lengths[(mode, seed)] = (path_len, endpoint_dist, curvature)
    print(f'  {mode} seed{seed}: path={path_len:.4f} endpt={endpoint_dist:.4f} curvature={curvature:.2f}x')

# Aggregate by mode
print('\n--- By Mode ---')
for mode in MODES:
    vals = [(path_lengths[(m,s)], trajectories[(m,s)]) for (m,s) in path_lengths if m == mode]
    if vals:
        paths = np.array([v[0][0] for v in vals])
        endps = np.array([v[0][1] for v in vals])
        curvs = np.array([v[0][2] for v in vals])
        print(f'  {mode:8}: path={paths.mean():.4f}±{paths.std():.4f}  '
              f'endpt={endps.mean():.4f}±{endps.std():.4f}  curvature={curvs.mean():.1f}x')

# ============================================================
# Visualization: Trajectory in feature space
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax_idx, mode in enumerate(MODES):
    ax = axes[ax_idx]
    for (m, seed), (mus, steps) in trajectories.items():
        if m != mode or len(mus) < 2:
            continue
        # Use first two PC-like dimensions: MyHandSize vs MyStrength
        x = mus[:, 1]  # MyHandSize
        y = mus[:, 2]  # MyStrength
        ax.plot(x, y, 'o-', alpha=0.7, label=f's{seed}', markersize=4, linewidth=1)
        # Mark start and end
        ax.scatter(x[0], y[0], s=60, marker='s', zorder=5, edgecolors='black', linewidth=0.5)
        ax.scatter(x[-1], y[-1], s=60, marker='*', zorder=5, edgecolors='black', linewidth=0.5)
    ax.set_title(f'{mode.upper()}', fontsize=12)
    ax.set_xlabel('MyHandSize', fontsize=9)
    ax.set_ylabel('MyStrength', fontsize=9)
    if ax_idx == 0:
        ax.legend(fontsize=7, ncol=2)

fig.suptitle('Training Trajectories in Feature Space', fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'trajectories.png'), dpi=200)
print(f'\nSaved trajectories.png')

# ============================================================
# Path length comparison bar chart
# ============================================================
fig2, ax2 = plt.subplots(figsize=(8, 4))
x = np.arange(len(MODES))
w = 0.15
colors = ['#2196F3', '#4CAF50', '#FF9800']

for i, mode in enumerate(MODES):
    vals = [path_lengths[(m,s)][0] for (m,s) in path_lengths if m == mode]
    if vals:
        for j, v in enumerate(vals):
            ax2.bar(i + (j - len(vals)/2 + 0.5) * w, v, w, alpha=0.7, color=colors[i])
        ax2.text(i, max(vals) + 0.01, f'm={np.mean(vals):.3f}', ha='center', fontsize=9)

ax2.set_xticks(x)
ax2.set_xticklabels(MODES, fontsize=11)
ax2.set_ylabel('Path Integral (cumulative L2 distance)', fontsize=10)
ax2.set_title('Policy Wandering Distance During Training', fontsize=12)
fig2.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, 'path_lengths.png'), dpi=200)
print('Saved path_lengths.png')

print('\nDone.')
