"""
Path Integral Analysis for Overcooked.

Tracks policy behaviour through feature space during training,
measuring path length and curvature.
"""
import os
import json
import glob
import numpy as np

from stable_baselines3 import PPO
from overcooked_wrapper import OvercookedHiddenPartner, DEFAULT_LAYOUT

OUTPUT_DIR = 'overcooked_path_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CKPT_STEPS = list(range(100_000, 1_100_000, 100_000))
N_EVAL_EPS = 50

FEATURE_NAMES = [
    'agent_pos_x',        # Normalised agent x position
    'agent_pos_y',        # Normalised agent y position
    'agent_has_object',   # Whether agent carries object
    'partner_pos_x',      # Normalised partner x position
    'partner_pos_y',      # Normalised partner y position
    'agent_distance',     # Normalised distance between agents
    'pots_progress',      # Max onions in any pot
    'soups_delivered',    # Cumulative soups delivered
]
FEATURE_DIM = len(FEATURE_NAMES)


def extract_features(env_wrapper):
    """Extract features from current Overcooked state."""
    state = env_wrapper.base_env.state
    mdp = env_wrapper.mdp

    p0 = state.players[0]
    p1 = state.players[1]

    # Agent position (normalised by grid dims)
    max_x = max(1, max(p.position[0] for p in mdp.terrain_pos_dict.get(' ', [(0, 0)])))
    max_y = max(1, max(p[1] for p in mdp.terrain_pos_dict.get(' ', [(0, 0)])))
    agent_pos_x = p0.position[0] / max(max_x, 1)
    agent_pos_y = p0.position[1] / max(max_y, 1)
    agent_has_object = 1.0 if p0.has_object() else 0.0
    partner_pos_x = p1.position[0] / max(max_x, 1)
    partner_pos_y = p1.position[1] / max(max_y, 1)

    # Distance between agents
    dist = np.linalg.norm(np.array(p0.position) - np.array(p1.position))
    max_dist = np.linalg.norm(np.array([max_x, max_y]))
    agent_distance = dist / max(max_dist, 1)

    # Pot progress
    pot_states = mdp.get_pot_states(state)
    max_onions = 0
    for key, pots in pot_states.items():
        for _ in pots:
            n = int(key.replace('_items', '')) if '_items' in key else 0
            max_onions = max(max_onions, n)
    pots_progress = max_onions / 3.0

    # Soups delivered
    soups_delivered = len(state.all_orders) if hasattr(state, 'all_orders') else 0
    soups_delivered = soups_delivered / 5.0  # normalise

    return np.array([
        agent_pos_x, agent_pos_y, agent_has_object,
        partner_pos_x, partner_pos_y, agent_distance,
        pots_progress, soups_delivered,
    ], dtype=np.float32)


def eval_checkpoint(model_path, env, n_eps=N_EVAL_EPS):
    """Compute feature expectations for one checkpoint."""
    if not os.path.exists(model_path):
        return None

    model = PPO.load(model_path, device='cpu')
    all_features = []

    for ep in range(n_eps):
        obs, _ = env.reset()
        done = False
        while not done:
            feats = extract_features(env)
            all_features.append(feats)
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, trunc, _ = env.step(int(action))

    return np.mean(np.array(all_features, dtype=np.float32), axis=0)


def compute_path_stats(mus):
    """Compute path length, endpoint distance, curvature, directness."""
    mus = np.array(mus)
    diffs = np.diff(mus, axis=0)
    path_len = np.sum(np.linalg.norm(diffs, axis=1))
    endpoint_dist = np.linalg.norm(mus[-1] - mus[0]) if len(mus) > 1 else 0.0
    curvature = path_len / max(endpoint_dist, 1e-10)
    directness = endpoint_dist / max(path_len, 1e-10)
    return path_len, endpoint_dist, curvature, directness


def main():
    model_dir = 'models_overcooked'
    modes = ['single', 'static', 'dynamic']
    seeds = [41, 42, 43]

    results = {}

    for mode in modes:
        results[mode] = {}
        for seed in seeds:
            env = OvercookedHiddenPartner(
                layout_name=DEFAULT_LAYOUT, mode=mode, horizon=400,
            )
            mus = []
            for step in CKPT_STEPS:
                model_path = os.path.join(
                    model_dir, f'overcooked_{mode}_seed{seed}_{step}_steps.zip'
                )
                mu = eval_checkpoint(model_path, env)
                if mu is not None:
                    mus.append(mu)

            env.close()

            if len(mus) < 2:
                print(f'SKIP {mode} seed {seed}: only {len(mus)} checkpoints')
                continue

            path_len, ep_dist, curv, direct = compute_path_stats(mus)
            results[mode][str(seed)] = {
                'path_length': float(path_len),
                'endpoint_distance': float(ep_dist),
                'curvature': float(curv),
                'directness': float(direct),
                'n_checkpoints': len(mus),
            }
            print(f'{mode} seed {seed}: path_len={path_len:.4f}, '
                  f'endpoint={ep_dist:.4f}, curv={curv:.4f}, '
                  f'directness={direct:.4f}')

    print('\n' + '=' * 60)
    print('PATH INTEGRAL SUMMARY')
    print('=' * 60)
    for mode in modes:
        vals = [v['curvature'] for v in results[mode].values()]
        if vals:
            print(f'{mode:10s} curvature: mean={np.mean(vals):.2f}, '
                  f'std={np.std(vals):.2f}, values={[f"{v:.2f}" for v in vals]}')

    out_path = os.path.join(OUTPUT_DIR, 'path_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to {out_path}')


if __name__ == '__main__':
    main()
