"""
Kappa (κ) computation for Overcooked hidden partner experiment.

Measures gradient contraction between two partner types:
  κ = ||(g_A+g_B)/2||^2 / (||g_A||^2+||g_B||^2)/2

κ ≈ 1: gradients align (partner types treated the same)
κ → 0: gradients cancel (partner types pull in opposite directions)

Hypothesis: DYNAMIC mode should show lower κ than SINGLE mode.
"""
import os
import json
import numpy as np
import torch
from stable_baselines3 import PPO

from overcooked_wrapper import OvercookedHiddenPartner, DEFAULT_LAYOUT

OUTPUT_DIR = 'overcooked_kappa'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def rollout(model, env, partner_idx, n_eps=30):
    """Run episodes with a specific partner type, forcing no mid-episode switches."""
    trajectories = []
    for _ in range(n_eps):
        env._current_partner_idx = partner_idx
        env.mode = 'single'  # freeze partner for rollout
        obs, _ = env.reset()
        done = False
        obs_list, act_list, rew_list = [], [], []
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                distribution = model.policy.get_distribution(obs_t)
            action = distribution.get_actions().item()
            obs_list.append(obs.copy())
            act_list.append(action)
            obs, r, done, trunc, _ = env.step(action)
            rew_list.append(r)
        trajectories.append((obs_list, act_list, rew_list))
    return trajectories


def compute_pair_gradients(model, traj_A, traj_B):
    """Compute mean policy gradient for rollouts with partner A and B."""
    grads = []
    for traj in [traj_A, traj_B]:
        total_grad = None
        n_samples = 0
        for obs_list, act_list, rew_list in traj:
            ret = sum(rew_list)
            for obs, act in zip(obs_list, act_list):
                obs_t = torch.FloatTensor(obs).unsqueeze(0)
                distribution = model.policy.get_distribution(obs_t)
                log_prob = distribution.log_prob(torch.tensor([act]))
                model.policy.zero_grad()
                (-log_prob * ret).backward()

                grad_vec = []
                for p in model.policy.parameters():
                    if p.grad is not None:
                        grad_vec.append(p.grad.detach().clone().flatten())
                if grad_vec:
                    g_flat = torch.cat(grad_vec)
                    total_grad = g_flat if total_grad is None else total_grad + g_flat
                    n_samples += 1

        total_grad /= max(n_samples, 1)
        grads.append(total_grad)

    return grads[0], grads[1]


def compute_kappa(grad_A, grad_B):
    avg = (grad_A + grad_B) / 2.0
    num = torch.norm(avg) ** 2
    denom = (torch.norm(grad_A) ** 2 + torch.norm(grad_B) ** 2) / 2.0
    return (num / max(denom, 1e-10)).item()


def compute_kappa_for_model(model_path, n_eps=30):
    """Compute κ for a trained model."""
    model = PPO.load(model_path, device='cpu')

    # Create env in DYNAMIC mode but force partner for rollouts
    env = OvercookedHiddenPartner(
        layout_name=DEFAULT_LAYOUT, mode='dynamic',
        partner_types=['greedy', 'random'], horizon=400,
    )

    print(f'  Collecting rollout with greedy partner...')
    traj_greedy = rollout(model, env, partner_idx=0, n_eps=n_eps)
    print(f'  Collecting rollout with random partner...')
    traj_random = rollout(model, env, partner_idx=1, n_eps=n_eps)

    g_greedy, g_random = compute_pair_gradients(model, traj_greedy, traj_random)
    kappa = compute_kappa(g_greedy, g_random)

    env.close()
    return kappa


def main():
    model_dir = 'models_overcooked'
    modes = ['single', 'static', 'dynamic']
    seeds = [41, 42, 43]
    results = {}

    for mode in modes:
        results[mode] = {}
        for seed in seeds:
            model_path = os.path.join(model_dir, f'overcooked_{mode}_seed{seed}_final.zip')
            if not os.path.exists(model_path):
                print(f'SKIP: {model_path} not found')
                continue
            print(f'Computing κ for {mode} seed {seed}...')
            kappa = compute_kappa_for_model(model_path)
            results[mode][str(seed)] = kappa
            print(f'  κ = {kappa:.4f}')

    print('\n' + '=' * 50)
    print('KAPPA RESULTS')
    print('=' * 50)
    for mode in modes:
        vals = [v for v in results[mode].values()]
        if vals:
            print(f'{mode:10s}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, '
                  f'values={[f"{v:.4f}" for v in vals]}')

    out_path = os.path.join(OUTPUT_DIR, 'kappa_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to {out_path}')


if __name__ == '__main__':
    main()
