"""
Fast batch kappa computation for all trained models.
"""
import os, sys, json, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from overcooked_wrapper import OvercookedHiddenPartner
from stable_baselines3 import PPO

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models_overcooked')
RESULT_DIR = os.path.join(os.path.dirname(__file__), '..', 'overcooked_kappa')
os.makedirs(RESULT_DIR, exist_ok=True)

MODES = ['single', 'static', 'dynamic']
SEEDS = [41, 42, 43]
N_EPS = 30


def rollout(env, partner_idx, model, n_eps=N_EPS):
    env.mode = 'single'
    env._current_partner_idx = partner_idx
    trajectories = []
    for _ in range(n_eps):
        obs, _ = env.reset()
        done = False
        olist, alist, rlist = [], [], []
        while not done:
            olist.append(obs.copy())
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                dist = model.policy.get_distribution(obs_t)
            act = dist.get_actions().item()
            alist.append(act)
            obs, r, done, trunc, _ = env.step(act)
            rlist.append(r)
        trajectories.append((olist, alist, rlist))
    return trajectories


def compute_grads(model, traj):
    total_grad = None
    n = 0
    for olist, alist, rlist in traj:
        ret = sum(rlist)
        for obs, act in zip(olist, alist):
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            dist = model.policy.get_distribution(obs_t)
            lp = dist.log_prob(torch.tensor([act]))
            model.policy.zero_grad()
            (-lp * ret).backward()
            gv = torch.cat([p.grad.detach().clone().flatten()
                           for p in model.policy.parameters() if p.grad is not None])
            total_grad = gv if total_grad is None else total_grad + gv
            n += 1
    return total_grad / max(n, 1)


def kappa(gA, gB):
    avg = (gA + gB) / 2.0
    return (torch.norm(avg)**2 / max((torch.norm(gA)**2 + torch.norm(gB)**2) / 2.0, 1e-10)).item()


results = {}
for mode in MODES:
    results[mode] = {}
    for seed in SEEDS:
        path = os.path.join(MODEL_DIR, f'overcooked_{mode}_seed{seed}_final.zip')
        if not os.path.exists(path):
            print(f'SKIP: {mode} seed{seed} not found')
            continue

        print(f'{mode} seed{seed}...')
        model = PPO.load(path, device='cpu')
        env = OvercookedHiddenPartner(layout_name='cramped_room', mode=mode, horizon=400)

        traj_greedy = rollout(env, 0, model, N_EPS)
        traj_random = rollout(env, 1, model, N_EPS)

        gG = compute_grads(model, traj_greedy)
        gR = compute_grads(model, traj_random)
        k = kappa(gG, gR)
        results[mode][f'seed{seed}'] = k
        print(f'  kappa = {k:.4f}')
        env.close()

# Summary
print(f'\n{"="*60}')
print('KAPPA RESULTS')
print(f'{"="*60}')
for mode in MODES:
    vals = list(results[mode].values())
    if vals:
        print(f'{mode:10s}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, '
              f'seeds={[f"{v:.4f}" for v in vals]}')

with open(os.path.join(RESULT_DIR, 'kappa_final.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f'\nSaved to overcooked_kappa/kappa_final.json')
