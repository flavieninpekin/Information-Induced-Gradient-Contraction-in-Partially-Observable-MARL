"""Compute PPO SINGLE kappa from new models."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '510k-env'))
from env.dqn_wrapper import FiveTenKMaskedEnv
from stable_baselines3 import PPO

MDIR = os.path.join(os.path.dirname(__file__), '..', 'models_510k_ppo_sa')

def kappa_model(fp, mode):
    m = PPO.load(fp, device='cpu'); grads = []
    for _ in range(2):
        env = FiveTenKMaskedEnv(mode=mode); ta = []
        for _ in range(30):
            o, _ = env.reset(); done = False
            while not done:
                ot = torch.FloatTensor(o).unsqueeze(0); d = m.policy.get_distribution(ot)
                a = d.get_actions().item(); no, r, done, trunc, _ = env.step(a)
                ta.append((o, a, r)); o = no
        env.close(); g = None; n = 0
        for o, a, r in ta:
            ot = torch.FloatTensor(o).unsqueeze(0); d = m.policy.get_distribution(ot)
            lp = d.log_prob(torch.tensor([a])); m.policy.zero_grad(); (-lp * r).backward()
            gv = torch.cat([p.grad.detach().clone().flatten()
                          for p in m.policy.parameters() if p.grad is not None])
            g = gv if g is None else g + gv; n += 1
        grads.append(g / max(n, 1))
    gA, gB = grads; avg = (gA + gB) / 2.0
    return (torch.norm(avg)**2 / max((torch.norm(gA)**2 + torch.norm(gB)**2) / 2.0, 1e-10)).item()

# SINGLE
svals = []
for seed in [41, 42, 43, 44]:
    fp = os.path.join(MDIR, f'ppo_sa_single_seed{seed}_final.zip')
    if os.path.exists(fp):
        k = kappa_model(fp, 'single'); svals.append(k)
        print(f'PPO SINGLE s{seed}: k={k:.4f}')

# DYNAMIC
dvals = []
for seed in [41, 42, 43, 44, 45, 46, 47, 48, 49]:
    fp = os.path.join(MDIR, f'ppo_sa_dynamic_seed{seed}_final.zip')
    if os.path.exists(fp):
        k = kappa_model(fp, 'dynamic'); dvals.append(k)

print(f'\nPPO SINGLE: mean={np.mean(svals):.4f} std={np.std(svals):.4f} n={len(svals)}')
print(f'PPO DYNAMIC: mean={np.mean(dvals):.4f} std={np.std(dvals):.4f} n={len(dvals)}')
direction = 'SINGLE > DYNAMIC: CONFIRMED' if np.mean(svals) > np.mean(dvals) else 'REVERSED'
print(direction)
