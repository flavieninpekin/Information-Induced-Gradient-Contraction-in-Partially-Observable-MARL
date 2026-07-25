"""Compute PPO kappa from existing 510K models."""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '510k-env'))

from sb3_contrib import MaskablePPO
from env.env_510k import FiveTenKEnv

def rollout_ppo(model, env, n_eps=30):
    """Run episodes, collect (obs, action, reward)."""
    trajectories = []
    for ep in range(n_eps):
        obs, _ = env.reset()
        done = False
        olist, alist, rlist = [], [], []
        while not done:
            mask = env.unwrapped._get_action_mask()
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                distribution = model.policy.get_distribution(obs_t)
            action = distribution.get_actions().item()
            olist.append(obs.copy())
            alist.append(action)
            obs, r, done, trunc, info = env.step(action)
            rlist.append(r)
        trajectories.append((olist, alist, rlist))
    return trajectories

def kappa_ppo(model, traj_A, traj_B):
    """Compute κ from REINFORCE gradients of two trajectory sets."""
    grads = []
    for traj in [traj_A, traj_B]:
        total_grad = None; n = 0
        for olist, alist, rlist in traj:
            ret = sum(rlist)
            for obs, act in zip(olist, alist):
                obs_t = torch.FloatTensor(obs).unsqueeze(0)
                distribution = model.policy.get_distribution(obs_t)
                log_prob = distribution.log_prob(torch.tensor([act]))
                model.policy.zero_grad()
                (-log_prob * ret).backward()
                gv = torch.cat([p.grad.detach().clone().flatten()
                              for p in model.policy.parameters() if p.grad is not None])
                total_grad = gv if total_grad is None else total_grad + gv
                n += 1
        grads.append(total_grad / max(n, 1))
    gA, gB = grads
    avg = (gA + gB) / 2.0
    return (torch.norm(avg)**2 / max((torch.norm(gA)**2 + torch.norm(gB)**2) / 2.0, 1e-10)).item()

# SINGLE
print('Loading PPO SINGLE...')
m = MaskablePPO.load(r'C:\Users\Flavi\llmprojects\project3\models\510k_single_final.zip', device='cpu')
env = FiveTenKEnv(mode='single')
ta = rollout_ppo(m, env, 30); env.close()
env2 = FiveTenKEnv(mode='single')
tb = rollout_ppo(m, env2, 30); env2.close()
ks = kappa_ppo(m, ta, tb)
ra = np.mean([sum(rl) for _,_,rl in ta])
rb = np.mean([sum(rl) for _,_,rl in tb])
print(f'PPO SINGLE: κ={ks:.4f}  rA={ra:.2f}  rB={rb:.2f}')

# STATIC
print('Loading PPO STATIC...')
m2 = MaskablePPO.load(r'C:\Users\Flavi\llmprojects\project3\models_selfplay\510k_static_seed41_final.zip', device='cpu')
env = FiveTenKEnv(mode='static')
ta = rollout_ppo(m2, env, 30); env.close()
env2 = FiveTenKEnv(mode='static')
tb = rollout_ppo(m2, env2, 30); env2.close()
kt = kappa_ppo(m2, ta, tb)
ra = np.mean([sum(rl) for _,_,rl in ta])
rb = np.mean([sum(rl) for _,_,rl in tb])
print(f'PPO STATIC s41: κ={kt:.4f}  rA={ra:.2f}  rB={rb:.2f}')

print(f'\n=== RESULT ===')
print(f'SINGLE κ={ks:.4f}  STATIC κ={kt:.4f}')
if ks > kt: print('SINGLE > STATIC: κ framework CONFIRMED')
else: print('SINGLE <= STATIC: REVERSED (unexpected)')
