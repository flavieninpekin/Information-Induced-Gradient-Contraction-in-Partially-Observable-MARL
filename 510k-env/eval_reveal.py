"""Compute kappa for half-reveal models + continuous reveal curve."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '510k-env'))
from env.env_510k import FiveTenKEnv
from sb3_contrib import MaskablePPO

MDIR = os.path.join(os.path.dirname(__file__), '..', 'models_reveal')

def kappa_maskable(model, mode='obvious', n_eps=30):
    """Compute kappa for MaskablePPO model between two game seeds."""
    grads = []
    for _ in range(2):
        env = FiveTenKEnv(mode=mode)
        g = None; n = 0
        for _ in range(n_eps):
            o, info = env.reset(); done = False
            while not done:
                mask = env._get_action_mask()
                ot = torch.FloatTensor(o).unsqueeze(0)
                d = model.policy.get_distribution(ot)
                a = d.get_actions().item()
                no, r, done, trunc, info = env.step(int(a))
                # REINFORCE gradient
                d2 = model.policy.get_distribution(torch.FloatTensor(o).unsqueeze(0))
                lp = d2.log_prob(torch.tensor([a]))
                model.policy.zero_grad(); (-lp * r).backward()
                gv = torch.cat([p.grad.detach().clone().flatten()
                              for p in model.policy.parameters() if p.grad is not None])
                g = gv if g is None else g + gv; n += 1; o = no
        env.close()
        grads.append(g / max(n, 1))
    gA, gB = grads; avg = (gA + gB) / 2.0
    return (torch.norm(avg)**2 / max((torch.norm(gA)**2 + torch.norm(gB)**2) / 2.0, 1e-10)).item()

# Compute for half models
hvals = []
for seed in [41, 42, 43]:
    fp = os.path.join(MDIR, f'ppo_half_{seed}.zip')
    if os.path.exists(fp):
        m = MaskablePPO.load(fp, device='cpu')
        k = kappa_maskable(m, 'obvious', 30)
        hvals.append(k)
        print(f'HALF s{seed}: k={k:.4f}')

# Also compute for OBVIOUS model
ofp = os.path.join(MDIR, 'ppo_obvious_41_final.zip')
if os.path.exists(ofp):
    m = MaskablePPO.load(ofp, device='cpu')
    k = kappa_maskable(m, 'obvious', 30)
    print(f'OBVIOUS: k={k:.4f}')

# DYNAMIC value from earlier computation
dyn_k = 0.531  # mean from 9 seeds

print(f'\n=== CONTINUOUS REVEAL CURVE ===')
print(f'  0% (DYNAMIC):   k={dyn_k:.4f}')
if hvals: print(f'  50% (HALF):     k={np.mean(hvals):.4f} +- {np.std(hvals):.4f}')
if 'k' in dir(): print(f'  100% (OBVIOUS):  k={k:.4f}')
