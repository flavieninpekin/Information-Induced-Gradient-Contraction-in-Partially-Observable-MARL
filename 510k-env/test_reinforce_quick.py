"""Quick REINFORCE test: 2 seeds x 2 modes x 500 episodes."""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '510k-env'))
from env.dqn_wrapper import FiveTenKMaskedEnv, MASK_DIM, MAX_ACTIONS
from env.discrete_sac import Actor
from train_510k_reinforce import train_one, rollout, grad_kappa, compute_returns
import torch.nn.functional as F

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models_510k_reinforce')

# Override for quick test
import train_510k_reinforce as tr
tr.SEEDS = [41, 42]
tr.TOTAL_EPISODES = 5000
tr.LR = 1e-3
tr.MODES = ['single', 'dynamic']

for mode in tr.MODES:
    for seed in tr.SEEDS:
        fp = os.path.join(MODEL_DIR, f'reinforce_{mode}_seed{seed}.pt')
        if os.path.exists(fp):
            os.remove(fp)
        train_one(mode, seed)

print('\n--- KAPPA ---')
for mode in tr.MODES:
    vals = []
    for seed in tr.SEEDS:
        fp = os.path.join(MODEL_DIR, f'reinforce_{mode}_seed{seed}.pt')
        if not os.path.exists(fp): continue
        ckpt = torch.load(fp, map_location='cpu')
        obs_dim = 112 + MASK_DIM
        actor = Actor(obs_dim, MASK_DIM, MAX_ACTIONS)
        actor.load_state_dict(ckpt['actor']); actor.eval()
        env_a = FiveTenKMaskedEnv(mode=mode)
        ta = rollout(actor, env_a); ra = np.mean([sum(rl) for _,_,rl in ta]); env_a.close()
        env_b = FiveTenKMaskedEnv(mode=mode)
        tb = rollout(actor, env_b); rb = np.mean([sum(rl) for _,_,rl in tb]); env_b.close()
        k = grad_kappa(actor, ta, tb)
        vals.append(k)
        print(f'{mode} s{seed}: k={k:.4f} rA={ra:.2f} rB={rb:.2f}')
    if vals:
        print(f'{mode}: mean={np.mean(vals):.4f} std={np.std(vals):.4f}')
print('DONE')
