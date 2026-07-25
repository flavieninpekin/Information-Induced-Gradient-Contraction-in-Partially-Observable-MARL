"""Toy PPO: fill PPO row in Toy column."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '510k-env'))
from stable_baselines3 import PPO
from env.toy_env import HiddenMatchingEnv

results = {'HIDDEN': [], 'REVEALED': []}
for seed in range(8):
    for name, revealed in [('HIDDEN', False), ('REVEALED', True)]:
        env = HiddenMatchingEnv(revealed=revealed, n_steps=20)
        model = PPO('MlpPolicy', env, learning_rate=1e-3, gamma=0.99, n_steps=64, batch_size=32,
                    policy_kwargs=dict(net_arch=[32, 32]), verbose=0, seed=seed, device='cpu')
        model.learn(total_timesteps=10000, progress_bar=False)
        grads = []
        for partner in [0, 1]:
            env._forced_partner = partner
            g = None; n = 0
            for _ in range(20):
                env.partner = partner; o, _ = env.reset(); done = False
                while not done:
                    ot = torch.FloatTensor(o).unsqueeze(0)
                    with torch.no_grad(): d = model.policy.get_distribution(ot)
                    a = d.get_actions().item()
                    no, r, done, _, _ = env.step(a)
                    d2 = model.policy.get_distribution(torch.FloatTensor(o).unsqueeze(0))
                    lp = d2.log_prob(torch.tensor([a]))
                    model.policy.zero_grad(); (-lp * r).backward()
                    gv = torch.cat([p.grad.detach().clone().flatten()
                                  for p in model.policy.parameters() if p.grad is not None])
                    g = gv if g is None else g + gv; n += 1; o = no
            grads.append(g / max(n, 1))
        gA, gB = grads; avg = (gA + gB) / 2.0
        k = (torch.norm(avg)**2 / max((torch.norm(gA)**2 + torch.norm(gB)**2) / 2.0, 1e-10)).item()
        results[name].append(k)
        print(f'Toy PPO {name} s{seed}: k={k:.4f}')

print(f'\nPPO Toy HIDDEN:  mean={np.mean(results["HIDDEN"]):.4f}  std={np.std(results["HIDDEN"]):.4f}')
print(f'PPO Toy REVEALED: mean={np.mean(results["REVEALED"]):.4f}  std={np.std(results["REVEALED"]):.4f}')
