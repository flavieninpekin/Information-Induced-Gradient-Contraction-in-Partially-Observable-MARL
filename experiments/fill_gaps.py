"""Fill gaps: Toy A2C 8 seeds + 510K SAC 4 more seeds."""
import os, sys, json, time, numpy as np, torch

# ---- Toy A2C 8 seeds ----
print('=== Toy A2C 8 seeds ===')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '510k-env'))
from stable_baselines3 import A2C
from env.toy_env import HiddenMatchingEnv

results = {'HIDDEN': [], 'REVEALED': []}
for seed in range(8):
    for name, revealed in [('HIDDEN', False), ('REVEALED', True)]:
        env = HiddenMatchingEnv(revealed=revealed, n_steps=20)
        model = A2C('MlpPolicy', env, learning_rate=1e-3, gamma=0.99,
                    policy_kwargs=dict(net_arch=[32, 32]),
                    verbose=0, seed=seed, device='cpu')
        model.learn(total_timesteps=10000, progress_bar=False)

        traj_B, traj_C = [], []
        for partner in [0, 1]:
            env._forced_partner = partner
            for _ in range(20):
                env.partner = partner; o, _ = env.reset(); done = False
                ol, al, rl = [], [], []
                while not done:
                    ol.append(o.copy())
                    ot = torch.FloatTensor(o).unsqueeze(0)
                    with torch.no_grad(): d = model.policy.get_distribution(ot)
                    a = d.get_actions().item(); al.append(a)
                    o, r, done, _, _ = env.step(a); rl.append(r)
                (traj_B if partner == 0 else traj_C).append((ol, al, rl))

        grads = []
        for traj in [traj_B, traj_C]:
            g = None; n = 0
            for ol, al, rl in traj:
                ret = sum(rl)
                for o2, a2 in zip(ol, al):
                    ot = torch.FloatTensor(o2).unsqueeze(0)
                    d = model.policy.get_distribution(ot)
                    lp = d.log_prob(torch.tensor([a2]))
                    model.policy.zero_grad(); (-lp * ret).backward()
                    gv = torch.cat([p.grad.detach().clone().flatten()
                                  for p in model.policy.parameters() if p.grad is not None])
                    g = gv if g is None else g + gv; n += 1
            grads.append(g / max(n, 1))
        gA, gB = grads
        avg = (gA + gB) / 2.0
        k = (torch.norm(avg)**2 / max((torch.norm(gA)**2 + torch.norm(gB)**2) / 2.0, 1e-10)).item()
        results[name].append(k)
        print(f'  Toy A2C {name} seed{seed}: kappa={k:.4f}')

print(f'\nToy A2C 8-seed summary:')
print(f'  HIDDEN:  mean={np.mean(results["HIDDEN"]):.4f}  std={np.std(results["HIDDEN"]):.4f}  '
      f'vals={[f"{v:.3f}" for v in results["HIDDEN"]]}')
print(f'  REVEALED: mean={np.mean(results["REVEALED"]):.4f}  std={np.std(results["REVEALED"]):.4f}  '
      f'vals={[f"{v:.3f}" for v in results["REVEALED"]]}')

import sys; sys.stdout.flush()

# ---- 510K SAC 4 more seeds (43-46) ----
print('\n=== 510K SAC seeds 43-46 ===')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '510k-env'))
from env.dqn_wrapper import FiveTenKMaskedEnv, MASK_DIM, MAX_ACTIONS
from env.discrete_sac import DiscreteSAC

for mode in ['single', 'dynamic']:
    for seed in [43, 44, 45, 46]:
        fp = os.path.join(
            os.path.dirname(__file__), '..', 'models_510k_sac',
            f'510k_sac_{mode}_seed{seed}.pt'
        )
        if os.path.exists(fp): print(f'SKIP {mode} seed{seed}'); continue
        print(f'SAC {mode} seed{seed}...'); sys.stdout.flush()
        env = FiveTenKMaskedEnv(mode=mode)
        obs_dim = env.observation_space.shape[0]
        sac = DiscreteSAC(obs_dim, MASK_DIM, MAX_ACTIONS, lr=3e-4, device='cuda')
        obs, _ = env.reset(); step = 0; t0 = time.time()
        while step < 500_000:
            action = sac.select_action(obs)
            next_obs, reward, done, trunc, info = env.step(action)
            mask = info.get('action_mask', np.ones(MASK_DIM, dtype=np.float32))
            sac.buffer.add(obs, action, reward, next_obs, done, mask)
            sac.update(batch_size=64); obs = next_obs; step += 1
            if done or trunc: obs, _ = env.reset()
        sac.save(fp); env.close()
        print(f'  done {time.time()-t0:.0f}s (alpha={sac.alpha_val:.3f})')

# Kappa for SAC
print('\n=== KAPPA ===')
results_sac = {}
for mode in ['single', 'dynamic']:
    results_sac[mode] = {}
    for seed in [41, 42, 43, 44, 45, 46]:
        fp = os.path.join(os.path.dirname(__file__), '..', 'models_510k_sac', f'510k_sac_{mode}_seed{seed}.pt')
        if not os.path.exists(fp): continue
        sac = DiscreteSAC(112 + MASK_DIM, MASK_DIM, MAX_ACTIONS, device='cpu')
        sac.load(fp); sac.actor.eval()
        # Rollout A
        env_a = FiveTenKMaskedEnv(mode=mode)
        ta_obs, ta_act, ta_rew = [], [], []
        for _ in range(30):
            o, _ = env_a.reset(); done = False
            while not done:
                ot = torch.FloatTensor(o).unsqueeze(0)
                with torch.no_grad(): a, _ = sac.actor.get_action(ot, deterministic=True)
                act = a.item(); ta_obs.append(o.copy()); ta_act.append(act)
                o, r, done, trunc, _ = env.step(act); ta_rew.append(r)
        env_a.close()
        gA = sac.actor_gradient(ta_obs, ta_act)
        # Rollout B
        env_b = FiveTenKMaskedEnv(mode=mode)
        tb_obs, tb_act, tb_rew = [], [], []
        for _ in range(30):
            o, _ = env_b.reset(); done = False
            while not done:
                ot = torch.FloatTensor(o).unsqueeze(0)
                with torch.no_grad(): a, _ = sac.actor.get_action(ot, deterministic=True)
                act = a.item(); tb_obs.append(o.copy()); tb_act.append(act)
                o, r, done, trunc, _ = env.step(act); tb_rew.append(r)
        env_b.close()
        gB = sac.actor_gradient(tb_obs, tb_act)
        avg = (gA + gB) / 2.0
        k = (torch.norm(avg)**2 / max((torch.norm(gA)**2 + torch.norm(gB)**2) / 2.0, 1e-10)).item()
        results_sac[mode][f'seed{seed}'] = {'kappa': k, 'rA': np.mean(ta_rew), 'rB': np.mean(tb_rew)}
        print(f'SAC {mode} s{seed}: k={k:.4f} rA={np.mean(ta_rew):.2f} rB={np.mean(tb_rew):.2f}')

print('\n=== SAC SUMMARY ===')
for mode in ['single','dynamic']:
    vals = [v['kappa'] for v in results_sac[mode].values()]
    if vals: print(f'{mode}: mean={np.mean(vals):.4f} std={np.std(vals):.4f} seeds={[f"{v:.3f}" for v in vals]}')

print('\n=== ALL DONE ===')
