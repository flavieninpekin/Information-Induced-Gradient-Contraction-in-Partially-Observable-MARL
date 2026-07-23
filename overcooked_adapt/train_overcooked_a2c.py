"""
Overcooked A2C experiment: STATIC vs DYNAMIC.
"""
import os, sys, json, time, multiprocessing
import numpy as np, torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'overcooked_adapt'))
from overcooked_v3_env import OvercookedV3Env
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

MDIR = os.path.join(os.path.dirname(__file__), '..', 'models_overcooked_a2c')
KDIR = os.path.join(os.path.dirname(__file__), '..', 'overcooked_kappa_a2c')
os.makedirs(MDIR, exist_ok=True); os.makedirs(KDIR, exist_ok=True)

MODES = ['static', 'dynamic']; SEEDS = list(range(41, 49)); N_ENVS = 8; TS = 1_000_000

def make_env(mode, seed, rank):
    def _(): return OvercookedV3Env(mode=mode, horizon=400, switch_interval=30, seed=seed + rank * 100)
    return _

def rollout(env, ptype, model, n=30):
    env._force_partner = ptype; tr = []
    for _ in range(n):
        o, _ = env.reset(); done = False
        while not done:
            ot = torch.FloatTensor(o).unsqueeze(0)
            with torch.no_grad(): d = model.policy.get_distribution(ot)
            a = d.get_actions().item(); no, r, done, trunc, _ = env.step(a)
            tr.append((o, a, r, no, done)); o = no
    env._force_partner = None; return tr

def grads(model, traj):
    g = None; n = 0
    for o, a, r, _, _ in traj:
        ot = torch.FloatTensor(o).unsqueeze(0); d = model.policy.get_distribution(ot)
        lp = d.log_prob(torch.tensor([a])); model.policy.zero_grad(); (-lp * r).backward()
        gv = torch.cat([p.grad.detach().clone().flatten() for p in model.policy.parameters() if p.grad is not None])
        g = gv if g is None else g + gv; n += 1
    return g / max(n, 1)

def kappa(gA, gB):
    avg = (gA + gB) / 2.0
    return (torch.norm(avg)**2 / max((torch.norm(gA)**2 + torch.norm(gB)**2) / 2.0, 1e-10)).item()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    for mode in MODES:
        for seed in SEEDS:
            fp = os.path.join(MDIR, f'oc_a2c_{mode}_seed{seed}_final.zip')
            if os.path.exists(fp): print(f'SKIP {mode} s{seed}'); continue
            print(f'TRAIN OC A2C {mode} s{seed}...')
            env = SubprocVecEnv([make_env(mode, seed, i) for i in range(N_ENVS)], start_method='spawn')
            env = VecMonitor(env)
            m = A2C('MlpPolicy', env, learning_rate=3e-4, n_steps=256, gamma=0.99, gae_lambda=0.95, ent_coef=0.01,
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])), verbose=0, seed=seed, device='cuda')
            m.learn(total_timesteps=TS); m.save(fp); env.close()
            print(f'  done')

    results = {}
    for mode in MODES:
        results[mode] = {}
        for seed in SEEDS:
            fp = os.path.join(MDIR, f'oc_a2c_{mode}_seed{seed}_final.zip')
            if not os.path.exists(fp): continue
            m = A2C.load(fp, device='cpu')
            e = OvercookedV3Env(mode=mode, horizon=400)
            tc = rollout(e, 'chef', m, 30); rc = np.mean([t[2] for t in tc])
            tw = rollout(e, 'waiter', m, 30); rw = np.mean([t[2] for t in tw])
            gc = grads(m, tc); gw = grads(m, tw); kv = kappa(gc, gw)
            results[mode][f'seed{seed}'] = {'kappa': kv, 'r_chef': rc, 'r_waiter': rw}
            print(f'OC A2C {mode} s{seed}: k={kv:.4f} rc={rc:.1f} rw={rw:.1f}')
            e.close()

    print(f'\n{"="*40}')
    for mode in MODES:
        vals = [v['kappa'] for v in results[mode].values()]
        if vals: print(f'{mode}: kappa mean={np.mean(vals):.4f} std={np.std(vals):.4f} '
                       f'seeds={[f"{v:.3f}" for v in vals]}')
    with open(os.path.join(KDIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
