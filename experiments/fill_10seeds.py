"""
Final data fill: extend to 10 seeds + export raw data.
"""
import os, sys, json, time, multiprocessing
import numpy as np, torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '510k-env'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'overcooked_adapt'))

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ========== TOY PPO: seeds 48,49 ==========
print('=== Toy PPO seeds 48,49 ===')
from stable_baselines3 import PPO
from env.toy_env import HiddenMatchingEnv

for seed in [48, 49]:
    for name, revealed in [('HIDDEN', False), ('REVEALED', True)]:
        env = HiddenMatchingEnv(revealed=revealed, n_steps=20)
        model = PPO('MlpPolicy', env, learning_rate=1e-3, gamma=0.99, n_steps=64, batch_size=32,
                    policy_kwargs=dict(net_arch=[32, 32]), verbose=0, seed=seed, device='cpu')
        model.learn(total_timesteps=10000, progress_bar=False)
        grads = []
        for partner in [0, 1]:
            env._forced_partner = partner; g = None; n = 0
            for _ in range(20):
                env.partner = partner; o, _ = env.reset(); done = False
                while not done:
                    ot = torch.FloatTensor(o).unsqueeze(0)
                    with torch.no_grad(): d = model.policy.get_distribution(ot)
                    a = d.get_actions().item(); no, r, done, _, _ = env.step(a)
                    d2 = model.policy.get_distribution(torch.FloatTensor(o).unsqueeze(0))
                    lp = d2.log_prob(torch.tensor([a])); model.policy.zero_grad(); (-lp * r).backward()
                    gv = torch.cat([p.grad.detach().clone().flatten()
                                  for p in model.policy.parameters() if p.grad is not None])
                    g = gv if g is None else g + gv; n += 1; o = no
            grads.append(g / max(n, 1))
        gA, gB = grads; avg = (gA + gB) / 2.0
        k = (torch.norm(avg)**2 / max((torch.norm(gA)**2 + torch.norm(gB)**2) / 2.0, 1e-10)).item()
        print(f'  Toy PPO {name} s{seed}: k={k:.4f}')

# ========== TOY A2C: seeds 48,49 ==========
print('\n=== Toy A2C seeds 48,49 ===')
from stable_baselines3 import A2C

for seed in [48, 49]:
    for name, revealed in [('HIDDEN', False), ('REVEALED', True)]:
        env = HiddenMatchingEnv(revealed=revealed, n_steps=20)
        model = A2C('MlpPolicy', env, learning_rate=1e-3, gamma=0.99,
                    policy_kwargs=dict(net_arch=[32, 32]), verbose=0, seed=seed, device='cpu')
        model.learn(total_timesteps=10000, progress_bar=False)
        grads = []
        for partner in [0, 1]:
            env._forced_partner = partner; g = None; n = 0
            for _ in range(20):
                env.partner = partner; o, _ = env.reset(); done = False
                while not done:
                    ot = torch.FloatTensor(o).unsqueeze(0)
                    with torch.no_grad(): d = model.policy.get_distribution(ot)
                    a = d.get_actions().item(); no, r, done, _, _ = env.step(a)
                    d2 = model.policy.get_distribution(torch.FloatTensor(o).unsqueeze(0))
                    lp = d2.log_prob(torch.tensor([a])); model.policy.zero_grad(); (-lp * r).backward()
                    gv = torch.cat([p.grad.detach().clone().flatten()
                                  for p in model.policy.parameters() if p.grad is not None])
                    g = gv if g is None else g + gv; n += 1; o = no
            grads.append(g / max(n, 1))
        gA, gB = grads; avg = (gA + gB) / 2.0
        k = (torch.norm(avg)**2 / max((torch.norm(gA)**2 + torch.norm(gB)**2) / 2.0, 1e-10)).item()
        print(f'  Toy A2C {name} s{seed}: k={k:.4f}')

# ========== 510K PPO DYNAMIC: seeds 45-48 ==========
print('\n=== 510K PPO DYNAMIC seeds 45-48 ===')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '510k-env'))
from env.dqn_wrapper import FiveTenKMaskedEnv, MASK_DIM, MAX_ACTIONS
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

MODEL_DIR = os.path.join(PROJECT, 'models_510k_ppo_sa')
os.makedirs(MODEL_DIR, exist_ok=True)
N_ENVS = 8; TS = 1_000_000

def make_ppo_env(mode, seed, rank):
    def _(): return FiveTenKMaskedEnv(mode=mode)
    return _

def compute_kappa_from_model(model_path, mode):
    m = PPO.load(model_path, device='cpu')
    grads = []
    for _ in range(2):
        env = FiveTenKMaskedEnv(mode=mode)
        ta = []
        for _ in range(30):
            o, _ = env.reset(); done = False
            while not done:
                ot = torch.FloatTensor(o).unsqueeze(0)
                with torch.no_grad(): d = m.policy.get_distribution(ot)
                a = d.get_actions().item(); no, r, done, trunc, _ = env.step(a)
                ta.append((o, a, r)); o = no
        env.close()
        g = None; n = 0
        for o, a, r in ta:
            ot = torch.FloatTensor(o).unsqueeze(0); d = m.policy.get_distribution(ot)
            lp = d.log_prob(torch.tensor([a])); m.policy.zero_grad(); (-lp * r).backward()
            gv = torch.cat([p.grad.detach().clone().flatten() for p in m.policy.parameters() if p.grad is not None])
            g = gv if g is None else g + gv; n += 1
        grads.append(g / max(n, 1))
    gA, gB = grads; avg = (gA + gB) / 2.0
    return (torch.norm(avg)**2 / max((torch.norm(gA)**2 + torch.norm(gB)**2) / 2.0, 1e-10)).item()

if __name__ == '__main__':
    multiprocessing.freeze_support()

    for mode in ['dynamic']:
        for seed in [45, 46, 47, 48, 49]:
            fp = os.path.join(MODEL_DIR, f'ppo_sa_{mode}_seed{seed}_final.zip')
            if os.path.exists(fp):
                print(f'  SKIP {mode} s{seed}')
                k = compute_kappa_from_model(fp, mode)
                ra = 0  # placeholder
                print(f'  KAPPA: {mode} s{seed}: k={k:.4f}')
                continue
            print(f'  TRAIN {mode} s{seed}...')
            env = SubprocVecEnv([make_ppo_env(mode, seed, i) for i in range(N_ENVS)], start_method='spawn')
            env = VecMonitor(env)
            m = PPO('MlpPolicy', env, learning_rate=3e-4, n_steps=256, batch_size=256, n_epochs=10,
                    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                    policy_kwargs=dict(net_arch=dict(pi=[256,256], vf=[256,256])),
                    verbose=0, seed=seed, device='cuda')
            t0 = time.time(); m.learn(total_timesteps=TS); m.save(fp); env.close()
            print(f'  done {time.time()-t0:.0f}s')
            k = compute_kappa_from_model(fp, mode)
            print(f'  KAPPA: {mode} s{seed}: k={k:.4f}')

    print('\nALL DONE')
