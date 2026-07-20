"""
Quick experiment: train SINGLE + DYNAMIC, compute κ.
One seed each, 200k steps.
"""
import os, sys, time, json, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from overcooked_wrapper import OvercookedHiddenPartner

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models_overcooked')
RESULT_DIR = os.path.join(os.path.dirname(__file__), '..', 'overcooked_kappa')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

SEED = 41
TIMESTEPS = 200_000
N_EPS_KAPPA = 30

# ========== STEP 1: TRAIN ==========

for mode in ['single', 'dynamic']:
    print(f'\n{"="*50}')
    print(f'TRAINING {mode.upper()} MODE')
    print(f'{"="*50}')

    env = OvercookedHiddenPartner(
        layout_name='cramped_room', mode=mode, horizon=400,
    )

    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4, n_steps=2048, batch_size=64,
        n_epochs=10, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.01,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=1, seed=SEED,
        tensorboard_log=os.path.join(os.path.dirname(__file__), '..', 'logs_overcooked', mode),
    )

    t0 = time.time()
    model.learn(total_timesteps=TIMESTEPS)
    elapsed = time.time() - t0

    path = os.path.join(MODEL_DIR, f'overcooked_{mode}_seed{SEED}_final.zip')
    model.save(path)
    print(f'  Saved: {path}  ({elapsed:.0f}s)')
    env.close()

# ========== STEP 2: KAPPA ==========

print(f'\n{"="*50}')
print('COMPUTING KAPPA')
print(f'{"="*50}')

def rollout(env, partner_idx, n_eps=30):
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

def pair_grads(model, traj_A, traj_B):
    grads = []
    for traj in [traj_A, traj_B]:
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
                gv = torch.cat([p.grad.detach().clone().flatten() for p in model.policy.parameters() if p.grad is not None])
                total_grad = gv if total_grad is None else total_grad + gv
                n += 1
        grads.append(total_grad / max(n, 1))
    return grads[0], grads[1]

def kappa(gA, gB):
    avg = (gA + gB) / 2.0
    return (torch.norm(avg)**2 / max((torch.norm(gA)**2 + torch.norm(gB)**2) / 2.0, 1e-10)).item()

results = {}
for mode in ['single', 'dynamic']:
    model_path = os.path.join(MODEL_DIR, f'overcooked_{mode}_seed{SEED}_final.zip')
    print(f'\nLoading {mode} model: {model_path}')
    model = PPO.load(model_path, device='cpu')

    env = OvercookedHiddenPartner(
        layout_name='cramped_room', mode=mode, horizon=400,
    )

    print(f'  Rollout greedy partner...')
    traj_g = rollout(env, partner_idx=0, n_eps=N_EPS_KAPPA)
    print(f'  Rollout random partner...')
    traj_r = rollout(env, partner_idx=1, n_eps=N_EPS_KAPPA)

    gG, gR = pair_grads(model, traj_g, traj_r)
    k = kappa(gG, gR)
    results[mode] = k
    print(f'  SINGLE κ = {k:.4f}' if mode == 'single' else f'  DYNAMIC κ = {k:.4f}')
    env.close()

# Save
with open(os.path.join(RESULT_DIR, 'quick_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f'\n{"="*50}')
print('RESULTS')
print(f'{"="*50}')
for mode, k in results.items():
    print(f'  {mode:10s}: κ = {k:.4f}')

if results.get('single', 0) > results.get('dynamic', 0):
    print(f'\n  SINGLE κ ({results["single"]:.4f}) > DYNAMIC κ ({results["dynamic"]:.4f})')
    print('  Hypothesis CONFIRMED: DYNAMIC shows gradient contraction.')
else:
    print(f'\n  SINGLE κ ({results.get("single",0):.4f}) <= DYNAMIC κ ({results.get("dynamic",0):.4f})')
    print('  Hypothesis NOT confirmed - need more seeds/steps.')
