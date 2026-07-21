"""Train + Kappa for role-based Overcooked experiment."""
import os, sys, time, json, multiprocessing
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from overcooked_role_env import OvercookedRoleEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models_overcooked_v2')
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs_overcooked_v2')
KAPPA_DIR = os.path.join(os.path.dirname(__file__), '..', 'overcooked_kappa_v2')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(KAPPA_DIR, exist_ok=True)

N_ENVS = 8
MODES = ['single', 'dynamic']
SEEDS = list(range(41, 49))  # 41-48, 8 seeds per mode
TOTAL_STEPS = 1_000_000
SAVE_EVERY = 100_000


def make_env_fn(mode, seed, rank):
    def _init():
        return OvercookedRoleEnv(
            layout_name='cramped_room', mode=mode, horizon=400,
            switch_interval=40, seed=seed + rank * 100,
        )
    return _init


def rollout(env, partner_name, model, n_eps=30):
    env.mode = 'single'
    env._current_partner_idx = {'chef': 0, 'waiter': 1, 'chaos': 2}[partner_name]
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


if __name__ == '__main__':
    multiprocessing.freeze_support()

    # --- TRAIN ---
    for mode in MODES:
        for seed in SEEDS:
            final_path = os.path.join(MODEL_DIR, f'overcookedv2_{mode}_seed{seed}_final.zip')
            if os.path.exists(final_path):
                print(f'SKIP: {mode} seed{seed}')
                continue

            print(f'\n{"="*60}')
            print(f'TRAINING: {mode} seed{seed} ({N_ENVS} envs, {TOTAL_STEPS} steps)')

            env_fns = [make_env_fn(mode, seed, i) for i in range(N_ENVS)]
            env = SubprocVecEnv(env_fns, start_method='spawn')
            env = VecMonitor(env)

            model = PPO(
                "MlpPolicy", env,
                learning_rate=3e-4, n_steps=256, batch_size=256,
                n_epochs=10, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.01,
                policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
                verbose=0, seed=seed,
                tensorboard_log=os.path.join(LOG_DIR, mode),
                device='cuda',
            )

            ckpt_name = f'overcookedv2_{mode}_seed{seed}'
            callback = CheckpointCallback(
                save_freq=SAVE_EVERY, save_path=MODEL_DIR, name_prefix=ckpt_name,
            )

            t0 = time.time()
            model.learn(total_timesteps=TOTAL_STEPS, callback=callback)
            model.save(final_path)
            env.close()
            print(f'  DONE in {time.time()-t0:.0f}s')
            sys.stdout.flush()

    # --- KAPPA ---
    print(f'\n{"="*60}')
    print('COMPUTING KAPPA')
    print(f'{"="*60}')
    results = {}

    for mode in MODES:
        results[mode] = {}
        for seed in SEEDS:
            path = os.path.join(MODEL_DIR, f'overcookedv2_{mode}_seed{seed}_final.zip')
            if not os.path.exists(path):
                results[mode][f'seed{seed}'] = None
                continue

            print(f'{mode} seed{seed}...')
            model = PPO.load(path, device='cpu')
            env = OvercookedRoleEnv(layout_name='cramped_room', mode=mode, horizon=400)

            traj_chef = rollout(env, 'chef', model, n_eps=30)
            traj_waiter = rollout(env, 'waiter', model, n_eps=30)

            gC = compute_grads(model, traj_chef)
            gW = compute_grads(model, traj_waiter)
            k = kappa(gC, gW)
            results[mode][f'seed{seed}'] = k
            print(f'  κ = {k:.4f}')
            env.close()

    # Summary
    print(f'\n{"="*60}')
    print('RESULTS')
    print(f'{"="*60}')
    for mode in MODES:
        vals = [v for v in results[mode].values() if v is not None]
        if vals:
            print(f'{mode:10s}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  '
                  f'seeds={[f"{v:.4f}" for v in vals]}')

    with open(os.path.join(KAPPA_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
