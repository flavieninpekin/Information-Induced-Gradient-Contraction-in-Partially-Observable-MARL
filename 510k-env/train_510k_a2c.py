"""
A2C on 510K with action masking.
Parallel envs + kappa from REINFORCE gradient.
"""
import os, sys, json, time, multiprocessing
import numpy as np, torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '510k-env'))
from env.dqn_wrapper import FiveTenKMaskedEnv, MASK_DIM, MAX_ACTIONS
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models_510k_a2c')
KAPPA_DIR = os.path.join(os.path.dirname(__file__), '..', '510k_kappa_a2c')
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs_510k_a2c')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(KAPPA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

N_ENVS = 8
MODES = ['single', 'dynamic']
SEEDS = list(range(41, 49))
TOTAL_STEPS = 1_000_000
SAVE_EVERY = 100_000


def make_env_fn(mode, seed, rank):
    def _init():
        return FiveTenKMaskedEnv(mode=mode)
    return _init


def rollout(model, env, n_eps=30):
    transitions = []
    for ep in range(n_eps):
        obs, info = env.reset()
        done = False
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                distribution = model.policy.get_distribution(obs_t)
            action = distribution.get_actions().item()
            next_obs, reward, done, trunc, info = env.step(action)
            transitions.append((obs, action, reward, next_obs, done))
            obs = next_obs
    return transitions


def pair_grads(model, traj_A, traj_B):
    grads = []
    for traj in [traj_A, traj_B]:
        total_grad = None; n = 0
        for obs, act, rew, _, _ in traj:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            distribution = model.policy.get_distribution(obs_t)
            log_prob = distribution.log_prob(torch.tensor([act]))
            model.policy.zero_grad()
            (-log_prob * rew).backward()
            gv = torch.cat([p.grad.detach().clone().flatten()
                          for p in model.policy.parameters() if p.grad is not None])
            total_grad = gv if total_grad is None else total_grad + gv; n += 1
        grads.append(total_grad / max(n, 1))
    return grads[0], grads[1]


def kappa(gA, gB):
    avg = (gA + gB) / 2.0
    return (torch.norm(avg)**2 / max((torch.norm(gA)**2 + torch.norm(gB)**2) / 2.0, 1e-10)).item()


if __name__ == '__main__':
    multiprocessing.freeze_support()

    for mode in MODES:
        for seed in SEEDS:
            fp = os.path.join(MODEL_DIR, f'510k_a2c_{mode}_seed{seed}_final.zip')
            if os.path.exists(fp):
                print(f'SKIP {mode} seed{seed}')
                continue
            print(f'TRAIN A2C {mode} seed{seed}...')
            env = SubprocVecEnv([make_env_fn(mode, seed, i) for i in range(N_ENVS)], start_method='spawn')
            env = VecMonitor(env)
            model = A2C("MlpPolicy", env, learning_rate=3e-4, n_steps=256,
                        gamma=0.99, gae_lambda=0.95, ent_coef=0.01,
                        policy_kwargs=dict(net_arch=dict(pi=[256,256], vf=[256,256])),
                        verbose=0, seed=seed,
                        tensorboard_log=os.path.join(LOG_DIR, mode), device='cuda')
            ckpt = CheckpointCallback(save_freq=SAVE_EVERY, save_path=MODEL_DIR,
                                      name_prefix=f'510k_a2c_{mode}_seed{seed}')
            t0 = time.time()
            model.learn(total_timesteps=TOTAL_STEPS, callback=ckpt)
            model.save(fp); env.close()
            print(f'  done {time.time()-t0:.0f}s')
            sys.stdout.flush()

    results = {}
    for mode in MODES:
        results[mode] = {}
        for seed in SEEDS:
            fp = os.path.join(MODEL_DIR, f'510k_a2c_{mode}_seed{seed}_final.zip')
            if not os.path.exists(fp): continue
            model = A2C.load(fp, device='cpu')
            env_a = FiveTenKMaskedEnv(mode=mode)
            ta = rollout(model, env_a); ra = np.mean([t[2] for t in ta]); env_a.close()
            env_b = FiveTenKMaskedEnv(mode=mode)
            tb = rollout(model, env_b); rb = np.mean([t[2] for t in tb]); env_b.close()
            gA, gB = pair_grads(model, ta, tb)
            k = kappa(gA, gB)
            results[mode][f'seed{seed}'] = {'kappa': k, 'rA': ra, 'rB': rb}
            print(f'A2C {mode} s{seed}: κ={k:.4f} rA={ra:.2f} rB={rb:.2f}')

    print(f'\n{"="*60}')
    print('510K A2C KAPPA')
    print(f'{"="*60}')
    for mode in MODES:
        vals = [v['kappa'] for v in results[mode].values()]
        if vals:
            print(f'{mode}: mean={np.mean(vals):.4f} std={np.std(vals):.4f} '
                  f'seeds={[f"{v:.3f}" for v in vals]}')
    with open(os.path.join(KAPPA_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
