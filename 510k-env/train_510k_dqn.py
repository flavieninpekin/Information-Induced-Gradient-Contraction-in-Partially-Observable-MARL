"""
510K DQN experiment: verify gradient contraction with DQN.

Hypothesis: DYNAMIC < SINGLE in kappa, matching PPO results.
Two rollouts with different game seeds = different teammate assignments.

Strategy: concatenate action_mask to observation, use standard DQN.
"""
import os, sys, json, time, multiprocessing
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '510k-env'))
from env.dqn_wrapper import FiveTenKMaskedEnv
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models_510k_dqn')
KAPPA_DIR = os.path.join(os.path.dirname(__file__), '..', '510k_kappa_dqn')
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs_510k_dqn')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(KAPPA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

N_ENVS = 4
MODES = ['single', 'dynamic']
SEEDS = list(range(41, 49))
TOTAL_STEPS = 1_000_000
SAVE_EVERY = 100_000


def make_env_fn(mode, seed, rank):
    def _init():
        return FiveTenKMaskedEnv(mode=mode)
    return _init


def rollout(model, env, n_eps=30):
    """Rollout with deterministic policy."""
    transitions = []
    for ep in range(n_eps):
        obs, info = env.reset(seed=(ep * 100 + hash(str(env)) % 10000))
        done = False
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                q_values = model.q_net(obs_t)
            action = q_values.argmax(dim=1).item()
            next_obs, reward, done, trunc, info = env.step(action)
            transitions.append((obs, action, reward, next_obs, done))
            obs = next_obs
    return transitions


def dqn_gradient(model, transitions):
    if not transitions:
        return torch.zeros(1)
    obs_b = torch.FloatTensor(np.array([t[0] for t in transitions]))
    act_b = torch.tensor([t[1] for t in transitions])
    rew_b = torch.tensor([t[2] for t in transitions], dtype=torch.float32)
    nxt_b = torch.FloatTensor(np.array([t[3] for t in transitions]))
    don_b = torch.tensor([float(t[4]) for t in transitions], dtype=torch.float32)

    with torch.no_grad():
        next_q = model.q_net_target(nxt_b)
        max_q = next_q.max(dim=1)[0]
        targets = rew_b + (1.0 - don_b) * model.gamma * max_q

    q_vals = model.q_net(obs_b)
    q_pred = q_vals[range(len(act_b)), act_b]
    loss = ((q_pred - targets) ** 2).mean()

    model.q_net.zero_grad()
    loss.backward()
    gv = [p.grad.detach().clone().flatten() for p in model.q_net.parameters() if p.grad is not None]
    return torch.cat(gv) if gv else torch.zeros(1)


def kappa(gA, gB):
    avg = (gA + gB) / 2.0
    return (torch.norm(avg)**2 / max((torch.norm(gA)**2 + torch.norm(gB)**2) / 2.0, 1e-10)).item()


if __name__ == '__main__':
    multiprocessing.freeze_support()

    for mode in MODES:
        for seed in SEEDS:
            fp = os.path.join(MODEL_DIR, f'510k_dqn_{mode}_seed{seed}_final.zip')
            if os.path.exists(fp):
                print(f'SKIP {mode} seed{seed}')
                continue
            print(f'TRAIN DQN 510K {mode} seed{seed}...')
            env = SubprocVecEnv([make_env_fn(mode, seed, i) for i in range(N_ENVS)],
                                start_method='spawn')
            env = VecMonitor(env)
            model = DQN(
                "MlpPolicy", env,
                learning_rate=1e-3, buffer_size=50000, learning_starts=5000,
                batch_size=64, tau=0.005, gamma=0.99,
                train_freq=4, gradient_steps=1,
                target_update_interval=500,
                exploration_fraction=0.3,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.02,
                policy_kwargs=dict(net_arch=[256, 256]),
                verbose=0, seed=seed,
                tensorboard_log=os.path.join(LOG_DIR, mode),
                device='cuda',
            )
            ckpt = CheckpointCallback(save_freq=SAVE_EVERY, save_path=MODEL_DIR,
                                      name_prefix=f'510k_dqn_{mode}_seed{seed}')
            t0 = time.time()
            model.learn(total_timesteps=TOTAL_STEPS, callback=ckpt)
            model.save(fp); env.close()
            print(f'  done {time.time()-t0:.0f}s')

    # Kappa
    results = {}
    for mode in MODES:
        results[mode] = {}
        for seed in SEEDS:
            fp = os.path.join(MODEL_DIR, f'510k_dqn_{mode}_seed{seed}_final.zip')
            if not os.path.exists(fp):
                continue
            model = DQN.load(fp, device='cpu')
            model.q_net.eval()

            # Rollout A: seed A → assignment A
            env_a = FiveTenKMaskedEnv(mode=mode)
            ta = rollout(model, env_a, n_eps=30)
            ra = np.mean([t[2] for t in ta])
            env_a.close()

            # Rollout B: seed B → assignment B
            env_b = FiveTenKMaskedEnv(mode=mode)
            tb = rollout(model, env_b, n_eps=30)
            rb = np.mean([t[2] for t in tb])
            env_b.close()

            gA = dqn_gradient(model, ta)
            gB = dqn_gradient(model, tb)
            k = kappa(gA, gB)
            results[mode][f'seed{seed}'] = {'kappa': k, 'rA': ra, 'rB': rb}
            print(f'DQN {mode} s{seed}: κ={k:.4f} rA={ra:.2f} rB={rb:.2f}')

    print(f'\n{"="*60}')
    print('510K DQN KAPPA')
    print(f'{"="*60}')
    for mode in MODES:
        vals = [v['kappa'] for v in results[mode].values()]
        if vals:
            print(f'{mode}: mean={np.mean(vals):.4f} std={np.std(vals):.4f} '
                  f'seeds={[f"{v:.3f}" for v in vals]}')

    with open(os.path.join(KAPPA_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
