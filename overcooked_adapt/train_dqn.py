"""
Overcooked DQN kappa experiment - mirrors train_v3.py with DQN.

STATIC (revealed): partner type visible → agent specializes → κ ≈ 0.5
DYNAMIC (hidden): partner switches hidden → gradient death → κ → 0
"""
import os, sys, time, json, multiprocessing
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from overcooked_v3_env import OvercookedV3Env
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models_overcooked_dqn')
KAPPA_DIR = os.path.join(os.path.dirname(__file__), '..', 'overcooked_kappa_dqn')
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs_overcooked_dqn')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(KAPPA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

N_ENVS = 4
MODES = ['static', 'dynamic']
SEEDS = list(range(41, 49))
TOTAL_STEPS = 1_000_000
SAVE_EVERY = 100_000


def make_env_fn(mode, seed, rank):
    def _init():
        return OvercookedV3Env(mode=mode, horizon=400, switch_interval=30,
                               seed=seed + rank * 100)
    return _init


def rollout_dqn(env, ptype, model, n_eps=30):
    """Rollout with fixed partner, DQN deterministic policy."""
    env._force_partner = ptype
    transitions = []
    for _ in range(n_eps):
        obs, _ = env.reset()
        done = False
        ep = []
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                q_values = model.q_net(obs_t)
            action = q_values.argmax(dim=1).item()
            next_obs, reward, done, trunc, _ = env.step(action)
            ep.append((obs, action, reward, next_obs, done))
            obs = next_obs
        transitions.extend(ep)
    env._force_partner = None
    return transitions


def dqn_gradient(model, transitions):
    """TD(0) loss gradient on Q-network for a list of (s,a,r,s',done)."""
    if not transitions:
        return torch.zeros(1)

    batch_obs = torch.FloatTensor(np.array([t[0] for t in transitions]))
    batch_acts = torch.tensor([t[1] for t in transitions])
    batch_rews = torch.tensor([t[2] for t in transitions], dtype=torch.float32)
    batch_next = torch.FloatTensor(np.array([t[3] for t in transitions]))
    batch_dones = torch.tensor([float(t[4]) for t in transitions], dtype=torch.float32)

    with torch.no_grad():
        next_q = model.q_net_target(batch_next)
        max_next_q = next_q.max(dim=1)[0]
        targets = batch_rews + (1.0 - batch_dones) * model.gamma * max_next_q

    q_values = model.q_net(batch_obs)
    q_pred = q_values[range(len(batch_acts)), batch_acts]
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
            fp = os.path.join(MODEL_DIR, f'overcooked_dqn_{mode}_seed{seed}_final.zip')
            if os.path.exists(fp):
                print(f'SKIP {mode} seed{seed}')
                continue
            print(f'TRAIN DQN {mode} seed{seed}...')
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
                                      name_prefix=f'overcooked_dqn_{mode}_seed{seed}')
            t0 = time.time()
            model.learn(total_timesteps=TOTAL_STEPS, callback=ckpt)
            model.save(fp); env.close()
            print(f'  done {time.time()-t0:.0f}s')
            sys.stdout.flush()

    # Kappa
    results = {}
    for mode in MODES:
        results[mode] = {}
        for seed in SEEDS:
            fp = os.path.join(MODEL_DIR, f'overcooked_dqn_{mode}_seed{seed}_final.zip')
            model = DQN.load(fp, device='cpu')
            model.q_net.eval()
            env = OvercookedV3Env(mode=mode, horizon=400)

            tc = rollout_dqn(env, 'chef', model, 30)
            tw = rollout_dqn(env, 'waiter', model, 30)

            rc = sum(t[2] for t in tc) / 30
            rw = sum(t[2] for t in tw) / 30

            gc = dqn_gradient(model, tc)
            gw = dqn_gradient(model, tw)
            k = kappa(gc, gw)
            results[mode][f'seed{seed}'] = {'kappa': k, 'reward_chef': rc, 'reward_waiter': rw}
            print(f'DQN {mode} seed{seed}:  κ={k:.4f}  r_chef={rc:.1f}  r_waiter={rw:.1f}')
            env.close()

    print(f'\n{"="*60}')
    print('DQN RESULTS')
    print(f'{"="*60}')
    for mode in MODES:
        vals = [v['kappa'] for v in results[mode].values()]
        rcs = [v['reward_chef'] for v in results[mode].values()]
        rws = [v['reward_waiter'] for v in results[mode].values()]
        print(f'{mode}:')
        print(f'  κ:   mean={np.mean(vals):.4f}  std={np.std(vals):.4f}')
        print(f'  r_c: mean={np.mean(rcs):.1f}  std={np.std(rcs):.1f}')
        print(f'  r_w: mean={np.mean(rws):.1f}  std={np.std(rws):.1f}')
        print(f'  seeds: {[f"{v:.4f}" for v in vals]}')

    with open(os.path.join(KAPPA_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
