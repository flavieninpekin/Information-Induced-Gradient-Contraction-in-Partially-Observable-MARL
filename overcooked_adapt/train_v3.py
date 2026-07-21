"""V3: STATIC vs DYNAMIC with revealed/hidden partner role."""
import os, sys, time, json, multiprocessing
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from overcooked_v3_env import OvercookedV3Env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models_overcooked_v3')
KAPPA_DIR = os.path.join(os.path.dirname(__file__), '..', 'overcooked_kappa_v3')
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs_overcooked_v3')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(KAPPA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

N_ENVS = 8
MODES = ['static', 'dynamic']
SEEDS = list(range(41, 49))
TOTAL_STEPS = 1_000_000
SAVE_EVERY = 100_000


def make_env_fn(mode, seed, rank):
    def _init():
        return OvercookedV3Env(mode=mode, horizon=400, switch_interval=30,
                               seed=seed + rank * 100)
    return _init


def rollout(env, ptype, model, n_eps=30):
    env._force_partner = ptype
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
    env._force_partner = None
    return trajectories


def grads(model, traj):
    total = None; n = 0
    for ol, al, rl in traj:
        ret = sum(rl)
        for o, a in zip(ol, al):
            ot = torch.FloatTensor(o).unsqueeze(0)
            d = model.policy.get_distribution(ot)
            lp = d.log_prob(torch.tensor([a]))
            model.policy.zero_grad()
            (-lp * ret).backward()
            gv = torch.cat([p.grad.detach().clone().flatten()
                           for p in model.policy.parameters() if p.grad is not None])
            total = gv if total is None else total + gv; n += 1
    return total / max(n, 1)


def kappa(gA, gB):
    avg = (gA + gB) / 2.0
    return (torch.norm(avg)**2 / max((torch.norm(gA)**2 + torch.norm(gB)**2) / 2.0, 1e-10)).item()


if __name__ == '__main__':
    multiprocessing.freeze_support()

    for mode in MODES:
        for seed in SEEDS:
            fp = os.path.join(MODEL_DIR, f'overcookedv3_{mode}_seed{seed}_final.zip')
            if os.path.exists(fp):
                print(f'SKIP {mode} seed{seed}')
                continue
            print(f'TRAIN {mode} seed{seed}...')
            env = SubprocVecEnv([make_env_fn(mode, seed, i) for i in range(N_ENVS)],
                                start_method='spawn')
            env = VecMonitor(env)
            model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=256,
                        batch_size=256, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                        clip_range=0.2, ent_coef=0.01,
                        policy_kwargs=dict(net_arch=dict(pi=[256,256], vf=[256,256])),
                        verbose=0, seed=seed,
                        tensorboard_log=os.path.join(LOG_DIR, mode), device='cuda')
            ckpt = CheckpointCallback(save_freq=SAVE_EVERY, save_path=MODEL_DIR,
                                      name_prefix=f'overcookedv3_{mode}_seed{seed}')
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
            fp = os.path.join(MODEL_DIR, f'overcookedv3_{mode}_seed{seed}_final.zip')
            model = PPO.load(fp, device='cpu')
            # Use same mode as training for rollout (observation dims match)
            env = OvercookedV3Env(mode=mode, horizon=400)

            # Use STATIC mode for eval (revealed partner type ensures fair comparison)
            tc = rollout(env, 'chef', model, 30)
            tw = rollout(env, 'waiter', model, 30)

            # Check if agent actually gets reward
            rc = np.mean([sum(rl) for _,_,rl in tc])
            rw = np.mean([sum(rl) for _,_,rl in tw])

            gc = grads(model, tc)
            gw = grads(model, tw)
            k = kappa(gc, gw)
            results[mode][f'seed{seed}'] = {'kappa': k, 'reward_chef': rc, 'reward_waiter': rw}
            print(f'{mode} seed{seed}:  κ={k:.4f}  r_chef={rc:.1f}  r_waiter={rw:.1f}')
            env.close()

    print(f'\n{"="*60}')
    print('RESULTS')
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
