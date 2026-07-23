"""
510K SAC experiment - stable version.
"""
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '510k-env'))
from env.dqn_wrapper import FiveTenKMaskedEnv, MASK_DIM, MAX_ACTIONS
from env.discrete_sac import DiscreteSAC

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models_510k_sac')
KAPPA_DIR = os.path.join(os.path.dirname(__file__), '..', '510k_kappa_sac')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(KAPPA_DIR, exist_ok=True)

MODES = ['single', 'dynamic']
SEEDS = [41, 42]
TOTAL_STEPS = 500_000
SAVE_EVERY = 100_000


def train_one(mode, seed):
    fp = os.path.join(MODEL_DIR, f'510k_sac_{mode}_seed{seed}.pt')
    if os.path.exists(fp):
        print(f'SKIP {mode} seed{seed}')
        return

    print(f'TRAIN SAC {mode} seed{seed}...')
    sys.stdout.flush()

    env = FiveTenKMaskedEnv(mode=mode)
    obs_dim = env.observation_space.shape[0]
    sac = DiscreteSAC(obs_dim, MASK_DIM, MAX_ACTIONS, lr=3e-4, device='cuda')

    obs, _ = env.reset()
    step = 0
    t0 = time.time()
    next_report = 100_000
    r_buffer = []

    try:
        while step < TOTAL_STEPS:
            action = sac.select_action(obs)
            next_obs, reward, done, trunc, info = env.step(action)
            mask = info.get('action_mask', np.ones(MASK_DIM, dtype=np.float32))
            r_buffer.append(reward)
            sac.buffer.add(obs, action, reward, next_obs, done, mask)
            sac.update(batch_size=64)
            obs = next_obs
            step += 1
            if done or trunc:
                obs, _ = env.reset()
            if step >= next_report:
                elapsed = time.time() - t0
                fps = step / elapsed
                avg_r = np.mean(r_buffer[-10000:]) if r_buffer else 0
                print(f'  {mode} s{seed}: {step//1000}k steps  '
                      f'{fps:.0f}fps  r={avg_r:.3f}  α={sac.alpha_val:.3f}')
                sys.stdout.flush()
                next_report += 100_000
            if step % SAVE_EVERY == 0:
                ckpt = os.path.join(MODEL_DIR, f'510k_sac_{mode}_seed{seed}_{step}_steps.pt')
                sac.save(ckpt)
        sac.save(fp)
        elapsed = time.time() - t0
        print(f'  DONE {mode} seed{seed} in {elapsed:.0f}s ({elapsed/60:.1f}m)')
        sys.stdout.flush()
    except Exception as e:
        print(f'  CRASH {mode} seed{seed} at step {step}: {e}')
        import traceback; traceback.print_exc()
        sys.stdout.flush()
    finally:
        env.close()


def rollout(model, env, n_eps=30):
    transitions = []
    for ep in range(n_eps):
        obs, info = env.reset()
        done = False
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(model.device)
            with torch.no_grad():
                act, _ = model.actor.get_action(obs_t, deterministic=True)
            action = act.item()
            next_obs, reward, done, trunc, info = env.step(action)
            transitions.append((obs, action, reward, next_obs, done))
            obs = next_obs
    return transitions


def kappa(gA, gB):
    avg = (gA + gB) / 2.0
    return (torch.norm(avg)**2 / max((torch.norm(gA)**2 + torch.norm(gB)**2) / 2.0, 1e-10)).item()


if __name__ == '__main__':
    for mode in MODES:
        for seed in SEEDS:
            train_one(mode, seed)

    # Kappa
    results = {}
    for mode in MODES:
        results[mode] = {}
        for seed in SEEDS:
            fp = os.path.join(MODEL_DIR, f'510k_sac_{mode}_seed{seed}.pt')
            if not os.path.exists(fp):
                continue
            sac = DiscreteSAC(112 + MASK_DIM, MASK_DIM, MAX_ACTIONS, device='cpu')
            sac.load(fp)
            sac.actor.eval()

            env_a = FiveTenKMaskedEnv(mode=mode)
            ta = rollout(sac, env_a); ra = np.mean([t[2] for t in ta]); env_a.close()
            env_b = FiveTenKMaskedEnv(mode=mode)
            tb = rollout(sac, env_b); rb = np.mean([t[2] for t in tb]); env_b.close()

            # Actor gradient
            obs_a = [t[0] for t in ta]; act_a = [t[1] for t in ta]
            obs_b = [t[0] for t in tb]; act_b = [t[1] for t in tb]
            gA = sac.actor_gradient(obs_a, act_a)
            gB = sac.actor_gradient(obs_b, act_b)
            k = kappa(gA, gB)
            results[mode][f'seed{seed}'] = {'kappa': k, 'rA': ra, 'rB': rb}
            print(f'SAC {mode} s{seed}: κ={k:.4f} rA={ra:.2f} rB={rb:.2f}')

    print(f'\n{"="*60}')
    print('510K SAC KAPPA')
    print(f'{"="*60}')
    for mode in MODES:
        vals = [v['kappa'] for v in results[mode].values()]
        if vals:
            print(f'{mode}: mean={np.mean(vals):.4f} std={np.std(vals):.4f} '
                  f'seeds={[f"{v:.3f}" for v in vals]}')
    with open(os.path.join(KAPPA_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
