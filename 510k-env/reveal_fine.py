"""
Fine-grained continuous reveal: train at 0%, 25%, 50%, 75%, 100%.
Uses MaskablePPO + ActionMasker for action-masked training.
"""
import os, sys, time, multiprocessing, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '510k-env'))

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from env.env_510k import FiveTenKEnv

MDIR = os.path.join(os.path.dirname(__file__), '..', 'models_reveal')
os.makedirs(MDIR, exist_ok=True)

REVEAL_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEEDS = [41, 42]
N_ENVS = 8
TS = 1_000_000


class RevealEnv(FiveTenKEnv):
    def __init__(self, reveal_fraction=1.0, **kw):
        super().__init__(mode='obvious', **kw)
        self.reveal_fraction = reveal_fraction

    def _get_obs(self):
        obs = super()._get_obs()
        if self.reveal_fraction < 1.0 and self.game:
            team = obs[-4:].copy()
            mask = (np.random.random(4) < self.reveal_fraction).astype(np.float32)
            obs[-4:] = team * mask
        return obs


def train_one(reveal_frac, seed):
    name = f'{reveal_frac:.2f}'
    fp = os.path.join(MDIR, f'ppo_reveal_{name}_s{seed}.zip')
    if os.path.exists(fp):
        print(f'SKIP reveal={name} s{seed}')
        return fp

    print(f'TRAIN reveal={name} s{seed}...')
    sys.stdout.flush()

    def mf():
        e = RevealEnv(reveal_fraction=reveal_frac)
        e = ActionMasker(e, lambda env: env.unwrapped._get_action_mask())
        return e

    env = SubprocVecEnv([mf for _ in range(N_ENVS)], start_method='spawn')
    env = VecMonitor(env)

    m = MaskablePPO(MaskableActorCriticPolicy, env, learning_rate=3e-4, n_steps=2048, batch_size=64,
                    n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
                    verbose=0, seed=seed, device='cuda')
    t0 = time.time()
    m.learn(total_timesteps=TS)
    m.save(fp)
    env.close()
    elapsed = time.time() - t0
    print(f'  done {elapsed:.0f}s ({elapsed/60:.1f}m)')
    sys.stdout.flush()
    return fp


def compute_kappa(maskable_model, n_eps=30):
    """Computes kappa for a MaskablePPO model."""
    grads = []
    for _ in range(2):
        env = FiveTenKEnv(mode='obvious')
        g = None; n = 0
        for _ in range(n_eps):
            o, info = env.reset()
            done = False
            while not done:
                mask = env._get_action_mask()
                ot = torch.FloatTensor(o).unsqueeze(0)
                d = maskable_model.policy.get_distribution(ot)
                a = d.get_actions().item()
                no, r, done, trunc, info = env.step(int(a))
                d2 = maskable_model.policy.get_distribution(torch.FloatTensor(o).unsqueeze(0))
                lp = d2.log_prob(torch.tensor([a]))
                maskable_model.policy.zero_grad()
                (-lp * r).backward()
                gv = torch.cat([p.grad.detach().clone().flatten()
                              for p in maskable_model.policy.parameters() if p.grad is not None])
                g = gv if g is None else g + gv
                n += 1
                o = no
        env.close()
        grads.append(g / max(n, 1))
    gA, gB = grads
    avg = (gA + gB) / 2.0
    return (torch.norm(avg)**2 / max((torch.norm(gA)**2 + torch.norm(gB)**2) / 2.0, 1e-10)).item()


if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Train all missing models
    for frac in REVEAL_LEVELS:
        for seed in SEEDS:
            train_one(frac, seed)

    # Compute kappa for all
    print('\n=== KAPPA CURVE ===')
    results = {}
    for frac in REVEAL_LEVELS:
        vals = []
        for seed in SEEDS:
            name = f'{frac:.2f}'
            fp = os.path.join(MDIR, f'ppo_reveal_{name}_s{seed}.zip')
            if not os.path.exists(fp):
                continue
            m = MaskablePPO.load(fp, device='cpu')
            m.policy.eval()
            k = compute_kappa(m, 30)
            vals.append(k)
            print(f'  reveal={name} s{seed}: k={k:.4f}')
        if vals:
            results[frac] = {'mean': np.mean(vals), 'std': np.std(vals), 'n': len(vals)}

    print(f'\n{"="*50}')
    print(f'{"Reveal":>8}  {"kappa_mean":>10}  {"std":>8}  {"n":>4}')
    print(f'{"="*50}')
    for frac in REVEAL_LEVELS:
        if frac in results:
            r = results[frac]
            bar = '█' * int(r['mean'] * 40)
            print(f'{frac:>8.0%}  {r["mean"]:>10.4f}  {r["std"]:>8.4f}  {r["n"]:>4}  {bar}')
    print(f'{"="*50}')

    # Shape detection
    ks = [results[f]['mean'] for f in REVEAL_LEVELS if f in results]
    if len(ks) >= 3:
        if ks[-1] > ks[0] and min(ks) < min(ks[0], ks[-1]):
            print('Shape: U-SHAPED (noisy info penalty detected)')
        elif ks == sorted(ks):
            print('Shape: MONOTONIC (more info = higher kappa)')
        else:
            print(f'Shape: IRREGULAR (values: {[f"{k:.3f}" for k in ks]})')
