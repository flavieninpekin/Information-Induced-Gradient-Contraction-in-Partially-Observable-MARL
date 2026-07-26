"""
Continuous Reveal: train on OBVIOUS mode, then mask team bits at eval.
Shows κ monotonically increases with revealed information.
"""
import os, sys, time, numpy as np, torch, multiprocessing
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '510k-env'))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from env.env_510k import FiveTenKEnv, MAX_ACTIONS
from env.game import Game, GameMode
from env.obs_utils import obs_for_player, action_mask_for_player

MDIR = os.path.join(os.path.dirname(__file__), '..', 'models_reveal')
os.makedirs(MDIR, exist_ok=True)

class PartialRevealEnv(FiveTenKEnv):
    """OBVIOUS mode env with externally-controlled team-bit masking."""
    def __init__(self, reveal_fraction=1.0, **kwargs):
        super().__init__(mode='obvious', **kwargs)
        self.reveal_fraction = reveal_fraction

    def _get_obs(self):
        obs = super()._get_obs()  # 116 dims (112 + 4 team bits)
        if self.reveal_fraction < 1.0 and self.game:
            # Mask some team bits
            team = obs[-4:].copy()
            mask = (np.random.random(4) < self.reveal_fraction).astype(np.float32)
            obs[-4:] = team * mask
        return obs

def train_one():
    fp = os.path.join(MDIR, 'ppo_obvious_41_final.zip')
    if os.path.exists(fp):
        print(f'MODEL EXISTS: {fp}')
        return fp

    print('Training PPO on OBVIOUS...')
    def mf():
        e = PartialRevealEnv(reveal_fraction=1.0)
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        e = ActionMasker(e, lambda env: env.unwrapped._get_action_mask())
        return e

    env = SubprocVecEnv([mf for _ in range(8)], start_method='spawn')
    env = VecMonitor(env)

    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
    m = MaskablePPO(MaskableActorCriticPolicy, env, learning_rate=3e-4, n_steps=2048, batch_size=64,
                    n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                    policy_kwargs=dict(net_arch=dict(pi=[256,256], vf=[256,256])),
                    verbose=0, seed=41, device='cuda')
    t0 = time.time(); m.learn(total_timesteps=1_000_000); m.save(fp); env.close()
    print(f'  done {time.time()-t0:.0f}s')
    return fp

def rollout_reveal(model, reveal_fraction, n_eps=30):
    """Rollout with specific reveal fraction. model is MaskablePPO."""
    env = FiveTenKEnv(mode='obvious')
    transitions = []
    for _ in range(n_eps):
        obs, info = env.reset()
        done = False
        while not done:
            mask = env._get_action_mask()
            # Mask team bits
            if reveal_fraction < 1.0:
                team = obs[-4:].copy()
                rmask = (np.random.random(4) < reveal_fraction).astype(np.float32)
                obs = np.concatenate([obs[:-4], team * rmask])

            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            next_obs, r, done, trunc, info = env.step(int(action))
            # Also mask next_obs team bits
            if reveal_fraction < 1.0:
                next_team = next_obs[-4:].copy()
                next_rmask = (np.random.random(4) < reveal_fraction).astype(np.float32)
                next_obs = np.concatenate([next_obs[:-4], next_team * next_rmask])

            transitions.append((obs, int(action), r))
            obs = next_obs
    env.close()
    return transitions

def kappa_masked(model, tA, tB):
    grads = []
    for trans in [tA, tB]:
        g = None; n = 0
        for o, a, r in trans:
            ot = torch.FloatTensor(o).unsqueeze(0)
            d = model.policy.get_distribution(ot)
            lp = d.log_prob(torch.tensor([a]))
            model.policy.zero_grad(); (-lp * r).backward()
            gv = torch.cat([p.grad.detach().clone().flatten()
                          for p in model.policy.parameters() if p.grad is not None])
            g = gv if g is None else g + gv; n += 1
        grads.append(g / max(n, 1))
    gA, gB = grads; avg = (gA + gB) / 2.0
    return (torch.norm(avg)**2 / max((torch.norm(gA)**2 + torch.norm(gB)**2) / 2.0, 1e-10)).item()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    model_path = train_one()

    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(model_path, device='cpu')
    model.policy.eval()

    fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
    kappas = []

    print('\n=== CONTINUOUS REVEAL ===')
    for frac in fractions:
        ta = rollout_reveal(model, frac, 30)
        tb = rollout_reveal(model, frac, 30)
        k = kappa_masked(model, ta, tb)
        ra = np.mean([t[2] for t in ta])
        kappas.append(k)
        print(f'  reveal={frac:.2f}:  k={k:.4f}  r={ra:.2f}')

    print(f'\n{"reveal":>10} {"k":>8}')
    for f, k in zip(fractions, kappas):
        print(f'  {f:.2f}     {k:.4f}')

    diffs = np.diff(kappas)
    mono = all(d >= -0.02 for d in diffs)
    print(f'\nMonotonic: {"YES" if mono else "NO"}')
