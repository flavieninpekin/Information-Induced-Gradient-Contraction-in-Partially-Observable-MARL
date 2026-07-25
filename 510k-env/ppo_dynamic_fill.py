"""510K PPO DYNAMIC: single-agent training + kappa."""
import os, sys, numpy as np, torch, time, multiprocessing
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '510k-env'))
from env.dqn_wrapper import FiveTenKMaskedEnv, MASK_DIM, MAX_ACTIONS
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models_510k_ppo_sa')
MDIR = os.path.join(os.path.dirname(__file__), '..', 'models_selfplay')
os.makedirs(MODEL_DIR, exist_ok=True)

N_ENVS = 8; TS = 1_000_000

def make_env(mode, seed, rank):
    def _(): return FiveTenKMaskedEnv(mode=mode)
    return _

def rollout(model, env, n_eps=30):
    traj = []
    for _ in range(n_eps):
        o, _ = env.reset(); done = False
        while not done:
            ot = torch.FloatTensor(o).unsqueeze(0)
            with torch.no_grad(): d = model.policy.get_distribution(ot)
            a = d.get_actions().item()
            no, r, done, trunc, _ = env.step(a)
            traj.append((o, a, r, no, done)); o = no
    return traj

def kappa(model, tA, tB):
    grads = []
    for traj in [tA, tB]:
        g = None; n = 0
        for o, a, r, _, _ in traj:
            ot = torch.FloatTensor(o).unsqueeze(0); d = model.policy.get_distribution(ot)
            lp = d.log_prob(torch.tensor([a])); model.policy.zero_grad(); (-lp * r).backward()
            gv = torch.cat([p.grad.detach().clone().flatten() for p in model.policy.parameters() if p.grad is not None])
            g = gv if g is None else g + gv; n += 1
        grads.append(g / max(n, 1))
    gA, gB = grads; avg = (gA + gB) / 2.0
    return (torch.norm(avg)**2 / max((torch.norm(gA)**2 + torch.norm(gB)**2) / 2.0, 1e-10)).item()

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Train DYNAMIC (single-agent PPO, no self-play)
    for mode in ['dynamic']:
        for seed in [41, 42, 43, 44]:
            fp = os.path.join(MODEL_DIR, f'ppo_sa_{mode}_seed{seed}_final.zip')
            if os.path.exists(fp): print(f'SKIP {mode} s{seed}'); continue
            print(f'PPO SA {mode} s{seed}...')
            env = SubprocVecEnv([make_env(mode, seed, i) for i in range(N_ENVS)], start_method='spawn')
            env = VecMonitor(env)
            m = PPO('MlpPolicy', env, learning_rate=3e-4, n_steps=256, batch_size=256, n_epochs=10,
                    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                    policy_kwargs=dict(net_arch=dict(pi=[256,256], vf=[256,256])),
                    verbose=0, seed=seed, device='cuda')
            t0 = time.time(); m.learn(total_timesteps=TS); m.save(fp); env.close()
            print(f'  done {time.time()-t0:.0f}s')

    # Kappa: SINGLE from saved model, DYNAMIC from new models
    print('\n=== KAPPA ===')
    # Load existing SINGLE model
    sfp = os.path.join(MDIR, '510k_single_final.zip')
    if os.path.exists(MDIR):
        # Try old models path
        try:
            from sb3_contrib import MaskablePPO
            sm = MaskablePPO.load(r'C:\Users\Flavi\llmprojects\project3\models\510k_single_final.zip', device='cpu')
            env = FiveTenKMaskedEnv(mode='single')
            ta = rollout(sm, env, 30); env.close()
            env2 = FiveTenKMaskedEnv(mode='single')
            tb = rollout(sm, env2, 30); env2.close()
            ks = kappa(sm, ta, tb)
            print(f'PPO SINGLE (old): k={ks:.4f}')
        except Exception as e:
            print(f'SINGLE load failed: {e}')
            ks = None

    # New DYNAMIC models
    for mode in ['dynamic']:
        for seed in [41, 42, 43, 44]:
            fp = os.path.join(MODEL_DIR, f'ppo_sa_{mode}_seed{seed}_final.zip')
            if not os.path.exists(fp): continue
            m = PPO.load(fp, device='cpu')
            env = FiveTenKMaskedEnv(mode=mode)
            ta = rollout(m, env, 30); env.close()
            env2 = FiveTenKMaskedEnv(mode=mode)
            tb = rollout(m, env2, 30); env2.close()
            kd = kappa(m, ta, tb)
            ra = np.mean([t[2] for t in ta]); rb = np.mean([t[2] for t in tb])
            print(f'PPO DYNAMIC s{seed}: k={kd:.4f} rA={ra:.2f} rB={rb:.2f}')

    # Also compute SINGLE from the existing models/ checkpoints
    try:
        m2 = PPO.load(r'C:\Users\Flavi\llmprojects\project3\models\510k_single_114688_steps.zip', device='cpu')
        env = FiveTenKMaskedEnv(mode='single')
        ta = rollout(m2, env, 30); env.close()
        env2 = FiveTenKMaskedEnv(mode='single')
        tb = rollout(m2, env2, 30); env2.close()
        ks2 = kappa(m2, ta, tb)
        print(f'PPO SINGLE (models/ checkpoint): k={ks2:.4f}')
    except: pass
