"""Multi-seed toy with path integrals."""
import numpy as np, json, torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from env.toy_env import HiddenMatchingEnv
from toy_experiment import rollout, compute_pair_gradients, compute_kappa

N_SEEDS = 5
N_STEPS = 8000

print(f'=== Multi-seed Toy + Path ({N_SEEDS} seeds) ===\n')

for mode_name, revealed in [('HIDDEN', False), ('REVEALED', True)]:
    print(f'\n--- {mode_name} ---')
    all_paths, all_kgs, all_kbs, all_rs = [], [], [], []

    for seed in range(N_SEEDS):
        env = HiddenMatchingEnv(revealed=revealed, n_steps=20)
        model = PPO(
            ActorCriticPolicy, env,
            learning_rate=1e-3, n_steps=256, batch_size=32, n_epochs=5,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
            policy_kwargs=dict(net_arch=dict(pi=[32,32], vf=[32,32])),
            verbose=0, seed=seed, device='cpu',
        )

        # Track features at checkpoints for path integral
        p0B_traj, p0C_traj = [], []
        total = 0
        while total < N_STEPS:
            model.learn(total_timesteps=1000, progress_bar=False)
            total += 1000

            env._forced_partner = 0; env.partner = 0
            obs0, _ = env.reset()
            obs_t = torch.FloatTensor(obs0).unsqueeze(0)
            with torch.no_grad():
                p0B = model.policy.get_distribution(obs_t).distribution.probs[0][0].item()

            env._forced_partner = 1; env.partner = 1
            obs1, _ = env.reset()
            obs_t = torch.FloatTensor(obs1).unsqueeze(0)
            with torch.no_grad():
                p0C = model.policy.get_distribution(obs_t).distribution.probs[0][0].item()

            p0B_traj.append(p0B)
            p0C_traj.append(p0C)

        # Path integral
        feats = np.column_stack([p0B_traj, p0C_traj])
        path_len = np.sum(np.linalg.norm(np.diff(feats, axis=0), axis=1))

        # κ measurements at final
        traj_B = rollout(model, env, 0, n_eps=10)
        traj_C = rollout(model, env, 1, n_eps=10)
        g_B, g_C = compute_pair_gradients(model, traj_B, traj_C)
        kg = compute_kappa(g_B, g_C)
        kb = abs(p0B_traj[-1] - p0C_traj[-1])

        # Reward
        rs = []
        for p in [0, 1]:
            for _ in range(30):
                env._forced_partner = p; env.partner = p
                obs, _ = env.reset(); done, r = False, 0
                while not done:
                    obs_t = torch.FloatTensor(obs).unsqueeze(0)
                    with torch.no_grad():
                        a = model.policy.get_distribution(obs_t).get_actions().item()
                    obs, sr, done, _, _ = env.step(a); r += sr
                rs.append(r)
        avg_r = np.mean(rs)

        all_paths.append(path_len); all_kgs.append(kg); all_kbs.append(kb); all_rs.append(avg_r)
        print(f'  s{seed}: PATH={path_len:.4f} kg={kg:.4f} kb={kb:.4f} R={avg_r:+.2f}')
        env.close()

    print(f'  MEAN: PATH={np.mean(all_paths):.4f}+-{np.std(all_paths):.4f}  '
          f'kg={np.mean(all_kgs):.4f}+-{np.std(all_kgs):.4f}  '
          f'kb={np.mean(all_kbs):.4f}+-{np.std(all_kbs):.4f}  '
          f'R={np.mean(all_rs):+.2f}+-{np.std(all_rs):.2f}')
