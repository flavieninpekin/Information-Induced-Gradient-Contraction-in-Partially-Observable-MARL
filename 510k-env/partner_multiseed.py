"""Multi-seed Partner environment: verify hidden=long path + κ measurement."""
import numpy as np, torch, json, os
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from env.partner_env import HiddenPartnerEnv
from toy_experiment import rollout, compute_pair_gradients, compute_kappa

N_SEEDS = 5
N_STEPS = 60000  # partner env is more complex, needs more training

print(f'=== Partner Env: Multi-seed ({N_SEEDS} seeds) ===\n')

for mode_name, revealed in [('HIDDEN', False), ('REVEALED', True)]:
    print(f'\n--- {mode_name} ---')
    all_paths, all_kgs, all_kbs, all_rs, all_hedge = [], [], [], [], []

    for seed in range(N_SEEDS):
        env = HiddenPartnerEnv(revealed=revealed, n_steps=30)
        model = PPO(
            ActorCriticPolicy, env,
            learning_rate=1e-3, n_steps=512, batch_size=64, n_epochs=5,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.02,
            policy_kwargs=dict(net_arch=dict(pi=[64, 32], vf=[64, 32])),
            verbose=0, seed=seed, device='cpu',
        )

        p_traj = []
        total = 0
        while total < N_STEPS:
            model.learn(total_timesteps=2000, progress_bar=False)
            total += 2000

            env._forced_partner = 0; env.partner = 0
            obs0, _ = env.reset()
            obs_t = torch.FloatTensor(obs0).unsqueeze(0)
            with torch.no_grad():
                p0 = model.policy.get_distribution(obs_t).distribution.probs.cpu().numpy()[0]
            env._forced_partner = 1; env.partner = 1
            obs1, _ = env.reset()
            obs_t = torch.FloatTensor(obs1).unsqueeze(0)
            with torch.no_grad():
                p1 = model.policy.get_distribution(obs_t).distribution.probs.cpu().numpy()[0]
            p_traj.append({'steps': total, 'p0': p0.tolist(), 'p1': p1.tolist()})

        # Path integral from hedge probability trajectory
        hedge_traj = np.array([(d['p0'][2] + d['p1'][2]) / 2 for d in p_traj])
        diff_traj = np.array([abs(d['p0'][0] - d['p1'][0]) + abs(d['p0'][1] - d['p1'][1]) for d in p_traj])
        feats = np.column_stack([hedge_traj, diff_traj])
        path_len = np.sum(np.linalg.norm(np.diff(feats, axis=0), axis=1))

        # κ at final
        traj_B = rollout(model, env, 0, n_eps=10)
        traj_C = rollout(model, env, 1, n_eps=10)
        g_B, g_C = compute_pair_gradients(model, traj_B, traj_C)
        kg = compute_kappa(g_B, g_C)
        kb = diff_traj[-1]

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
        hedge_final = hedge_traj[-1]

        all_paths.append(path_len); all_kgs.append(kg); all_kbs.append(kb)
        all_rs.append(avg_r); all_hedge.append(hedge_final)
        print(f'  s{seed}: PATH={path_len:.2f} kg={kg:.4f} kb={kb:.4f} hedge={hedge_final:.3f} R={avg_r:+.1f}')
        env.close()

    print(f'  MEAN: PATH={np.mean(all_paths):.1f}+-{np.std(all_paths):.0f}  '
          f'kg={np.mean(all_kgs):.4f}+-{np.std(all_kgs):.4f}  '
          f'kb={np.mean(all_kbs):.4f}+-{np.std(all_kbs):.4f}  '
          f'hedge={np.mean(all_hedge):.3f}  '
          f'R={np.mean(all_rs):+.1f}+-{np.std(all_rs):.1f}')

print('\n' + '=' * 70)
print('THREE-ENVIRONMENT SUMMARY')
print('=' * 70)
print(f'{"Env":<12} {"Pattern":<25} {"PATH":>8} {"kg":>8} {"Reward":>8} {"Diagnosis":>15}')
print('-' * 70)
print(f'{"510K (DYNAMIC)":<12} {"hidden = SHORT path":<25} {"0.293":>8} {"N/A":>8} {"~82":>8} {"deceptive stability":>15}')
print(f'{"Partner (HIDDEN)":<12} {"hidden = LONG path":<25} {"TBD":>8} {"TBD":>8} {"TBD":>8} {"spurious exploration":>15}')
print(f'{"Toy (HIDDEN)":<12} {"hidden = kg→0":<25} {"N/A":>8} {"0.0007":>8} {"0":>8} {"gradient death":>15}')
