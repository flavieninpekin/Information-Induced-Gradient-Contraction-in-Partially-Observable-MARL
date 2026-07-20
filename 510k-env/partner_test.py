"""Quick test: does hidden partner env show short-path-under-hidden pattern?"""
import numpy as np, torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from env.partner_env import HiddenPartnerEnv

for mode_name, revealed in [('HIDDEN', False), ('REVEALED', True)]:
    print(f'\n=== {mode_name} ===')
    env = HiddenPartnerEnv(revealed=revealed, n_steps=30)

    model = PPO(
        ActorCriticPolicy, env,
        learning_rate=1e-3, n_steps=512, batch_size=64, n_epochs=5,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.02,
        policy_kwargs=dict(net_arch=dict(pi=[64, 32], vf=[64, 32])),
        verbose=0, seed=42, device='cpu',
    )

    p_traj = []
    for phase in range(30):
        model.learn(total_timesteps=2048, progress_bar=False)
        steps = (phase + 1) * 2048

        # Measure action probs under both partners
        probs = []
        for partner in [0, 1]:
            env._forced_partner = partner; env.partner = partner
            obs, _ = env.reset()
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                p = model.policy.get_distribution(obs_t).distribution.probs.cpu().numpy()[0]
            probs.append(p)
        p_diff = abs(probs[0][0] - probs[1][0]) + abs(probs[0][1] - probs[1][1])
        
        # Reward
        rs = []
        for p in [0, 1]:
            for _ in range(20):
                env._forced_partner = p; env.partner = p
                obs, _ = env.reset(); done, r = False, 0
                while not done:
                    obs_t = torch.FloatTensor(obs).unsqueeze(0)
                    with torch.no_grad():
                        a = model.policy.get_distribution(obs_t).get_actions().item()
                    obs, sr, done, _, _ = env.step(a); r += sr
                rs.append(r)
        avg_r = np.mean(rs)
        hedge_prob = (probs[0][2] + probs[1][2]) / 2
        p_traj.append({'steps': steps, 'hedge': hedge_prob, 'p_diff': p_diff, 'reward': avg_r})
    
    # Path integral (from hedge probability trajectory + p_diff trajectory)
    hedge_traj = np.array([d['hedge'] for d in p_traj])
    diff_traj = np.array([d['p_diff'] for d in p_traj])
    feats = np.column_stack([hedge_traj, diff_traj])
    path_len = np.sum(np.linalg.norm(np.diff(feats, axis=0), axis=1))

    final = p_traj[-1]
    print(f'  PATH={path_len:.4f}')
    print(f'  Final reward={final["reward"]:+.2f}  hedge_prob={final["hedge"]:.3f}  p_diff={final["p_diff"]:.3f}')
    env.close()
