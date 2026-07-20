"""Multi-seed toy experiment to verify robustness."""
import numpy as np, json, os, torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from env.toy_env import HiddenMatchingEnv


def rollout(model, env, partner, n_eps=10):
    env._forced_partner = partner
    trajectories = []
    for _ in range(n_eps):
        env.partner = partner
        obs, _ = env.reset()
        done = False
        obs_list, act_list, rew_list = [], [], []
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                distribution = model.policy.get_distribution(obs_t)
            action = distribution.get_actions().item()
            obs_list.append(obs.copy())
            act_list.append(action)
            obs, r, done, _, _ = env.step(action)
            rew_list.append(r)
        trajectories.append((obs_list, act_list, rew_list))
    return trajectories


def compute_pair_gradients(model, traj_B, traj_C):
    grads = []
    for traj in [traj_B, traj_C]:
        total_grad = None
        n_samples = 0
        for obs_list, act_list, rew_list in traj:
            for obs, act, ret in zip(obs_list, act_list, rew_list):
                obs_t = torch.FloatTensor(obs).unsqueeze(0)
                distribution = model.policy.get_distribution(obs_t)
                log_prob = distribution.log_prob(torch.tensor([act]))
                model.policy.zero_grad()
                (-log_prob * ret).backward()
                grad_vec = []
                for p in model.policy.parameters():
                    if p.grad is not None:
                        grad_vec.append(p.grad.detach().clone().flatten())
                if grad_vec:
                    g_flat = torch.cat(grad_vec)
                    total_grad = g_flat if total_grad is None else total_grad + g_flat
                    n_samples += 1
        grads.append(total_grad / max(n_samples, 1))
    return grads[0], grads[1]


def compute_kappa(grad_B, grad_C):
    avg = (grad_B + grad_C) / 2.0
    num = torch.norm(avg) ** 2
    denom = (torch.norm(grad_B) ** 2 + torch.norm(grad_C) ** 2) / 2.0
    return (num / max(denom, 1e-10)).item()


# ============================================================
N_SEEDS = 5
N_STEPS = 5000

print(f'=== Multi-seed Toy ({N_SEEDS} seeds x {N_STEPS} steps) ===\n')
all_results = {}

for mode_name, revealed in [('HIDDEN', False), ('REVEALED', True)]:
    print(f'\n--- {mode_name} ---')
    seed_results = []
    for seed in range(N_SEEDS):
        env = HiddenMatchingEnv(revealed=revealed, n_steps=20)
        model = PPO(
            ActorCriticPolicy, env,
            learning_rate=1e-3, n_steps=256, batch_size=32, n_epochs=5,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
            policy_kwargs=dict(net_arch=dict(pi=[32, 32], vf=[32, 32])),
            verbose=0, seed=seed, device='cpu',
        )
        model.learn(total_timesteps=N_STEPS, progress_bar=False)

        # κ_gradient
        traj_B = rollout(model, env, 0, n_eps=10)
        traj_C = rollout(model, env, 1, n_eps=10)
        g_B, g_C = compute_pair_gradients(model, traj_B, traj_C)
        kg = compute_kappa(g_B, g_C)

        # κ_behavioral & reward
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
        kb = abs(p0B - p0C)

        rewards = []
        for p in [0, 1]:
            for _ in range(30):
                env._forced_partner = p; env.partner = p
                obs, _ = env.reset(); done, r = False, 0
                while not done:
                    obs_t = torch.FloatTensor(obs).unsqueeze(0)
                    with torch.no_grad():
                        a = model.policy.get_distribution(obs_t).get_actions().item()
                    obs, sr, done, _, _ = env.step(a); r += sr
                rewards.append(r)
        avg_r = np.mean(rewards)

        seed_results.append({
            'seed': seed, 'kappa_grad': kg, 'kappa_beh': kb,
            'p0B': p0B, 'p0C': p0C, 'reward': avg_r,
            'g_norm_B': torch.norm(g_B).item(), 'g_norm_C': torch.norm(g_C).item(),
        })
        print(f'  s{seed}: kg={kg:.4f} kb={kb:.4f} p0B={p0B:.3f} p0C={p0C:.3f} R={avg_r:+.2f}')
        env.close()

    kgs = [r['kappa_grad'] for r in seed_results]
    kbs = [r['kappa_beh'] for r in seed_results]
    rs = [r['reward'] for r in seed_results]
    print(f'  MEAN+STD: kg={np.mean(kgs):.4f}+-{np.std(kgs):.4f}  kb={np.mean(kbs):.4f}+-{np.std(kbs):.4f}  R={np.mean(rs):+.2f}+-{np.std(rs):.2f}')
    all_results[mode_name] = seed_results

print('\n' + '=' * 60)
for name in ['HIDDEN', 'REVEALED']:
    data = all_results[name]
    kgs = [r['kappa_grad'] for r in data]
    kbs = [r['kappa_beh'] for r in data]
    rs = [r['reward'] for r in data]
    print(f'{name:>8}: kg={np.mean(kgs):.4f}+-{np.std(kgs):.4f}  kb={np.mean(kbs):.4f}+-{np.std(kbs):.4f}  R={np.mean(rs):+.2f}+-{np.std(rs):.2f}')

with open('toy_results/multi_seed.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print('\nSaved.')
