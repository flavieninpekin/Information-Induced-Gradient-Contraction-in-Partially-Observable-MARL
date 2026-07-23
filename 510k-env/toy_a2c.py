"""
A2C on Toy Matching: verify gradient contraction pattern.
One-line change from toy_experiment.py: PPO -> A2C.
"""
import os, json, numpy as np, torch
from stable_baselines3 import A2C
from env.toy_env import HiddenMatchingEnv

OUTPUT_DIR = 'toy_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def rollout(model, env, partner, n_eps=30):
    env._forced_partner = partner
    trajectories = []
    for _ in range(n_eps):
        env.partner = partner
        obs, _ = env.reset()
        done = False
        olist, alist, rlist = [], [], []
        while not done:
            olist.append(obs.copy())
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                distribution = model.policy.get_distribution(obs_t)
            action = distribution.get_actions().item()
            alist.append(action)
            obs, r, done, _, _ = env.step(action)
            rlist.append(r)
        trajectories.append((olist, alist, rlist))
    return trajectories


def pair_grads(model, traj_B, traj_C):
    grads = []
    for traj in [traj_B, traj_C]:
        total_grad = None; n = 0
        for olist, alist, rlist in traj:
            ret = sum(rlist)
            for obs, act in zip(olist, alist):
                obs_t = torch.FloatTensor(obs).unsqueeze(0)
                distribution = model.policy.get_distribution(obs_t)
                log_prob = distribution.log_prob(torch.tensor([act]))
                model.policy.zero_grad()
                (-log_prob * ret).backward()
                gv = torch.cat([p.grad.detach().clone().flatten()
                               for p in model.policy.parameters() if p.grad is not None])
                total_grad = gv if total_grad is None else total_grad + gv
                n += 1
        grads.append(total_grad / max(n, 1))
    return grads[0], grads[1]


def kappa(gB, gC):
    avg = (gB + gC) / 2.0
    return (torch.norm(avg)**2 / max((torch.norm(gB)**2 + torch.norm(gC)**2) / 2.0, 1e-10)).item()


def avg_reward(model, env, n_both=30):
    rewards = []
    for partner in [0, 1]:
        for _ in range(n_both):
            env._forced_partner = partner; env.partner = partner
            obs, _ = env.reset()
            done, total_r = False, 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, done, _, _ = env.step(action); total_r += r
            rewards.append(total_r)
    return np.mean(rewards)


if __name__ == '__main__':
    print('=== Toy A2C: Hidden vs Revealed ===')
    results = {}

    for name, revealed in [('HIDDEN', False), ('REVEALED', True)]:
        print(f'\n--- A2C {name} ---')
        env = HiddenMatchingEnv(revealed=revealed, n_steps=20)
        model = A2C("MlpPolicy", env, learning_rate=1e-3, gamma=0.99,
                    policy_kwargs=dict(net_arch=[32, 32]),
                    verbose=0, seed=42, device='cpu')
        model.learn(total_timesteps=10000, progress_bar=False)

        traj_B = rollout(model, env, partner=0)
        traj_C = rollout(model, env, partner=1)
        gB, gC = pair_grads(model, traj_B, traj_C)
        k = kappa(gB, gC)
        r = avg_reward(model, env)
        results[name] = {'kappa': k, 'reward': r}
        print(f'  kappa={k:.4f}  reward={r:.3f}')

    print(f'\n{"="*40}')
    for name in ['HIDDEN', 'REVEALED']:
        print(f'{name}: k={results[name]["kappa"]:.4f} r={results[name]["reward"]:.3f}')

    with open(os.path.join(OUTPUT_DIR, 'toy_a2c.json'), 'w') as f:
        json.dump(results, f, indent=2)
