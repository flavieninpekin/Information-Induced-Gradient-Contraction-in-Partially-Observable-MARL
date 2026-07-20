"""
Toy experiment: PPO on hidden vs revealed matching.
Demonstrates information-induced gradient contraction with proper κ_gradient.
"""
import os, json, numpy as np, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from env.toy_env import HiddenMatchingEnv

OUTPUT_DIR = 'toy_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = 'cpu'


def rollout(model, env, partner, n_eps=30):
    """Run episodes with fixed partner, return (observations, actions, returns)."""
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
    """Compute policy gradient for each partner's trajectories."""
    grads = []
    for traj in [traj_B, traj_C]:
        total_grad = None
        n_samples = 0
        for obs_list, act_list, rew_list in traj:
            # Simple REINFORCE: ∇log π(a|s) * return
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
                    if total_grad is None:
                        total_grad = g_flat
                    else:
                        total_grad += g_flat
                    n_samples += 1
        
        total_grad /= max(n_samples, 1)
        grads.append(total_grad)
    
    return grads[0], grads[1]


def compute_kappa(grad_B, grad_C):
    """κ = ||(g_B+g_C)/2||² / (||g_B||²+||g_C||²)/2"""
    avg = (grad_B + grad_C) / 2.0
    num = torch.norm(avg) ** 2
    denom = (torch.norm(grad_B) ** 2 + torch.norm(grad_C) ** 2) / 2.0
    return (num / max(denom, 1e-10)).item()


def action_probs(model, env, partner, n_eps=30):
    """Probability of action 0 under a specific partner."""
    env._forced_partner = partner
    probs = []
    for _ in range(n_eps):
        env.partner = partner
        obs, _ = env.reset()
        done = False
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                distribution = model.policy.get_distribution(obs_t)
            p = distribution.distribution.probs.cpu().numpy()[0]
            probs.append(float(p[0]))
            action = distribution.get_actions().item()
            obs, _, done, _, _ = env.step(action)
    return np.mean(probs)


def avg_reward(model, env, n_both=30):
    """Average reward across both partners."""
    rewards = []
    for partner in [0, 1]:
        for _ in range(n_both):
            env._forced_partner = partner
            env.partner = partner
            obs, _ = env.reset()
            done, total_r = False, 0
            while not done:
                obs_t = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    distribution = model.policy.get_distribution(obs_t)
                action = distribution.get_actions().item()
                obs, r, done, _, _ = env.step(action)
                total_r += r
            rewards.append(total_r)
    return np.mean(rewards)


# ============================================================
print('=== Toy: Hidden vs Revealed Matching with Proper κ ===\n')
results = {}

for mode_name, revealed in [('HIDDEN', False), ('REVEALED', True)]:
    print(f'\n--- {mode_name} ---')
    env = HiddenMatchingEnv(revealed=revealed, n_steps=20)

    model = PPO(
        ActorCriticPolicy, env,
        learning_rate=1e-3, n_steps=256, batch_size=32, n_epochs=5,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
        policy_kwargs=dict(net_arch=dict(pi=[32, 32], vf=[32, 32])),
        verbose=0, seed=42, device='cpu',
    )

    ckpt_data = []
    total = 0
    while total < 10000:
        model.learn(total_timesteps=1000, progress_bar=False)
        total += 1000

        # Measure proper κ via gradient decomposition
        traj_B = rollout(model, env, partner=0, n_eps=20)
        traj_C = rollout(model, env, partner=1, n_eps=20)
        g_B, g_C = compute_pair_gradients(model, traj_B, traj_C)
        kg = compute_kappa(g_B, g_C)

        # Behavioral κ
        p0B = action_probs(model, env, partner=0)
        p0C = action_probs(model, env, partner=1)
        kb = abs(p0B - p0C)

        # Average reward
        rew = avg_reward(model, env)

        ckpt_data.append({
            'steps': total, 'kappa_grad': kg, 'kappa_beh': kb,
            'p0B': p0B, 'p0C': p0C, 'g_norm_B': torch.norm(g_B).item(),
            'g_norm_C': torch.norm(g_C).item(), 'reward': rew,
        })

        print(f'  {total:>5}: κ_g={kg:.4f}  κ_b={kb:.4f}  '
              f'p0B={p0B:.3f} p0C={p0C:.3f}  '
              f'||g_B||={torch.norm(g_B):.2f} ||g_C||={torch.norm(g_C):.2f}  '
              f'R={rew:+.3f}')

    # Path integral from action prob trajectory
    p0B_traj = np.array([d['p0B'] for d in ckpt_data])
    p0C_traj = np.array([d['p0C'] for d in ckpt_data])
    feats = np.column_stack([p0B_traj, p0C_traj])
    path_len = np.sum(np.linalg.norm(np.diff(feats, axis=0), axis=1))
    endpt = np.linalg.norm(feats[-1] - feats[0])
    curv = path_len / max(endpt, 1e-10)

    final = ckpt_data[-1]
    results[mode_name] = {
        'path_len': path_len, 'endpt': endpt, 'curv': curv,
        'final_kappa': final['kappa_grad'], 'final_reward': final['reward'],
        'data': ckpt_data,
    }
    print(f'  PATH={path_len:.4f} ENDPT={endpt:.4f} CURV={curv:.1f}x')
    print(f'  Final κ_g={final["kappa_grad"]:.4f} R={final["reward"]:+.3f}')

    with open(os.path.join(OUTPUT_DIR, f'{mode_name}.json'), 'w') as f:
        json.dump(ckpt_data, f, indent=2)

# ============================================================
# Summary
print('\n' + '=' * 60)
print('TOY EXPERIMENT SUMMARY')
print('=' * 60)
for name in ['HIDDEN', 'REVEALED']:
    s = results[name]
    print(f'{name:>8}: PATH={s["path_len"]:.4f} κ={s["final_kappa"]:.4f} R={s["final_reward"]:+.3f}')

# ============================================================
# Plots
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax_idx, metric in enumerate(['kappa_grad', 'kappa_beh', 'reward']):
    ax = axes[ax_idx]
    for name in ['HIDDEN', 'REVEALED']:
        data = results[name]['data']
        ax.plot([d['steps'] for d in data], [d[metric] for d in data],
                'o-', markersize=3, lw=1.5, label=name)
    ax.set_xlabel('Steps'); ax.set_ylabel(metric.replace('_',' ').title()); ax.legend()
    ax.set_title(metric.replace('_',' ').title())

fig.suptitle('Gradient Contraction in Hidden Matching: κ_grad reveals the signal', fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'toy.pdf'), dpi=200)
fig.savefig(os.path.join(OUTPUT_DIR, 'toy.png'), dpi=200)

# Also save the three-panel summary separately
print('\nSaved to toy_results/')
