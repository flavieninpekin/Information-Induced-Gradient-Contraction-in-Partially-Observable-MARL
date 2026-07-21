"""
Toy DQN experiment: DQN on hidden vs revealed matching.
Measures gradient contraction (kappa) and behavioral divergence.
Mirrors toy_experiment.py with DQN instead of PPO.
"""
import os, json, numpy as np, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from env.toy_env import HiddenMatchingEnv

OUTPUT_DIR = 'toy_results'


def rollout(model, env, partner, n_eps=30):
    """Run episodes with fixed partner using deterministic policy."""
    env._forced_partner = partner
    trajectories = []
    for _ in range(n_eps):
        env.partner = partner
        obs, _ = env.reset()
        done = False
        transitions = []
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                q_values = model.q_net(obs_t)
            action = q_values.argmax(dim=1).item()
            next_obs, reward, done, trunc, _ = env.step(action)
            transitions.append((obs, action, reward, next_obs, done))
            obs = next_obs
        trajectories.append(transitions)
    return trajectories


def compute_dqn_gradients(model, traj):
    """Compute average TD(0) loss gradient for a partner's trajectories."""
    all_obs, all_next_obs, all_actions, all_rewards, all_dones = [], [], [], [], []
    for transitions in traj:
        for obs, action, reward, next_obs, done in transitions:
            all_obs.append(obs)
            all_next_obs.append(next_obs)
            all_actions.append(action)
            all_rewards.append(reward)
            all_dones.append(float(done))

    if not all_obs:
        return torch.zeros(1)

    batch_obs = torch.FloatTensor(np.array(all_obs))
    batch_next_obs = torch.FloatTensor(np.array(all_next_obs))
    batch_actions = torch.tensor(all_actions)
    batch_rewards = torch.tensor(all_rewards, dtype=torch.float32)
    batch_dones = torch.tensor(all_dones, dtype=torch.float32)

    with torch.no_grad():
        next_q = model.q_net_target(batch_next_obs)
        max_next_q = next_q.max(dim=1)[0]
        targets = batch_rewards + (1.0 - batch_dones) * model.gamma * max_next_q

    q_values = model.q_net(batch_obs)
    q_pred = q_values[range(len(batch_actions)), batch_actions]
    loss = ((q_pred - targets) ** 2).mean()

    model.q_net.zero_grad()
    loss.backward()

    grad_vecs = [p.grad.detach().clone().flatten() for p in model.q_net.parameters() if p.grad is not None]
    return torch.cat(grad_vecs) if grad_vecs else torch.zeros(1)


def compute_kappa(grad_B, grad_C):
    """kappa = ||(g_B+g_C)/2||^2 / (||g_B||^2+||g_C||^2)/2"""
    avg = (grad_B + grad_C) / 2.0
    num = torch.norm(avg) ** 2
    denom = (torch.norm(grad_B) ** 2 + torch.norm(grad_C) ** 2) / 2.0
    return (num / max(denom, 1e-10)).item()


def action_probs_dqn(model, env, partner, n_eps=30):
    """Probability of action 0 under deterministic greedy policy."""
    env._forced_partner = partner
    actions = []
    for _ in range(n_eps):
        env.partner = partner
        obs, _ = env.reset()
        done = False
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                q_values = model.q_net(obs_t)
            action = q_values.argmax(dim=1).item()
            actions.append(action)
            obs, _, done, _, _ = env.step(action)
    return np.mean(actions) if actions else 0.5


def avg_reward_dqn(model, env, n_both=30):
    """Average reward across both partners with deterministic policy."""
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
                    q_values = model.q_net(obs_t)
                action = q_values.argmax(dim=1).item()
                obs, r, done, _, _ = env.step(action)
                total_r += r
            rewards.append(total_r)
    return np.mean(rewards)


def _run_experiment():
    print('=== Toy DQN: Hidden vs Revealed Matching ===\n')
    results = {}
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for mode_name, revealed in [('HIDDEN', False), ('REVEALED', True)]:
        print(f'\n--- DQN {mode_name} ---')
        env = HiddenMatchingEnv(revealed=revealed, n_steps=20)

        model = DQN(
            "MlpPolicy", env,
            learning_rate=1e-3, buffer_size=2000, learning_starts=200,
            batch_size=32, tau=0.005, gamma=0.99,
            train_freq=4, gradient_steps=1,
            target_update_interval=100,
            exploration_fraction=0.3,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.02,
            policy_kwargs=dict(net_arch=[32, 32]),
            verbose=0, seed=42, device='cpu',
        )

        model.learn(total_timesteps=10000, progress_bar=False)
        model.q_net.eval()

        traj_B = rollout(model, env, partner=0, n_eps=20)
        traj_C = rollout(model, env, partner=1, n_eps=20)
        g_B = compute_dqn_gradients(model, traj_B)
        g_C = compute_dqn_gradients(model, traj_C)
        kg = compute_kappa(g_B, g_C)

        p0B = action_probs_dqn(model, env, partner=0)
        p0C = action_probs_dqn(model, env, partner=1)
        kb = abs(p0B - p0C)
        rew = avg_reward_dqn(model, env)

        print(f'  k_g={kg:.4f}  k_b={kb:.4f}  '
              f'p0B={p0B:.3f} p0C={p0C:.3f}  '
              f'||g_B||={torch.norm(g_B):.2f} ||g_C||={torch.norm(g_C):.2f}  '
              f'R={rew:+.3f}')

        final_data = {
            'kappa_grad': kg, 'kappa_beh': kb,
            'p0B': p0B, 'p0C': p0C, 'g_norm_B': torch.norm(g_B).item(),
            'g_norm_C': torch.norm(g_C).item(), 'reward': rew,
        }
        results[mode_name] = {
            'final_kappa': kg, 'final_reward': rew, 'final_data': final_data,
        }
        print(f'  Final k_g={kg:.4f} R={rew:+.3f}')

        with open(os.path.join(OUTPUT_DIR, f'DQN_{mode_name}.json'), 'w') as f:
            json.dump(final_data, f, indent=2)

    print('\n' + '=' * 60)
    print('TOY DQN EXPERIMENT SUMMARY')
    print('=' * 60)
    for name in ['HIDDEN', 'REVEALED']:
        s = results[name]
        print(f'{name:>8}: k={s["final_kappa"]:.4f} R={s["final_reward"]:+.3f}')

    # PPO vs DQN comparison
    ppo_files = {'HIDDEN': 'HIDDEN.json', 'REVEALED': 'REVEALED.json'}
    dqn_files = {'HIDDEN': 'DQN_HIDDEN.json', 'REVEALED': 'DQN_REVEALED.json'}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for col, name in enumerate(['HIDDEN', 'REVEALED']):
        ppo_path = os.path.join(OUTPUT_DIR, ppo_files[name])
        dqn_path = os.path.join(OUTPUT_DIR, dqn_files[name])

        ppo_k = json.load(open(ppo_path))[-1]['kappa_grad'] if os.path.exists(ppo_path) else float('nan')
        ppo_r = json.load(open(ppo_path))[-1]['reward'] if os.path.exists(ppo_path) else float('nan')
        dqn_k = json.load(open(dqn_path))['kappa_grad'] if os.path.exists(dqn_path) else float('nan')
        dqn_r = json.load(open(dqn_path))['reward'] if os.path.exists(dqn_path) else float('nan')

        w = 0.35
        axes[0].bar(col - w/2, ppo_k, w, label='PPO' if col == 0 else '', color='C0')
        axes[0].bar(col + w/2, dqn_k, w, label='DQN' if col == 0 else '', color='C1')
        axes[1].bar(col - w/2, ppo_r, w, label='PPO' if col == 0 else '', color='C0')
        axes[1].bar(col + w/2, dqn_r, w, label='DQN' if col == 0 else '', color='C1')

    axes[0].set_title('Gradient Contraction (kappa_g)')
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['HIDDEN', 'REVEALED'])
    axes[0].legend()
    axes[1].set_title('Average Reward')
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['HIDDEN', 'REVEALED'])
    axes[1].legend()

    fig.suptitle('PPO vs DQN: Gradient Contraction on Toy Matching', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'toy_ppo_vs_dqn.pdf'), dpi=200)
    fig.savefig(os.path.join(OUTPUT_DIR, 'toy_ppo_vs_dqn.png'), dpi=200)
    print('\nSaved to toy_results/')


if __name__ == '__main__':
    _run_experiment()
