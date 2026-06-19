"""
训练 MaskablePPO 2^20 步，在对数坐标下绘制收敛曲线。
P0=agent, P1/P2/P3=random bot
追踪：结算分数、先出完率、平均牌局步数

使用分段训练避免 SB3 callback bug，每段结束保存中间模型。
"""
import os, csv, time, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import cast

import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from env.env_510k import FiveTenKEnv

N_EVAL_EPISODES = 100
LOG_STEPS = [2**k for k in range(5, 21)]   # 32 ~ 1,048,576
TOTAL_TIMESTEPS = 2**20
SEED = 42
MODEL_DIR = 'models/curve'


def mask_fn(env):
    return cast(FiveTenKEnv, env.unwrapped)._get_action_mask()


def make_env():
    env = FiveTenKEnv(mode='single', num_players=4)
    return ActionMasker(env, mask_fn)


def true_first_finish(game, agent_id=0):
    """Return True if agent finishes first (zero-sum metric)."""
    return len(game.finish_order) > 0 and game.finish_order[0] == agent_id


def evaluate(model, env, n_episodes=N_EVAL_EPISODES, seed=SEED):
    firsts = 0
    rewards = []
    steps_list = []
    final_510k = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        game = cast(FiveTenKEnv, env.unwrapped).game
        done = False
        steps = 0
        while not done:
            mask = info['action_mask']
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
            steps += 1
        rewards.append(reward)
        steps_list.append(steps)
        if true_first_finish(game):
            firsts += 1
        score = float(game.player_510k_scores[0])
        final_510k.append(score)

    return {
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'first_finish_rate': firsts / n_episodes,
        'mean_510k_score': float(np.mean(final_510k)),
        'mean_episode_steps': float(np.mean(steps_list)),
        'std_episode_steps': float(np.std(steps_list)),
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    eval_env = make_env()
    train_env = make_env()

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=0,
        seed=SEED,
        tensorboard_log='logs',
    )

    results = {}  # step -> metrics

    # Step 0: initial random policy
    print('Evaluating initial (random) policy...')
    m = evaluate(model, eval_env, N_EVAL_EPISODES)
    results[0] = m
    print(f'[Step       0]  reward={m["mean_reward"]:+5.1f} +/- {m["std_reward"]:.0f}  '
          f'first_finish={m["first_finish_rate"]:.1%}  510K={m["mean_510k_score"]:.1f}  '
          f'steps={m["mean_episode_steps"]:.0f}')

    # Train in segments between log-spaced evaluation points
    eval_points = sorted(LOG_STEPS)
    prev = 0
    for target in eval_points:
        delta = target - prev
        if delta > 0:
            model.learn(total_timesteps=delta, reset_num_timesteps=False)
        # Save intermediate checkpoint
        ckpt_path = os.path.join(MODEL_DIR, f'model_{target}.zip')
        model.save(ckpt_path)
        # Evaluate
        m = evaluate(model, eval_env, N_EVAL_EPISODES)
        results[target] = m
        print(f'[Step {target:>7d}]  reward={m["mean_reward"]:+5.1f} +/- {m["std_reward"]:.0f}  '
              f'first_finish={m["first_finish_rate"]:.1%}  510K={m["mean_510k_score"]:.1f}  '
              f'steps={m["mean_episode_steps"]:.0f}')
        prev = target

    # Final: train remaining to TOTAL_TIMESTEPS if needed
    if TOTAL_TIMESTEPS > prev:
        delta = TOTAL_TIMESTEPS - prev
        model.learn(total_timesteps=delta, reset_num_timesteps=False)
        model.save(os.path.join(MODEL_DIR, 'model_final.zip'))
        m = evaluate(model, eval_env, N_EVAL_EPISODES)
        results[TOTAL_TIMESTEPS] = m
        print(f'[Step {TOTAL_TIMESTEPS:>7d}]  reward={m["mean_reward"]:+5.1f} +/- {m["std_reward"]:.0f}  '
              f'first_finish={m["first_finish_rate"]:.1%}  510K={m["mean_510k_score"]:.1f}  '
              f'steps={m["mean_episode_steps"]:.0f}')

    train_env.close()
    eval_env.close()

    # Save CSV
    csv_path = 'training_curve.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'step', 'mean_reward', 'std_reward', 'first_finish_rate',
            'mean_510k_score', 'mean_episode_steps', 'std_episode_steps',
        ])
        w.writeheader()
        for step in sorted(results):
            r = results[step]
            w.writerow({
                'step': step,
                'mean_reward': r['mean_reward'],
                'std_reward': r['std_reward'],
                'first_finish_rate': r['first_finish_rate'],
                'mean_510k_score': r['mean_510k_score'],
                'mean_episode_steps': r['mean_episode_steps'],
                'std_episode_steps': r['std_episode_steps'],
            })
    print(f'\nSaved: {csv_path}')

    # Plot
    steps = sorted(results)
    rewards = [results[s]['mean_reward'] for s in steps]
    reward_stds = [results[s]['std_reward'] for s in steps]
    first_rates = [results[s]['first_finish_rate'] for s in steps]
    scores_510k = [results[s]['mean_510k_score'] for s in steps]
    mean_steps = [results[s]['mean_episode_steps'] for s in steps]
    step_stds = [results[s]['std_episode_steps'] for s in steps]

    fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)

    ax = axes[0]
    ax.errorbar(steps, rewards, yerr=reward_stds, fmt='o-', capsize=3, color='C0')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_ylabel('Mean Settlement Reward')
    ax.set_title('Training Convergence (Single Mode, P0 vs 3 Random Bots)')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogx(steps, first_rates, 'o-', color='C1')
    ax.axhline(0.25, color='gray', linestyle='--', linewidth=0.5, label='Random (1/4)')
    ax.set_ylabel('First-Finish Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.semilogx(steps, scores_510k, 'o-', color='C3')
    ax.set_ylabel('Mean 510K Score')
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.errorbar(steps, mean_steps, yerr=step_stds, fmt='o-', capsize=3, color='C2')
    ax.set_xlabel('Training Steps (log scale)')
    ax.set_ylabel('Mean Episode Steps')
    ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xscale('log')
        ax.set_xlim(min(steps) * 0.8, max(steps) * 1.2)

    plt.tight_layout()
    plt.savefig('training_curve.png', dpi=150)
    print(f'Saved: training_curve.png')

    # Summary
    print('\n===== Summary =====')
    for step in steps:
        r = results[step]
        print(f'Step {step:>7d}:  reward={r["mean_reward"]:+5.1f}  first_finish={r["first_finish_rate"]:.1%}  '
              f'510K={r["mean_510k_score"]:.1f}  steps={r["mean_episode_steps"]:.0f}')


if __name__ == '__main__':
    main()
