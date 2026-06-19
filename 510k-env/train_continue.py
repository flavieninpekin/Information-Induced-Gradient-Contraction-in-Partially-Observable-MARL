"""
从 2^20 步继续训练到 2^22 步，每 2^16 步评估一次。
加载之前的 final 模型，分段训练 + 评估。
线性坐标绘图。
"""
import os, csv, time, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import cast

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

sys.path.insert(0, os.path.dirname(__file__))
from env.env_510k import FiveTenKEnv
from env.scorer import Scorer

# === Config ===
END_STEP = 2**22             # 4,194,304
EVAL_INTERVAL = 2**16        # 65,536
N_EVAL = 200                 # episodes per eval
SEED = 42

MODEL_PATH = 'models/curve/model_1048576.zip'
MODEL_SAVE_DIR = 'models/curve_cont'
CSV_CONT_PATH = 'training_curve_cont.csv'
PLOT_PATH = 'training_curve_cont.png'


def mask_fn(env):
    return cast(FiveTenKEnv, env.unwrapped)._get_action_mask()


def make_env():
    env = FiveTenKEnv(mode='single', num_players=4)
    return ActionMasker(env, mask_fn)


def settlement_winner(all_rewards):
    return max(all_rewards, key=lambda k: all_rewards[k])


def evaluate(model, env, n_episodes=N_EVAL, seed=SEED, agent_id=0):
    wins, firsts = 0, 0
    p0_rewards, p0_510k, opp_rewards_list, steps_list = [], [], [], []
    n_players = 4

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        unwrapped = cast(FiveTenKEnv, env.unwrapped)
        game = unwrapped.game
        done, steps = False, 0
        while not done:
            mask = info['action_mask']
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
            steps += 1

        scorer = Scorer(game)
        all_rewards = scorer.compute_rewards()
        p0_r = all_rewards.get(agent_id, 0.0)
        p0_rewards.append(p0_r)
        p0_510k.append(float(game.player_510k_scores[agent_id]))

        if settlement_winner(all_rewards) == agent_id:
            wins += 1
        if len(game.finish_order) > 0 and game.finish_order[0] == agent_id:
            firsts += 1

        opp_rews = [all_rewards[pid] for pid in range(n_players) if pid != agent_id]
        opp_rewards_list.extend(opp_rews)
        steps_list.append(steps)

    p0_mean = float(np.mean(p0_rewards))
    opp_mean = float(np.mean(opp_rewards_list))
    
    return {
        'step': 0,  # filled later
        'mean_reward': p0_mean,
        'std_reward': float(np.std(p0_rewards)),
        'settlement_win_rate': wins / n_episodes,
        'first_finish_rate': firsts / n_episodes,
        'mean_510k_score': float(np.mean(p0_510k)),
        'mean_opponent_reward': round(opp_mean, 2),
        'relative_superiority': round(
            (p0_mean - opp_mean) / (abs(opp_mean) + 1e-8), 4
        ),
        'mean_episode_steps': float(np.mean(steps_list)),
        'std_episode_steps': float(np.std(steps_list)),
    }


def main():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    eval_env = make_env()
    train_env = make_env()

    print(f'Loading model from {MODEL_PATH}...')
    model = MaskablePPO.load(MODEL_PATH, env=train_env)
    actual_start = model.num_timesteps
    print(f'  Actual num_timesteps: {actual_start}')

    results = {}

    # Starting point: evaluate at actual_start
    m = evaluate(model, eval_env, N_EVAL, SEED)
    m['step'] = actual_start
    results[actual_start] = m
    print(f'[Step {actual_start:>7d}]  reward={m["mean_reward"]:+5.1f}  '
          f'win_rate={m["settlement_win_rate"]:.1%}  '
          f'first={m["first_finish_rate"]:.1%}')

    # Train in EVAL_INTERVAL chunks until END_STEP, using actual step counter
    while model.num_timesteps < END_STEP:
        t0 = time.time()
        model.learn(total_timesteps=EVAL_INTERVAL, reset_num_timesteps=False)
        elapsed = time.time() - t0
        actual_step = model.num_timesteps

        # Save checkpoint (named by actual step)
        ckpt_name = f'model_{actual_step}.zip'
        model.save(os.path.join(MODEL_SAVE_DIR, ckpt_name))

        # Evaluate
        m = evaluate(model, eval_env, N_EVAL, SEED)
        m['step'] = actual_step
        results[actual_step] = m
        print(f'[Step {actual_step:>7d}]  reward={m["mean_reward"]:+5.1f}  '
              f'win_rate={m["settlement_win_rate"]:.1%}  '
              f'first={m["first_finish_rate"]:.1%}  '
              f'510K={m["mean_510k_score"]:.1f}  '
              f'rel_sup={m["relative_superiority"]:+4.2f}  '
              f'steps={m["mean_episode_steps"]:.0f}  [{elapsed:.0f}s]')

    train_env.close()
    eval_env.close()

    model.save(os.path.join(MODEL_SAVE_DIR, 'model_final_4M.zip'))
    print(f'\nFinal model saved. Total timesteps: {model.num_timesteps}')

    # === Save CSV ===
    fieldnames = [
        'step', 'mean_reward', 'std_reward',
        'settlement_win_rate', 'first_finish_rate',
        'mean_510k_score', 'mean_opponent_reward',
        'relative_superiority', 'mean_episode_steps', 'std_episode_steps',
    ]
    with open(CSV_CONT_PATH, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for step in sorted(results):
            r = results[step]
            w.writerow({
                'step': step,
                'mean_reward': r['mean_reward'],
                'std_reward': r['std_reward'],
                'settlement_win_rate': r['settlement_win_rate'],
                'first_finish_rate': r['first_finish_rate'],
                'mean_510k_score': r['mean_510k_score'],
                'mean_opponent_reward': r['mean_opponent_reward'],
                'relative_superiority': r['relative_superiority'],
                'mean_episode_steps': r['mean_episode_steps'],
                'std_episode_steps': r['std_episode_steps'],
            })
    print(f'Saved: {CSV_CONT_PATH}')

    # === Plot (linear x-axis) ===
    steps = sorted(results)
    rewards = [results[s]['mean_reward'] for s in steps]
    reward_stds = [results[s]['std_reward'] for s in steps]
    win_rates = [results[s]['settlement_win_rate'] for s in steps]
    first_rates = [results[s]['first_finish_rate'] for s in steps]
    rel_sups = [results[s]['relative_superiority'] for s in steps]
    scores_510k = [results[s]['mean_510k_score'] for s in steps]
    opp_rews = [results[s]['mean_opponent_reward'] for s in steps]
    mean_steps = [results[s]['mean_episode_steps'] for s in steps]
    step_stds = [results[s]['std_episode_steps'] for s in steps]

    fig, axes = plt.subplots(5, 1, figsize=(12, 16), sharex=True)

    # 1: Rewards
    ax = axes[0]
    ax.errorbar(steps, rewards, yerr=reward_stds, fmt='o-', capsize=3, color='C0', label='Agent')
    ax.plot(steps, opp_rews, 's--', color='C1', label='Avg Opponent')
    ax.axhline(22.7, color='gray', linestyle=':', label='Random baseline (+22.7)')
    ax.set_ylabel('Settlement Reward')
    ax.set_title('Training 2^20 → 2^22 (65K eval interval, linear scale)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2: Win rates
    ax = axes[1]
    ax.plot(steps, win_rates, 'o-', color='C3', label='Settlement Win Rate')
    ax.plot(steps, first_rates, 'x--', color='C4', label='First-Finish Rate')
    ax.axhline(0.259, color='gray', linestyle=':', label='Random baseline (25.9%)')
    ax.set_ylabel('Rate')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # 3: Relative superiority
    ax = axes[2]
    ax.plot(steps, rel_sups, 'o-', color='C5')
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_ylabel('Relative Superiority')
    ax.grid(True, alpha=0.3)

    # 4: 510K
    ax = axes[3]
    ax.plot(steps, scores_510k, 'o-', color='C6')
    ax.set_ylabel('Mean 510K Score')
    ax.grid(True, alpha=0.3)

    # 5: Episode steps
    ax = axes[4]
    ax.errorbar(steps, mean_steps, yerr=step_stds, fmt='o-', capsize=3, color='C2')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Mean Steps/Episode')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f'Saved: {PLOT_PATH}')

    # Print summary table
    print(f'\n{"="*60}')
    print(f'{"Step":>8s}  {"Reward":>7s}  {"WinRate":>7s}  {"1stFin":>7s}  {"510K":>5s}  {"OppRew":>7s}  {"RelSup":>6s}  {"Steps":>5s}')
    print(f'{"="*60}')
    for step in steps:
        r = results[step]
        print(f'{step:>8d}  {r["mean_reward"]:+6.1f}  '
              f'{r["settlement_win_rate"]:.1%}  {r["first_finish_rate"]:.1%}  '
              f'{r["mean_510k_score"]:5.1f}  {r["mean_opponent_reward"]:+6.1f}  '
              f'{r["relative_superiority"]:+5.2f}  '
              f'{r["mean_episode_steps"]:5.0f}')


if __name__ == '__main__':
    main()
