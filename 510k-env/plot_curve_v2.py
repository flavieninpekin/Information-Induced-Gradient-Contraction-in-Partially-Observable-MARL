"""
从 training_curve_v2.csv 读取数据，绘制收敛曲线（结算胜率为核心指标）。
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


CSV_PATH = 'training_curve_v2.csv'


def load_data(path=CSV_PATH):
    steps, rewards, stds, win_rates, first_rates, scores_510k = [], [], [], [], [], []
    opp_rewards, rel_sups, mean_steps, step_stds = [], [], [], []
    base_reward, base_win_rate = None, None

    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            steps.append(int(row['step']))
            rewards.append(float(row['mean_reward']))
            stds.append(float(row['std_reward']))
            win_rates.append(float(row['settlement_win_rate']))
            first_rates.append(float(row['first_finish_rate']))
            scores_510k.append(float(row['mean_510k_score']))
            opp_rewards.append(float(row['mean_opponent_reward']))
            rel_sups.append(float(row['relative_superiority']))
            mean_steps.append(float(row['mean_episode_steps']))
            step_stds.append(float(row['std_episode_steps']))
            if base_reward is None:
                base_reward = float(row['random_baseline_reward'])
                base_win_rate = float(row['random_baseline_win_rate'])

    return (steps, rewards, stds, win_rates, first_rates, scores_510k,
            opp_rewards, rel_sups, mean_steps, step_stds, base_reward, base_win_rate)


def plot():
    (steps, rewards, stds, win_rates, first_rates, scores_510k,
     opp_rewards, rel_sups, mean_steps, step_stds, base_reward, base_win_rate) = load_data()

    fig, axes = plt.subplots(5, 1, figsize=(11, 16), sharex=True)

    colors = ['C0', 'C1', 'C3', 'C2', 'C4']

    # === 1: Agent reward + opponent reward ===
    ax = axes[0]
    ax.errorbar(steps, rewards, yerr=stds, fmt='o-', capsize=3, color=colors[0], label='Agent (P0)')
    ax.semilogx(steps, opp_rewards, 's--', color=colors[1], label='Avg Opponent (random)')
    ax.axhline(base_reward, color='gray', linestyle=':', linewidth=0.8, label=f'Random baseline ({base_reward:+.1f})')
    ax.set_ylabel('Settlement Reward')
    ax.set_title('Training Convergence (MaskablePPO, Single Mode, P0 vs 3 Random Bots)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # === 2: Win rate (settlement) + first finish ===
    ax = axes[1]
    ax.semilogx(steps, win_rates, 'o-', color=colors[2], label='Settlement Win Rate')
    ax.semilogx(steps, first_rates, 'x--', color=colors[3], label='First-Finish Rate')
    ax.axhline(base_win_rate, color='gray', linestyle=':', linewidth=0.8, label=f'Random baseline ({base_win_rate:.1%})')
    ax.axhline(0.25, color='gray', linestyle='--', linewidth=0.5)
    ax.set_ylabel('Rate')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # === 3: Relative superiority ===
    ax = axes[2]
    ax.semilogx(steps, rel_sups, 'o-', color=colors[4])
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_ylabel('Relative Superiority\n(agent - opponent)/|opponent|')
    ax.grid(True, alpha=0.3)

    # === 4: 510K score ===
    ax = axes[3]
    ax.semilogx(steps, scores_510k, 'o-', color='C5')
    ax.set_ylabel('Mean 510K Score')
    ax.grid(True, alpha=0.3)

    # === 5: Episode steps ===
    ax = axes[4]
    ax.errorbar(steps, mean_steps, yerr=step_stds, fmt='o-', capsize=3, color='C2')
    ax.set_xlabel('Training Steps (log scale)')
    ax.set_ylabel('Mean Episode Steps')
    ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xscale('log')
        ax.set_xlim(min(steps) * 0.8, max(steps) * 1.2)

    plt.tight_layout()
    out = 'training_curve_v2.png'
    plt.savefig(out, dpi=150)
    print(f'Saved: {out}')


if __name__ == '__main__':
    plot()
