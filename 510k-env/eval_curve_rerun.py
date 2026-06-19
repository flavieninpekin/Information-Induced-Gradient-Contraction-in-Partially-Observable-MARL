"""
重新评估所有中间 checkpoint，使用正确的结算分数胜率定义。
胜者 = 最终结算分数（base reward + 510K 得分）最高的人，不一定先出完。
同时计算相对于 random baseline 的优度。
"""
import os, csv, sys
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from typing import cast

sys.path.insert(0, os.path.dirname(__file__))
from env.env_510k import FiveTenKEnv
from env.scorer import Scorer

N_EVAL = 400     # evaluation episodes per checkpoint
SEED = 42
MODEL_DIR = 'models/curve'
RANDOM_BASELINE_SEED = 999


def mask_fn(env):
    return cast(FiveTenKEnv, env.unwrapped)._get_action_mask()


def make_env():
    env = FiveTenKEnv(mode='single', num_players=4)
    return ActionMasker(env, mask_fn)


def settlement_winner(all_rewards):
    """Return player_id with highest total reward (settlement score)."""
    return max(all_rewards, key=lambda k: all_rewards[k])


def evaluate_checkpoint(model, env, n_episodes=N_EVAL, seed=SEED, agent_id=0):
    """
    Evaluate a model vs 3 random bots.
    Returns dict of metrics.
    """
    n_players = 4
    p0_wins = 0          # P0 settlement reward > all others
    p0_first = 0         # P0 finishes first
    p0_rewards = []
    p0_510k = []
    opponent_rewards = [[] for _ in range(n_players)]  # per opponent
    episode_steps = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        unwrapped = cast(FiveTenKEnv, env.unwrapped)
        game = unwrapped.game
        done = False
        steps = 0
        while not done:
            mask = info['action_mask']
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
            steps += 1

        scorer = Scorer(game)
        all_rewards = scorer.compute_rewards()
        p0_r = all_rewards.get(agent_id, 0.0)
        p0_rewards.append(p0_r)

        # Check if P0 is the settlement winner
        winner = settlement_winner(all_rewards)
        if winner == agent_id:
            p0_wins += 1

        # Check if P0 finished first
        if len(game.finish_order) > 0 and game.finish_order[0] == agent_id:
            p0_first += 1

        # 510K score
        p0_510k.append(float(game.player_510k_scores[agent_id]))

        # Opponent rewards
        for pid in range(n_players):
            if pid != agent_id:
                opponent_rewards[pid].append(all_rewards.get(pid, 0.0))

        episode_steps.append(steps)

    opp_mean = np.mean([np.mean(r) for r in opponent_rewards if r])
    opp_std = np.mean([np.std(r) for r in opponent_rewards if r])

    p0_mean = float(np.mean(p0_rewards))
    p0_std = float(np.std(p0_rewards))

    return {
        'mean_reward': p0_mean,
        'std_reward': p0_std,
        'settlement_win_rate': p0_wins / n_episodes,
        'first_finish_rate': p0_first / n_episodes,
        'mean_510k_score': float(np.mean(p0_510k)),
        'mean_opponent_reward': float(opp_mean),
        'mean_opponent_std': float(opp_std),
        'relative_superiority': float(
            (p0_mean - opp_mean) / (abs(opp_mean) + 1e-8)
        ),
        'mean_episode_steps': float(np.mean(episode_steps)),
        'std_episode_steps': float(np.std(episode_steps)),
    }


def random_baseline(env, n_episodes=400, seed=42):
    """Run 4 random players and compute baseline metrics for one 'agent'."""
    # We'll just use a random policy: from any state, pick random valid action
    p_wins = [0, 0, 0, 0]
    p_first = [0, 0, 0, 0]
    p_rewards = [[] for _ in range(4)]
    p_510k = [[] for _ in range(4)]

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        unwrapped = cast(FiveTenKEnv, env.unwrapped)
        game = unwrapped.game
        done = False
        while not done:
            mask = info['action_mask']
            valid = np.where(mask)[0]
            action = int(np.random.choice(valid))
            obs, reward, done, truncated, info = env.step(action)

        scorer = Scorer(game)
        all_rewards = scorer.compute_rewards()
        winner = settlement_winner(all_rewards)
        for pid in range(4):
            p_rewards[pid].append(all_rewards.get(pid, 0.0))
            p_510k[pid].append(float(game.player_510k_scores[pid]))
            if winner == pid:
                p_wins[pid] += 1
            if len(game.finish_order) > 0 and game.finish_order[0] == pid:
                p_first[pid] += 1

    return {
        'mean_reward': float(np.mean(p_rewards[0])),
        'std_reward': float(np.std(p_rewards[0])),
        'settlement_win_rate': p_wins[0] / n_episodes,
        'first_finish_rate': p_first[0] / n_episodes,
        'mean_510k_score': float(np.mean(p_510k[0])),
        'mean_opponent_reward': float(np.mean(
            [np.mean(p_rewards[i]) for i in range(1, 4)]
        )),
    }


def main():
    env = make_env()

    # === Random baseline ===
    print('Computing random baseline...')
    base = random_baseline(env, n_episodes=800, seed=RANDOM_BASELINE_SEED)
    print(f'  Random P0:  reward={base["mean_reward"]:+5.1f}  '
          f'win_rate={base["settlement_win_rate"]:.1%}  '
          f'first_finish={base["first_finish_rate"]:.1%}  '
          f'510K={base["mean_510k_score"]:.1f}')

    # === Evaluate each checkpoint ===
    checkpoints = [0]  # step 0: will use untrained model
    chk_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.zip')]
    for f in sorted(chk_files):
        step = int(f.replace('model_', '').replace('.zip', ''))
        checkpoints.append(step)
    checkpoints = sorted(set(checkpoints))

    # Step 0: fresh untrained model with maskable policy
    if 0 in checkpoints:
        from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
        model_0 = MaskablePPO(
            MaskableActorCriticPolicy, env, verbose=0, seed=SEED,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        )
        results = {0: evaluate_checkpoint(model_0, env, n_episodes=N_EVAL, seed=SEED)}
        print(f'\n[Step       0]  reward={results[0]["mean_reward"]:+5.1f}  '
              f'win_rate={results[0]["settlement_win_rate"]:.1%}  '
              f'first={results[0]["first_finish_rate"]:.1%}  '
              f'510K={results[0]["mean_510k_score"]:.1f}  '
              f'rel_sup={results[0]["relative_superiority"]:+.2f}')

    # Remaining checkpoints
    for step in checkpoints:
        if step == 0:
            continue
        ckpt = os.path.join(MODEL_DIR, f'model_{step}.zip')
        if not os.path.exists(ckpt):
            continue
        model = MaskablePPO.load(ckpt)
        m = evaluate_checkpoint(model, env, n_episodes=N_EVAL, seed=SEED)
        results[step] = m
        print(f'[Step {step:>7d}]  reward={m["mean_reward"]:+5.1f}  '
              f'win_rate={m["settlement_win_rate"]:.1%}  '
              f'first={m["first_finish_rate"]:.1%}  '
              f'510K={m["mean_510k_score"]:.1f}  '
              f'rel_sup={m["relative_superiority"]:+.2f}')

    env.close()

    # === Save CSV ===
    csv_path = 'training_curve_v2.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'step', 'mean_reward', 'std_reward',
            'settlement_win_rate', 'first_finish_rate',
            'mean_510k_score',
            'mean_opponent_reward', 'relative_superiority',
            'mean_episode_steps', 'std_episode_steps',
            'random_baseline_reward', 'random_baseline_win_rate',
        ])
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
                'random_baseline_reward': base['mean_reward'],
                'random_baseline_win_rate': base['settlement_win_rate'],
            })
    print(f'\nSaved: {csv_path}')

    # === Print summary ===
    print(f'\nRandom baseline: reward={base["mean_reward"]:+5.1f}, '
          f'win_rate={base["settlement_win_rate"]:.1%}, '
          f'first_finish={base["first_finish_rate"]:.1%}')
    print(f'\n{"Step":>8s}  {"Reward":>7s}  {"WinRate":>7s}  {"1stFin":>7s}  '
          f'{"510K":>5s}  {"OppRew":>7s}  {"RelSup":>6s}  {"Steps":>5s}')
    for step in sorted(results):
        r = results[step]
        print(f'{step:>8d}  {r["mean_reward"]:+6.1f}  '
              f'{r["settlement_win_rate"]:.1%}  {r["first_finish_rate"]:.1%}  '
              f'{r["mean_510k_score"]:5.1f}  {r["mean_opponent_reward"]:+6.1f}  '
              f'{r["relative_superiority"]:+5.2f}  '
              f'{r["mean_episode_steps"]:5.0f}')


if __name__ == '__main__':
    main()
