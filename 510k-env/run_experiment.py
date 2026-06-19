"""
========================================================
510K RL 收敛实验：多种子训练 + 评测 + Head-to-head
========================================================
训练 3 个种子从 0 → 2^22 步，在关键 checkpoint 评估，
最后早期 vs 晚期模型实战对决。
========================================================
"""
import os, csv, time, sys, random, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import cast

import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

sys.path.insert(0, os.path.dirname(__file__))
from env.env_510k import FiveTenKEnv
from env.scorer import Scorer

# ========== CONFIG ==========
SEEDS = [42, 123, 456]
EVAL_N_EPISODES = 200
EVAL_BASELINE_EPISODES = 800
HEAD2HEAD_N_GAMES = 400
TOTAL_STEPS = 2**22  # 4,194,304

# Phase A: log-spaced eval points 2^5 .. 2^20
PHASE_A = [2**k for k in range(5, 21)]
# Phase B: every 2^16 from 2^20+2^16 to 2^22
PHASE_B = list(range(2**20 + 2**16, 2**22 + 1, 2**16))
EVAL_POINTS = sorted(set(PHASE_A + PHASE_B))

# Head-to-head key checkpoints
HH_CHECKPOINTS = [2**14, 2**17, 2**20, 2**21, 2**22]
HH_LABELS = {2**14: '16K', 2**17: '131K', 2**20: '1M', 2**21: '2M', 2**22: '4M'}

BASE_DIR = 'experiment_results'
TRAIN_SUBDIR = os.path.join(BASE_DIR, 'training')
HH_SUBDIR = os.path.join(BASE_DIR, 'head2head')
PLOTS_SUBDIR = os.path.join(BASE_DIR, 'plots')

# ========== HELPERS ==========


class SeededResetWrapper(gym.Wrapper):
    """Seeds every env.reset() during training."""
    def __init__(self, env, base_seed):
        super().__init__(env)
        self.base_seed = base_seed
        self.count = 0

    def reset(self, **kwargs):
        kwargs['seed'] = self.base_seed + self.count
        self.count += 1
        return self.env.reset(**kwargs)


def mask_fn(env):
    return cast(FiveTenKEnv, env.unwrapped)._get_action_mask()


def make_env():
    return ActionMasker(FiveTenKEnv(mode='single', num_players=4), mask_fn)


def settlement_winner(all_rewards):
    return max(all_rewards, key=lambda k: all_rewards[k])


def evaluate(model, env, n_episodes=EVAL_N_EPISODES, seed=42):
    wins = firsts = 0
    p0_rewards, p0_510k, all_opp, steps_list = [], [], [], []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        game = cast(FiveTenKEnv, env.unwrapped).game
        done = s = 0
        while not done:
            mask = info['action_mask']
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
            s += 1
        scorer = Scorer(game)
        ar = scorer.compute_rewards()
        p0_r = ar.get(0, 0.0)
        p0_rewards.append(p0_r)
        p0_510k.append(float(game.player_510k_scores[0]))
        if settlement_winner(ar) == 0:
            wins += 1
        if game.finish_order and game.finish_order[0] == 0:
            firsts += 1
        all_opp.extend([ar[i] for i in range(4) if i != 0])
        steps_list.append(s)
    p0_m = float(np.mean(p0_rewards))
    opp_m = float(np.mean(all_opp))
    return dict(
        mean_reward=round(p0_m, 2), std_reward=round(float(np.std(p0_rewards)), 2),
        settlement_win_rate=round(wins / n_episodes, 4),
        first_finish_rate=round(firsts / n_episodes, 4),
        mean_510k_score=round(float(np.mean(p0_510k)), 2),
        mean_opponent_reward=round(opp_m, 2),
        relative_superiority=round((p0_m - opp_m) / (abs(opp_m) + 1e-8), 4),
        mean_episode_steps=round(float(np.mean(steps_list)), 2),
        std_episode_steps=round(float(np.std(steps_list)), 2),
    )


def random_baseline(env, n_episodes=EVAL_BASELINE_EPISODES, seed=999):
    wins = [0] * 4
    p_r = [[], [], [], []]
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        game = cast(FiveTenKEnv, env.unwrapped).game
        done = False
        while not done:
            mask = info['action_mask']
            action = int(np.random.choice(np.where(mask)[0]))
            obs, reward, done, truncated, info = env.step(action)
        scorer = Scorer(game)
        ar = scorer.compute_rewards()
        w = settlement_winner(ar)
        for pid in range(4):
            p_r[pid].append(ar.get(pid, 0.0))
            if w == pid:
                wins[pid] += 1
    return dict(
        mean_reward=round(float(np.mean(p_r[0])), 2),
        std_reward=round(float(np.std(p_r[0])), 2),
        settlement_win_rate=round(wins[0] / n_episodes, 4),
        first_finish_rate=round(
            sum(1 for ep in range(n_episodes) for _ in [0]), 4
        ),  # placeholder, filled retroactively
        mean_510k_score=round(float(np.mean([np.mean(p_r[i]) for i in range(4)])), 2),
    )


# ========== PHASE 1: TRAIN ONE SEED ==========

def train_seed(seed, eval_env, log_f):
    """Train seed 0→2^22, evaluate at EVAL_POINTS. Returns {step: metrics}."""
    log_f.write(f'\n--- Seed {seed} ---\n'); log_f.flush()
    os.makedirs(os.path.join(TRAIN_SUBDIR, f'seed_{seed}'), exist_ok=True)

    train_env = SeededResetWrapper(make_env(), base_seed=seed * 1000)
    model = MaskablePPO(
        MaskableActorCriticPolicy, train_env,
        learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=0, seed=seed,
        tensorboard_log=os.path.join(BASE_DIR, f'tb_logs/seed_{seed}'),
    )

    results = {}
    seed_dir = os.path.join(TRAIN_SUBDIR, f'seed_{seed}')

    # Step 0: random policy
    m = evaluate(model, eval_env, EVAL_N_EPISODES, seed=seed * 100 + 1)
    results[0] = m
    log_f.write(f'  [Step       0]  reward={m["mean_reward"]:+5.1f}  '
                f'win_rate={m["settlement_win_rate"]:.1%}\n')

    # Train in segments: each segment trains up to the next eval point
    # (except the first few where the delta is smaller than n_steps=2048)
    prev_actual = 0
    for target in EVAL_POINTS:
        delta = target - prev_actual
        if delta <= 0:
            continue

        t0 = time.time()
        model.learn(total_timesteps=delta, reset_num_timesteps=False)
        actual = model.num_timesteps
        elapsed = time.time() - t0

        if actual <= prev_actual:
            continue  # no new training happened (shouldn't occur)

        # Save checkpoint
        model.save(os.path.join(seed_dir, f'model_{actual}.zip'))

        # Evaluate
        m = evaluate(model, eval_env, EVAL_N_EPISODES, seed=seed * 100 + 1)
        results[actual] = m
        log_f.write(f'  [Step {actual:>7d}]  reward={m["mean_reward"]:+5.1f}  '
                    f'win_rate={m["settlement_win_rate"]:.1%}  '
                    f'first={m["first_finish_rate"]:.1%}  '
                    f'510K={m["mean_510k_score"]:.1f}  '
                    f'rel_sup={m["relative_superiority"]:+5.2f}  '
                    f'[{elapsed:.0f}s, target={target}]\n')
        log_f.flush()

        prev_actual = actual

        if actual >= TOTAL_STEPS:
            break

    model.save(os.path.join(seed_dir, 'model_final.zip'))
    train_env.close()
    _save_seed_csv(seed, results, seed_dir)
    return results


def _save_seed_csv(seed, results, seed_dir):
    fields = [
        'step', 'mean_reward', 'std_reward', 'settlement_win_rate',
        'first_finish_rate', 'mean_510k_score', 'mean_opponent_reward',
        'relative_superiority', 'mean_episode_steps', 'std_episode_steps',
    ]
    with open(os.path.join(seed_dir, 'metrics.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in sorted(results):
            w.writerow({'step': s, **results[s]})


# ========== PHASE 2: PLOT ==========

def load_all():
    data = {}
    for s in SEEDS:
        path = os.path.join(TRAIN_SUBDIR, f'seed_{s}', 'metrics.csv')
        if not os.path.exists(path):
            continue
        data[s] = {}
        with open(path) as f:
            for row in csv.DictReader(f):
                step = int(row['step'])
                data[s][step] = {k: float(v) for k, v in row.items() if k != 'step'}
    return data


def plot_multi(data, baseline):
    os.makedirs(PLOTS_SUBDIR, exist_ok=True)
    common = sorted(set.intersection(*[set(d.keys()) for d in data.values()])
                    if data else [])

    def agg(key):
        return [{'mean': float(np.mean([data[s][st][key] for s in data])),
                 'std': float(np.std([data[s][st][key] for s in data], ddof=1))}
                for st in common]

    r = agg('mean_reward'); w = agg('settlement_win_rate')
    f = agg('first_finish_rate'); rs = agg('relative_superiority')
    k = agg('mean_510k_score'); st = agg('mean_episode_steps')

    fig, axs = plt.subplots(5, 1, figsize=(12, 16), sharex=True)

    def _p(ax, x, ylabel, color, bl=None):
        m = [v['mean'] for v in x]; sd = [v['std'] for v in x]
        ax.fill_between(common, [a - b for a, b in zip(m, sd)],
                        [a + b for a, b in zip(m, sd)], alpha=0.2, color=color)
        ax.plot(common, m, 'o-', color=color, markersize=3)
        if bl is not None:
            ax.axhline(bl, color='gray', linestyle=':', label=f'Random ({bl})')
            ax.legend(fontsize=8)
        ax.set_ylabel(ylabel); ax.set_xscale('log'); ax.grid(True, alpha=0.3)

    axs[0].set_title('Multi-Seed Convergence (n=3 seeds, shaded=±1σ)')
    _p(axs[0], r, 'Settlement Reward', 'C0', baseline['mean_reward'])
    _p(axs[1], w, 'Win Rate', 'C1', baseline['settlement_win_rate'])
    _p(axs[2], rs, 'Relative Superiority', 'C5')
    _p(axs[3], k, '510K Score', 'C6')
    _p(axs[4], st, 'Episode Steps', 'C2')
    axs[4].set_xlabel('Training Steps')
    for ax in axs:
        ax.set_xlim(common[0] * 0.8, common[-1] * 1.2)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_SUBDIR, 'multi_seed_convergence.png'), dpi=150)
    print(f'  Saved multi_seed_convergence.png')

    # Linear-scale zoom on phase B
    pbs = [s for s in common if s >= 2**20]
    if pbs:
        fig2, axs2 = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        for ax, vals, yl, c in [
            (axs2[0], w, 'Win Rate', 'C1'),
            (axs2[1], rs, 'Relative Superiority', 'C5'),
            (axs2[2], k, '510K Score', 'C6'),
        ]:
            idx = [common.index(s) for s in pbs]
            m = [vals[i]['mean'] for i in idx]
            sd = [vals[i]['std'] for i in idx]
            ax.fill_between(pbs, [a - b for a, b in zip(m, sd)],
                            [a + b for a, b in zip(m, sd)], alpha=0.2, color=c)
            ax.plot(pbs, m, 'o-', color=c, markersize=3)
            if yl == 'Win Rate':
                ax.axhline(baseline['settlement_win_rate'], color='gray',
                           linestyle=':', label=f'Random ({baseline["settlement_win_rate"]:.1%})')
                ax.legend(fontsize=8)
            ax.set_ylabel(yl); ax.grid(True, alpha=0.3)
        axs2[2].set_xlabel('Training Steps')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_SUBDIR, 'multi_seed_linear.png'), dpi=150)
        print(f'  Saved multi_seed_linear.png')

    # Merged CSV
    with open(os.path.join(BASE_DIR, 'merged_metrics.csv'), 'w', newline='') as f:
        w_csv = csv.writer(f)
        w_csv.writerow(['step', 'mean_reward', 'std_reward', 'mean_win_rate',
                        'std_win_rate', 'mean_relsup', 'std_relsup',
                        'mean_510k', 'std_510k'])
        for i, s in enumerate(common):
            w_csv.writerow([s] + [round(r[i]['mean'], 2), round(r[i]['std'], 2),
                                  round(w[i]['mean'], 4), round(w[i]['std'], 4),
                                  round(rs[i]['mean'], 4), round(rs[i]['std'], 4),
                                  round(k[i]['mean'], 2), round(k[i]['std'], 2)])
    print(f'  Saved merged_metrics.csv')
    return common


# ========== PHASE 3: HEAD-TO-HEAD ==========

def run_head2head(data):
    print(f'\n{"="*60}\n>>> Head-to-Head Tournament\n{"="*60}')
    os.makedirs(HH_SUBDIR, exist_ok=True)

    all_rows = []
    for seed in SEEDS:
        seed_dir = os.path.join(TRAIN_SUBDIR, f'seed_{seed}')
        # Map HH_CHECKPOINTS -> closest saved model
        model_cache = {}
        saved = [f for f in os.listdir(seed_dir) if f.startswith('model_') and f.endswith('.zip')]
        for ckpt in HH_CHECKPOINTS:
            best = min(saved, key=lambda f: abs(int(f.split('_')[1].split('.')[0]) - ckpt))
            step = int(best.split('_')[1].split('.')[0])
            model_cache[ckpt] = (step, MaskablePPO.load(os.path.join(seed_dir, best)))
            print(f'  [Seed {seed}] Loaded ~{HH_LABELS[ckpt]} (actual {step})')

        pairs = [
            (2**14, 2**22, 'early(16K)', 'late(4M)'),
            (2**17, 2**22, 'mid(131K)', 'late(4M)'),
            (2**20, 2**22, '1M', 'late(4M)'),
        ]

        for ca, cb, na, nb in pairs:
            if ca not in model_cache or cb not in model_cache:
                continue
            sa, ma = model_cache[ca]
            sb, mb = model_cache[cb]
            env = make_env()
            wa = wb = 0
            ra, rb = [], []
            for g in range(HEAD2HEAD_N_GAMES):
                if g % 2 == 0:
                    lineup = [(na, ma), (nb, mb), ('rand', None), ('rand', None)]
                else:
                    lineup = [(nb, mb), (na, ma), ('rand', None), ('rand', None)]
                obs, info = env.reset(seed=seed + g * 7 + 13)
                game = cast(FiveTenKEnv, env.unwrapped).game
                done = False
                while not done:
                    pid = game.current_player
                    _, mdl = lineup[pid]
                    if mdl is None:
                        action = int(np.random.choice(np.where(info['action_mask'])[0]))
                    else:
                        action, _ = mdl.predict(obs, action_masks=info['action_mask'],
                                                deterministic=True)
                    obs, reward, done, truncated, info = env.step(action)
                scorer = Scorer(game)
                ar = scorer.compute_rewards()
                ra.append(ar.get(0 if lineup[0][0] == na else 1, 0.0))
                rb.append(ar.get(0 if lineup[0][0] == nb else 1, 0.0))
                fp = game.finish_order[0] if game.finish_order else -1
                if lineup[fp][0] == na: wa += 1
                elif lineup[fp][0] == nb: wb += 1
            env.close()
            wna, wnb = wa / HEAD2HEAD_N_GAMES, wb / HEAD2HEAD_N_GAMES
            print(f'  {na}({sa}) vs {nb}({sb}): {na} {wna:.1%} vs {nb} {wnb:.1%}  '
                  f'→ {na if wna > wnb else nb}')
            all_rows.append(dict(seed=seed, model_a=na, model_b=nb,
                                 step_a=sa, step_b=sb,
                                 win_rate_a=round(wna, 4), win_rate_b=round(wnb, 4),
                                 mean_reward_a=round(float(np.mean(ra)), 2),
                                 mean_reward_b=round(float(np.mean(rb)), 2)))

    if all_rows:
        p = os.path.join(HH_SUBDIR, 'results.csv')
        with open(p, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader(); w.writerows(all_rows)
        print(f'  Saved {p}')


# ========== MAIN ==========

def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(TRAIN_SUBDIR, exist_ok=True)
    eval_env = make_env()

    # Random baseline
    print('Random baseline...')
    base = random_baseline(eval_env)
    print(f'  reward={base["mean_reward"]:+5.1f}  win_rate={base["settlement_win_rate"]:.1%}')
    with open(os.path.join(BASE_DIR, 'random_baseline.json'), 'w') as f:
        json.dump(base, f)

    # Train seeds
    log_path = os.path.join(BASE_DIR, 'experiment_progress.log')
    with open(log_path, 'w') as lf:
        lf.write(f'Start: {time.ctime()}\n')
        lf.write(f'Seeds: {SEEDS}, Eval points: {len(EVAL_POINTS)} per seed\n\n')
        all_data = {}
        for seed in SEEDS:
            r = train_seed(seed, eval_env, lf)
            all_data[seed] = r
        lf.write(f'\nDone: {time.ctime()}\n')

    eval_env.close()

    # Plot
    print(f'\n{"="*60}\n>>> Plotting\n{"="*60}')
    data = load_all()
    plot_multi(data, base)

    # Head-to-head
    run_head2head(data)

    print(f'\n{"="*60}\nComplete! Results in {BASE_DIR}/')
    print(f'  Progress log: {log_path}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
