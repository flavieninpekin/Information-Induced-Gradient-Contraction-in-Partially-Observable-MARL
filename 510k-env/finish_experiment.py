"""
收尾：合并 2 种子数据 → 绘图 → Head-to-head
"""
import os, csv, sys, random
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

SEEDS = [42, 123]
HH_CHECKPOINTS = [16384, 131072, 1048576, 2097152, 4194304]
HH_LABELS = {16384: '16K', 131072: '131K', 1048576: '1M', 2097152: '2M', 4194304: '4M'}
HH_N_GAMES = 400
BASE = 'experiment_results'
TRAIN_DIR = os.path.join(BASE, 'training')
HH_DIR = os.path.join(BASE, 'head2head')
PLOT_DIR = os.path.join(BASE, 'plots')
os.makedirs(HH_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


def mask_fn(env):
    return cast(FiveTenKEnv, env.unwrapped)._get_action_mask()


def make_env():
    return ActionMasker(FiveTenKEnv(mode='single', num_players=4), mask_fn)


def settlement_winner(all_rewards):
    return max(all_rewards, key=lambda k: all_rewards[k])


# ============ 1. PLOT ============

def load_seed(seed):
    path = os.path.join(TRAIN_DIR, f'seed_{seed}', 'metrics.csv')
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            step = int(row['step'])
            data[step] = {k: float(v) for k, v in row.items() if k != 'step'}
    return data


def plot():
    d42 = load_seed(42)
    d123 = load_seed(123)

    # Find common steps
    common = sorted(set(d42.keys()) & set(d123.keys()))
    print(f'Common eval points: {len(common)}')

    def agg(key):
        return [{'mean': float(np.mean([d42[s][key], d123[s][key]])),
                 'std': float(np.std([d42[s][key], d123[s][key]], ddof=1))}
                for s in common]

    r, w, f, rs, k, st = [agg(x) for x in
        ['mean_reward', 'settlement_win_rate', 'first_finish_rate',
         'relative_superiority', 'mean_510k_score', 'mean_episode_steps']]

    baseline = {'reward': 22.7, 'win_rate': 0.259}

    fig, axes = plt.subplots(5, 1, figsize=(12, 16), sharex=True)

    def _p(ax, vals, ylabel, c, bl=None):
        m = [v['mean'] for v in vals]; s = [v['std'] for v in vals]
        ax.fill_between(common, [a - b for a, b in zip(m, s)],
                        [a + b for a, b in zip(m, s)], alpha=0.2, color=c)
        ax.plot(common, m, 'o-', color=c, markersize=3)
        if bl is not None:
            ax.axhline(bl, color='gray', linestyle=':', label=f'Random ({bl})')
            ax.legend(fontsize=8)
        ax.set_ylabel(ylabel); ax.set_xscale('log'); ax.grid(True, alpha=0.3)

    axs = axes
    axs[0].set_title('Convergence (Seeds 42, 123; shaded = ±1σ)')
    _p(axs[0], r, 'Settlement Reward', 'C0', baseline['reward'])
    _p(axs[1], w, 'Win Rate', 'C1', f'{baseline["win_rate"]:.1%}')
    _p(axs[2], rs, 'Relative Superiority', 'C5')
    _p(axs[3], k, '510K Score', 'C6')
    _p(axs[4], st, 'Episode Steps', 'C2')
    axs[4].set_xlabel('Training Steps')
    for ax in axs:
        ax.set_xlim(common[0] * 0.8, common[-1] * 1.2)
    plt.tight_layout()
    p1 = os.path.join(PLOT_DIR, 'convergence_log.png')
    plt.savefig(p1, dpi=150); print(f'Saved: {p1}')

    # Linear zoom on Phase B
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
            s = [vals[i]['std'] for i in idx]
            ax.fill_between(pbs, [a - b for a, b in zip(m, s)],
                            [a + b for a, b in zip(m, s)], alpha=0.2, color=c)
            ax.plot(pbs, m, 'o-', color=c, markersize=3)
            if yl == 'Win Rate':
                ax.axhline(baseline['win_rate'], color='gray',
                           linestyle=':', label=f'Random ({baseline["win_rate"]:.1%})')
                ax.legend(fontsize=8)
            ax.set_ylabel(yl); ax.grid(True, alpha=0.3)
        axs2[2].set_xlabel('Training Steps')
        plt.tight_layout()
        p2 = os.path.join(PLOT_DIR, 'convergence_linear.png')
        plt.savefig(p2, dpi=150); print(f'Saved: {p2}')

    # Save merged CSV
    mpath = os.path.join(BASE, 'merged_2seed.csv')
    with open(mpath, 'w', newline='') as f:
        w_csv = csv.writer(f)
        w_csv.writerow(['step', 'mean_reward', 'std_reward', 'mean_win_rate',
                        'std_win_rate', 'mean_relsup', 'std_relsup',
                        'mean_510k', 'std_510k'])
        for i, s in enumerate(common):
            w_csv.writerow([s,
                round(r[i]['mean'], 2), round(r[i]['std'], 2),
                round(w[i]['mean'], 4), round(w[i]['std'], 4),
                round(rs[i]['mean'], 4), round(rs[i]['std'], 4),
                round(k[i]['mean'], 2), round(k[i]['std'], 2)])
    print(f'Saved: {mpath}')

    return common


# ============ 2. HEAD-TO-HEAD ============

def run_hh():
    results = []
    env = make_env()

    for seed in SEEDS:
        sd = os.path.join(TRAIN_DIR, f'seed_{seed}')
        models = {}
        for ckpt in HH_CHECKPOINTS:
            # Find closest model
            best = None; best_d = 1e9
            for f in os.listdir(sd):
                if not f.startswith('model_') or not f.endswith('.zip') or f == 'model_final.zip':
                    continue
                step = int(f.split('_')[1].split('.')[0])
                d = abs(step - ckpt)
                if d < best_d:
                    best_d = d; best = (step, f)
            if best:
                models[ckpt] = (best[0], MaskablePPO.load(os.path.join(sd, best[1])))
                print(f'  Seed {seed}: loaded ~{HH_LABELS[ckpt]} (actual {best[0]})')

        pairs = [
            (16384, 4194304, '16K', '4M'),
            (131072, 4194304, '131K', '4M'),
            (1048576, 4194304, '1M', '4M'),
        ]

        for ca, cb, na, nb in pairs:
            if ca not in models or cb not in models:
                continue
            sa, ma = models[ca]
            sb, mb = models[cb]
            wa = wb = 0
            ra, rb = [], []
            for g in range(HH_N_GAMES):
                if g % 2 == 0:
                    lineup = [(f'{na}(s{seed})', ma), (f'{nb}(s{seed})', mb),
                              ('rand', None), ('rand', None)]
                else:
                    lineup = [(f'{nb}(s{seed})', mb), (f'{na}(s{seed})', ma),
                              ('rand', None), ('rand', None)]
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
                    obs, _, done, _, info = env.step(action)
                ar = Scorer(game).compute_rewards()
                ra.append(ar.get(0 if lineup[0][0] == f'{na}(s{seed})' else 1, 0.0))
                rb.append(ar.get(0 if lineup[0][0] == f'{nb}(s{seed})' else 1, 0.0))
                fp = game.finish_order[0] if game.finish_order else -1
                if fp >= 0:
                    if lineup[fp][0] == f'{na}(s{seed})': wa += 1
                    elif lineup[fp][0] == f'{nb}(s{seed})': wb += 1
            env.close()

            wna, wnb = wa / HH_N_GAMES, wb / HH_N_GAMES
            mra, mrb = float(np.mean(ra)), float(np.mean(rb))
            results.append(dict(seed=seed, early=na, late=nb,
                                step_early=sa, step_late=sb,
                                win_rate_early=round(wna, 4),
                                win_rate_late=round(wnb, 4),
                                reward_early=round(mra, 2),
                                reward_late=round(mrb, 2)))
            winner = na if wna > wnb else nb
            print(f'  {na}(s{seed}) vs {nb}(s{seed}): {wna:.1%} vs {wnb:.1%} → {winner}')

    if results:
        p = os.path.join(HH_DIR, 'results.csv')
        with open(p, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader(); w.writerows(results)
        print(f'Saved: {p}')

    env.close()
    return results


# ============ MAIN ============

if __name__ == '__main__':
    print('=== 1. Plotting ===')
    plot()
    print('\n=== 2. Head-to-Head ===')
    r = run_hh()
    print(f'\n=== Done ===')
    for row in r:
        w = row['win_rate_early'] > row['win_rate_late']
        print(f"  Seed {row['seed']}: {row['early']}(~{row['step_early']}) "
              f"{'BEATS' if w else 'LOSES TO'} "
              f"{row['late']}(~{row['step_late']}) "
              f"({row['win_rate_early']:.1%} vs {row['win_rate_late']:.1%})")
