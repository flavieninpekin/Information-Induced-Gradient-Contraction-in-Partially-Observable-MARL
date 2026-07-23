"""
Master experiment runner for AAAI-27 paper.
Coordinates all training + kappa computation across environments and algorithms.

Environments: 510K, Overcooked (cramped_room)
Algorithms: PPO, DQN, SAC
Modes: single/static/dynamic

Usage:
    python run_all.py --all            # Run everything
    python run_all.py --env 510k       # Run 510K experiments only
    python run_all.py --algo ppo       # Run PPO experiments only
    python run_all.py --kappa-only     # Only compute kappa from existing models
"""
import os, sys, json, time, argparse, subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(ROOT)

# Add all source paths
sys.path.insert(0, os.path.join(PROJECT, '510k-env'))
sys.path.insert(0, os.path.join(PROJECT, 'overcooked_adapt'))

RESULTS_DIR = os.path.join(PROJECT, 'final_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEEDS_QUICK = [41, 42]      # 2 seeds for fast testing
SEEDS_FULL  = list(range(41, 49))  # 8 seeds for publication

CONFIG = {
    'ppo_510k': {
        'seeds': SEEDS_FULL,
        'timesteps': 1_000_000,
        'modes': ['single', 'static', 'dynamic'],
        'script': '510k-env/train_selfplay.py',
    },
    'ppo_overcooked': {
        'seeds': SEEDS_FULL,
        'timesteps': 1_000_000,
        'modes': ['static', 'dynamic'],
        'script': 'overcooked_adapt/train_v3.py',
    },
    'dqn_510k': {
        'seeds': SEEDS_FULL,
        'timesteps': 1_000_000,
        'modes': ['single', 'dynamic'],
        'script': '510k-env/train_510k_dqn.py',
    },
    'sac_510k': {
        'seeds': SEEDS_FULL,
        'timesteps': 1_000_000,
        'modes': ['single', 'dynamic'],
        'script': '510k-env/train_510k_sac.py',
    },
}

KAPPA_SCRIPTS = {
    'overcooked': 'overcooked_adapt/batch_kappa.py',
    '510k_dqn': '510k-env/train_510k_dqn.py',    # has built-in kappa
    '510k_sac': '510k-env/train_510k_sac.py',    # has built-in kappa
}


def run_cmd(cmd, cwd=PROJECT):
    """Run a command and stream output."""
    print(f'  CMD: {cmd}')
    sys.stdout.flush()
    result = subprocess.run(cmd, shell=True, cwd=cwd,
                           capture_output=False, text=True)
    print(f'  EXIT: {result.returncode}')
    return result.returncode


def train_510k_ppo():
    """Train PPO on 510K self-play."""
    cfg = CONFIG['ppo_510k']
    for mode in cfg['modes']:
        cmd = (f'python {cfg["script"]} --mode {mode} '
               f'--seeds {len(cfg["seeds"])} '
               f'--timesteps {cfg["timesteps"]}')
        run_cmd(cmd)


def train_overcooked_ppo():
    """Train PPO on Overcooked (parallel envs)."""
    cfg = CONFIG['ppo_overcooked']
    # train_v3.py handles all modes + seeds + kappa internally
    run_cmd(f'python {cfg["script"]}')


def train_510k_dqn():
    """Train DQN on 510K (parallel envs)."""
    run_cmd(f'python {CONFIG["dqn_510k"]["script"]}')


def train_510k_sac():
    """Train SAC on 510K (sequential)."""
    run_cmd(f'python {CONFIG["sac_510k"]["script"]}')


def collect_all_kappa():
    """Gather kappa results from all outputs into a single JSON."""
    all_results = {
        'ppo_510k': None,
        'ppo_overcooked': None,
        'dqn_510k': None,
        'sac_510k': None,
    }

    # Overcooked PPO
    path = os.path.join(PROJECT, 'overcooked_kappa_v3', 'results.json')
    if os.path.exists(path):
        with open(path) as f:
            all_results['ppo_overcooked'] = json.load(f)

    # 510K DQN
    path = os.path.join(PROJECT, '510k_kappa_dqn', 'results.json')
    if os.path.exists(path):
        with open(path) as f:
            all_results['dqn_510k'] = json.load(f)

    # 510K SAC
    path = os.path.join(PROJECT, '510k_kappa_sac', 'results.json')
    if os.path.exists(path):
        with open(path) as f:
            all_results['sac_510k'] = json.load(f)

    out = os.path.join(RESULTS_DIR, 'all_kappa.json')
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print(f'\n{"="*60}')
    print('KAPPA SUMMARY')
    print(f'{"="*60}')
    for name, data in all_results.items():
        if data is None:
            print(f'{name:20s}: NO DATA')
            continue
        for mode, vals in data.items():
            if isinstance(vals, dict):
                ks = [v['kappa'] for v in vals.values() if v]
                if ks:
                    print(f'{name:20s} {mode:10s}: mean={np.mean(ks):.4f} '
                          f'std={np.std(ks):.4f} n={len(ks)}')

    return all_results


if __name__ == '__main__':
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--env', choices=['510k', 'overcooked'])
    parser.add_argument('--algo', choices=['ppo', 'dqn', 'sac'])
    parser.add_argument('--kappa-only', action='store_true')
    parser.add_argument('--quick', action='store_true',
                       help='Use 2 seeds for fast testing')
    args = parser.parse_args()

    if args.quick:
        for k in CONFIG:
            CONFIG[k]['seeds'] = SEEDS_QUICK

    t0 = time.time()

    if args.kappa_only or args.all:
        print('\n=== COLLECTING KAPPA ===')
        collect_all_kappa()
        print(f'\nTotal time: {time.time()-t0:.0f}s')
        sys.exit(0)

    if args.all or args.env == '510k':
        if args.all or args.algo in (None, 'ppo'):
            print('\n=== 510K PPO ===')
            train_510k_ppo()
        if args.all or args.algo in (None, 'dqn'):
            print('\n=== 510K DQN ===')
            train_510k_dqn()
        if args.all or args.algo in (None, 'sac'):
            print('\n=== 510K SAC ===')
            train_510k_sac()

    if args.all or args.env == 'overcooked':
        if args.all or args.algo in (None, 'ppo'):
            print('\n=== OVERCOOKED PPO ===')
            train_overcooked_ppo()

    # Final kappa collection
    print('\n=== FINAL KAPPA ===')
    collect_all_kappa()

    print(f'\nTotal time: {(time.time()-t0)/3600:.1f}h')
