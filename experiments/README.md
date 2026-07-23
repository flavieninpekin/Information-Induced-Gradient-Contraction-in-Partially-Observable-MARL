# AAAI-27 Experiments

## Setup (once)

```bash
# Create Python env
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# Install deps
pip install -r experiments/requirements.txt

# Apply numpy 2.x compatibility patches
python experiments/setup.py
```

## Run

```bash
# Full experiment (PPO + DQN + SAC, all envs)
python experiments/run_all.py --all

# Quick sanity check (2 seeds, fast)
python experiments/run_all.py --quick --all

# Specific sub-experiments
python experiments/run_all.py --env 510k --algo ppo
python experiments/run_all.py --env overcooked --algo ppo
python experiments/run_all.py --kappa-only    # compute kappa from existing models
```

## What it runs

| Experiment | Algorithm | Env | Seeds | Steps | Est. time |
|-----------|-----------|-----|-------|-------|-----------|
| 510K | PPO | self-play | 8 | 1M | ~2h |
| 510K | DQN | parallel | 8 | 1M | ~2h |
| 510K | SAC | sequential | 8 | 1M | ~20h |
| Overcooked | PPO | parallel | 8 | 1M | ~1h |
| Total | — | — | — | — | ~25h |

## Output

- `models_*/` — trained models
- `*_kappa_*/results.json` — kappa values
- `final_results/all_kappa.json` — consolidated results

## B200 notes

- GPU: set `device='cuda'` (default), increase `N_ENVS` in train_v3.py for better utilization
- RAM: ~16GB sufficient
- Disk: ~2GB for all models
