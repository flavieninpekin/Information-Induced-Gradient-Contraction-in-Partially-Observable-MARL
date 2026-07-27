# Information-Induced Gradient Contraction in Partially Observable MARL

Research on how hidden relational information (teammate identity, partner roles) causes policy gradient cancellation in multi-agent RL — formalized as **kappa (Gradient Retention Ratio)**.

## What This Project Has Attempted

### Environments
- **510K** (4-player card game) — 4 modes: SINGLE (solo), STATIC (fixed teams), DYNAMIC (hidden teams), OBVIOUS (hidden teams+known info ablation). 22 seeds across modes.
- **Toy Matching** — minimal 2-action contextual bandit with HIDDEN/REVEALED partner types. PPO, DQN, A2C.
- **Overcooked** — cooperative cooking with V1/V2/V3 variants (partner hidden vs visible, chef/waiter roles). ~54 seeds total.

### Algorithms Tested
| Algorithm | 510K | Overcooked | Toy |
|-----------|------|-----------|-----|
| PPO (MaskablePPO) | 22 seeds | 16 seeds | 10 seeds |
| A2C | 8 seeds | 8 seeds | 1 seed |
| DQN | 8 seeds | 8 seeds | 1 seed |
| SAC | 2 seeds (crashed) | — | — |
| REINFORCE | 8 seeds | — | — |
| MAPPO | 5 seeds (incomplete) | — | — |

### Analysis Methods
- **Path Integral** — cumulative L2 distance through behavioral feature space across training
- **kappa** (Gradient Retention Ratio) — gradient alignment between hidden configurations
- **Stable-or-Stuck** — joint kappa × path integral diagnostic (4 regimes)
- **Drift-Diffusion** — decomposes trajectory into drift vs step variance
- **IRL** (Phase 1, abandoned) — MaxEnt and contrastive IRL for reward recovery
- **Feature Ablation** — 7/7 single-feature deletion confirms pattern robustness
- **Pythagorean Decomposition** — shared vs differential gradient energy

### Key Findings
- PG methods (PPO, A2C): kappa is higher in SINGLE than DYNAMIC (gradients cancel under hidden info)
- Value-based methods (DQN, SAC): pattern *reverses* (D > S) — TD gradients resist cancellation
- Overcooked V3: DYNAMIC mode kappa = 0.000 across all 8 seeds (total gradient cancellation)
- OBVIOUS ablation isolates information structure as causal driver
- Toy env: kappa~0 in HIDDEN reveals path integral alone is ambiguous

### Paper Status
Targeting **AAAI-27** (deadline July 28, 2026). Paper draft with appendix and 8 figures in `paper/`. Latest: continuous reveal experiments showing how gradient information is gradually recovered as hidden info is incrementally revealed.

### Dead Ends
- IRL diagnostic → abandoned as trivial framing
- LLM reward prediction → proposed but never executed
- Overcooked V1 → greedy/random partner not discriminative
- Overcooked V2 → suspected gradient computation bug
- MAPPO → PyTorch opcode bug blocked full training
- SAC → only 2 seeds before crash
