# Gap Analysis: What Each Storyline Needs

## STORYLINE A: "κ as a Diagnostic Tool"

### HAVE ✅
- Toy A2C: HIDDEN 0.24 vs REVEALED 0.84 (8 seeds, κ stable, reward clean)
- 510K A2C: SINGLE 0.64 vs DYNAMIC 0.52 (8 seeds, direction correct)
- 510K DQN: reversal confirmed (8 seeds, S<D)
- 510K SAC: reversal confirmed (2 seeds)
- 510K REINFORCE: high κ variance, poor convergence (8 seeds)
- Overcooked PPO: STATIC 0.50 vs DYNAMIC 0.00 (8 seeds, reward 187 vs 0)
- Overcooked DQN: reward=0 for both modes (8 seeds)

### MISSING ❌
| Gap | Severity | Fix |
|-----|----------|-----|
| **Toy PPO results** | Medium | Run toy_ppo.py with multiseed — 10 min. Shows κ pattern on simplest env with PPO. |
| **510K PPO κ values** | High | Crashed on self-play. Options: (a) compute κ from existing `models/` checkpoints in `models/` dir, (b) run clean PPO single-agent (not self-play), (c) cite A2C as stand-in. |
| **Formal κ-vs-reward correlation** | Medium | Plot κ vs. final reward per seed per mode. Existing data can generate this. |
| **DQN/SAC gradient-structure explanation** | High | Need 2-3 paragraphs of formal analysis: why TD-loss gradient ≠ REINFORCE gradient. Math not code. |
| **Path integral analysis** | Low | Existing `path_integral.py` + `path_data/` checkpoints. Run and report. |
| **Multi-seed SAC** | Medium | SAC crashes due to numpy 2.0 + patterns.py. Either fix patterns permanently or cite 2-seed result as preliminary. |

### BOTTOM LINE
**Blocked on**: 510K PPO κ (needed for the "PPO confirms" narrative), DQN/SAC theoretical explanation.
**Quick win**: Toy PPO takes 10 minutes. 510K PPO: try loading the partial `models/` checkpoints and computing κ from them.


## STORYLINE B: "The Phenomenon & Its Limits"

### HAVE ✅
(Same as A above)

### MISSING ❌
| Gap | Severity | Fix |
|-----|----------|-----|
| **510K PPO κ** | Critical | Same as A. Without this, the "PPO confirms" claim is weak. |
| **Formal boundary conditions** | High | Systematically characterize *which* environments amplify the effect: action space size, obs dimension, reward sparsity. |
| **Statistical significance tests** | Medium | t-test/Wilcoxon between SINGLE and DYNAMIC κ distributions per env/algo. |
| **Effect strength vs. environment complexity table** | High | Quantify: Toy(Δκ=0.60) > Overcooked(Δκ=0.50) > 510K(Δκ=0.12). What drives this? |
| **Cross-environment κ ranking** | Low | Why does Toy have largest κ gap? Simple → easy to detect hidden info. Overcooked kills DYNAMIC entirely. |

### BOTTOM LINE
**Blocked on**: 510K PPO κ, formal boundary analysis.
**Quick win**: The Δκ vs complexity table. Statistical tests on existing data.


## STORYLINE C: "PPO = A2C + Clip"

### HAVE ✅
- A2C on all 3 environments: Toy (works), 510K (works), Overcooked (fails)
- PPO on Overcooked: works (reward 187 vs 0)
- REINFORCE on 510K: fails to converge (high κ variance)

### MISSING ❌
| Gap | Severity | Fix |
|-----|----------|-----|
| **PPO on Toy** | High | Need PPO Toy HIDDEN/REVEALED κ values to compare with A2C Toy. |
| **PPO on 510K** | Critical | Need 510K PPO κ to establish "PPO κ margin" vs "A2C κ margin". |
| **Head-to-head PPO vs A2C on same task** | Critical | Without 510K PPO, we have A2C (S>D, κ gap=0.12) but no PPO to compare. Need PPO 510K showing bigger gap or better convergence. |
| **Ablation: PPO without clip = A2C** | High | Theoretical argument needs empirical validation. Run PPO with clip_range=∞ (= no clipping) on 510K/Overcooked. |
| **Advantage variance measurement** | Medium | Show that κ variance correlates with advantage variance — the mechanistic link. |
| **Overcooked A2C reward analysis** | Low | Already have (r≈0 for both modes). Need to explain *why* — κ=0.5 for STATIC suggests orthogonal gradients, not cancelling, but A2C still fails. |

### BOTTOM LINE
**Blocked on**: 510K PPO (critical), head-to-head comparison, clip ablation.
**Most data-hungry** storyline. Needs at least 2-3 additional experiments.


## OVERALL BLOCKERS (across all storylines)

| Priority | Item | Storylines affected | Est. time |
|----------|------|---------------------|-----------|
| 🔴 P0 | **510K PPO κ values** | A, B, C | Try loading existing models/ first. If fail: retrain single-agent PPO (not self-play) ~1h |
| 🟡 P1 | **Toy PPO** | A, C | 10 min script |
| 🟡 P1 | **DQN/SAC theoretical explanation** | A, B | Writing, no code needed |
| 🟢 P2 | **Statistical significance** | B | Run on existing data |
| 🟢 P2 | **Δκ vs complexity table** | B | Compile from existing data |
| 🟢 P2 | **Clip ablation** | C | 1-2h training |
| ⚪ P3 | **Path integral** | A | Run existing scripts |

## RECOMMENDED ACTION

1. **Immediately**: Try to extract 510K PPO κ from the existing `models/` checkpoints (singe-mode models at various steps exist). Even partial data helps.
2. **If that fails**: Accept A2C as the primary 510K PG result. Storyline A works without PPO on 510K if we frame it as "A2C (a PG method) confirms".
3. **Pick storyline A**: It requires the FEWEST additional experiments. The only hard blocker is the DQN/SAC theoretical explanation — pure writing, no code.
