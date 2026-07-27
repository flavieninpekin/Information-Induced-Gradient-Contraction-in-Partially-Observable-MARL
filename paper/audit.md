# Pre-Submission Audit: Top Issues to Fix Before AAAI

## 🔴 Critical (will cause reject)

### 1. Overcooked STATIC κ=0.500±0.000 for ALL 8 seeds
Identity-identical κ across all seeds looks like a degenerate/mechanical result, not real data. A reviewer will flag this immediately.
→ **Fix**: Recompute κ for Overcooked STATIC with the correct REINFORCE gradient (not the noisy single-step reward). The all-0.500 might be because reward=0 per step → zero gradient → κ=0/0=0.5 by denominator clamp. Need to verify.

### 2. "22 independent PPO runs" doesn't match κ data
Abstract claims 22 runs. κ tables show 10/10/9/8/8/8/2 seeds from different experiments. These are two different experimental campaigns mixed together.
→ **Fix**: Either unify the seed counts or separate the path-integral analysis (old data) from the κ analysis (new data) into clearly labeled sections.

### 3. Missing Overcooked path integral in Stable-or-Stuck table
Table in Discussion (line 455-463) has "---" for OC STATIC/DYNAMIC path lengths. The whole point of the framework is having BOTH κ AND P.
→ **Fix**: Compute path integral for Overcooked models or remove the column.

## 🟡 Major (will get rejected in strong competition)

### 4. DQN/SAC reversal explanation is hand-wavy
"Replay buffer averages" is an intuition, not an explanation. With 30k submissions, a reviewer will demand proof.
→ **Fix**: Add a simple controlled experiment: compute κ on DQN using only ON-POLICY data (no replay buffer) vs. off-policy. Or cite Toy DQN hidden/revealed results to show the pattern exists there too.

### 5. Toy section says "PPO" but most κ data is A2C
Table 5.2 (Toy) reports PPO κ but the text in Sec 1 says "A2C." Unclear which algorithm the Toy path integral is from.
→ **Fix**: Pick one algorithm per environment for the main narrative. Make cross-algorithm results separate.

### 6. PPO SINGLE footnote draws attention to infrastructure problems
"one salvaged self-play run (NumPy 2.x incompatibility)" — reviewer thinks: "they couldn't even run their own experiments reliably."
→ **Fix**: Remove the PPO SINGLE row or cite only A2C as the 510K PG representative. The footnote is worse than just omitting the row.

## 🟢 Minor (won't kill the paper but weaken it)

### 7. Figure files may not exist
`fig_path_lengths.pdf`, `kappa_figure.pdf`, `fig:reveal` referenced but need to be generated and placed in `paper/figures/`.

### 8. REINFORCE result buried
Line 404-407 is a single paragraph about REINFORCE failing. This is actually important for the "κ predicts convergence" claim. Elevate to its own sub-table.

### 9. Overcooked A2C/DQN all reward=0
Not discussed in paper. These are interesting negative results that support "PG methods uniquely vulnerable" but they're omitted.

### 10. No sensitivity analysis
How does κ change with n_eps (number of evaluation rollouts)? Not discussed.

### 11. References are placeholder \cite{} commands
Need to populate references.bib with real citations.

## Priority Action Items (do these first)

1. **Fix Overcooked κ=0.500 mystery** — recompute with proper per-episode REINFORCE gradient (not per-step reward)
2. **Unify seed counts** between abstract, path integral tables, and κ tables
3. **Generate the 3 missing figure PDFs**
4. **Remove PPO SINGLE footnote** from cross-algorithm table—cite A2C instead
5. **Add DQN reversal mini-experiment** — quick Toy DQN comparison to show pattern exists in controlled setting
