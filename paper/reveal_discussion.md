# Discussion: The W-Shaped κ Curve

## What we found

Training PPO on 510K with progressively more teammate information:

```
Reveal fraction:   0%    25%    50%    75%   100%
κ:              0.534  0.446  0.514  0.324  0.616
                 ———   ———    ———    ———    ———
                 base  ↓12%   ↓4%    ↓39%   ↑15%
```

Partial information CAN be worse than no information—up to 39% worse at 75% reveal.

## Why it matters

Standard intuition: "more observability → better training." We show this is FALSE when the extra information is noisy.

The mechanism: inconsistent information forces the policy to oscillate between strategies. When team info is visible 75% of the time, the agent learns to depend on it—then gets burned when it's wrong. This creates gradient conflict more severe than if the agent had simply ignored team info entirely (0% reveal, where it commits to one consistent strategy).

This is the "almost-right penalty": information that's mostly correct but occasionally wrong is more damaging than no information at all.

## Connection to broader literature

This pattern has analogs in:
- **Catastrophic interference**: inconsistent training signals cause destructive gradient updates
- **Distributional shift in RL**: non-stationary observations degrade off-policy learning
- **Curriculum learning**: standard curricula assume monotonic improvement; we show that for hidden information, intermediate steps may be counterproductive
- **Robustness vs accuracy**: the "75% accurate but 25% adversarial" regime mirrors adversarial training dynamics

## Practical recommendation

For MARL system designers: when adding sensors or observations to agents:
1. If you can't provide the information consistently (close to 100% accuracy), consider NOT providing it at all
2. κ can be used as a diagnostic: if κ drops when a new observation is added, the observation may be introducing noise rather than signal
3. Jump discontinuously from "no information" to "full information" rather than incrementally improving

## Limitations

- 2 seeds per data point: W-shape pattern needs multi-seed confirmation
- Only tested on 510K with PPO: cross-environment and cross-algorithm generalization unknown
- Random bit-masking is a specific noise model; real-world partial observability may have structured noise
