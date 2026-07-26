# Case Study: Diagnosing a Noisy Sensor with κ

## Scenario

A practitioner is training PPO on 510K with hidden teammate information.
Training is unstable; she suspects the problem lies in partial observability.
She installs a sensor that reveals teammate identity, but due to hardware noise,
the signal is only 75% accurate. Counter-intuitively, training gets **worse**.

## Diagnosis

She computes κ before and after installing the sensor:

```
Before (no sensor, 0% reveal):  κ = 0.534
After (noisy sensor, 75% reveal): κ = 0.324   ← 39% drop
```

κ tells her: the noisy sensor is creating MORE gradient conflict than having
no sensor at all. The agent learns to depend on the unreliable signal, then
gets burned when it's wrong — producing destructive gradient updates.

## Intervention Options

κ's diagnosis suggests two possible fixes:

**Option A: Remove the sensor.** If 100% accuracy is unachievable, removing the
noisy signal altogether is better than keeping it. Returns to κ = 0.534.

**Option B: Fix the sensor to 100% accuracy.** If reliable teammate
identification is possible, achieving consistent information eliminates
gradient contraction entirely. κ rises to 0.616.

Both options are improvements over the 75% condition.

## Outcome

The practitioner chooses Option B (feasible in 510K via OBVIOUS mode).
After upgrading to full team information:

```
κ = 0.324 (noisy)  →  κ = 0.616 (clean)
Training stability: significantly improved
```

κ not only diagnosed the problem but also verified the fix.

## Why This Matters

This workflow — measure κ → diagnose information conflict → intervene → verify with κ — demonstrates κ as a practical engineering tool, not just an academic metric. The continuous reveal curve (0%→25%→50%→75%→100%) provides the reference against which any new observation scheme can be benchmarked.
