## Why Value-Based Methods Show Reversed κ

### The Two Gradient Structures

Policy-gradient κ uses the **REINFORCE gradient**:

  g_PG(A) = E_{τ_A}[ ∇log π(a|s) · R(τ_A) ]

This gradient measures: "if I were to update the policy using ONLY data
from configuration A, which direction would the parameters move?"

When configuration A says "deliver soups" and configuration B says "cook
onions", g_PG(A) and g_PG(B) point in opposite directions → κ → 0.

Value-based κ uses the **TD-loss gradient**:

  g_DQN(A) = ∇_θ E_{(s,a,r,s')~A}[ (Q_θ(s,a) - (r + γ max Q_target(s',·)))² ]

This gradient measures: "how should I adjust the Q-function to better
predict values for transitions from configuration A?"

When configuration A and B are mixed in the replay buffer, the Q-function
must predict values for BOTH. The TD gradient updates the Q-function to
minimize prediction error across ALL stored transitions. Since both A and B
transitions are in the buffer, the gradient direction is a COMPROMISE that
reduces error for the mixture → gradients are more aligned → κ is HIGHER
in DYNAMIC mode (where the mixture is more uniform).

### Intuition

- PG gradient: "which action should I take?" → conflict when optimal actions differ
- TD gradient: "what will this action be worth?" → reconciliation when both worth predicting

### Formal Statement

In DYNAMIC mode with switching configurations A and B:

  κ_PG(D) = ||(g_A + g_B)/2||² / avg(||g_A||², ||g_B||²) → 0
    because g_A ≈ -g_B (opposite optimal actions)

  κ_DQN(D) = ||(g_A' + g_B')/2||² / avg(||g_A'||², ||g_B'||²) → 1
    because g_A' ≈ g_B' (both update toward same Q-values)

where g' denotes the TD-loss gradient.

### Prediction

This framework predicts: the κ reversal should be LARGEST when:
1. Hidden configurations require OPPOSITE optimal policies (PG conflict high)
2. Replay buffer contains UNIFORM mix of configurations (TD reconciliation high)

Overcooked provides the extreme case: chef vs waiter roles are opposite,
and the buffer naturally mixes both → PPO κ→0, DQN κ→0.64 (reversed).

### Testable Hypothesis

If we compute κ on SAC's ACTOR gradient (which is policy-gradient-like,
following π ← argmax_Q), the result should match PG methods (S > D).
Meanwhile, SAC's CRITIC gradient should match TD methods (D > S).

This would demonstrate that κ measures the gradient SOURCE, not the algorithm.
