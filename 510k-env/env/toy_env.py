"""
Analytic Toy: Hidden Matching Environment.

A 2-action contextual bandit where the true partner (B or C) is hidden.
Action 0 = cooperate with B, Action 1 = cooperate with C.
When hidden, gradients from the two partners cancel → no learning.
When revealed, agent learns to match.

This is the simplest environment exhibiting Information-Induced Gradient Contraction.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class HiddenMatchingEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, revealed=False, n_steps=10):
        super().__init__()
        self.revealed = revealed
        self.n_steps = n_steps

        if revealed:
            # 2-dim observation: [is_partner_B, is_partner_C]
            self.observation_space = spaces.Box(0, 1, (2,), dtype=np.float32)
        else:
            # 1-dim constant observation (no partner info)
            self.observation_space = spaces.Box(0, 1, (1,), dtype=np.float32)

        self.action_space = spaces.Discrete(2)
        self._forced_partner = -1
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._forced_partner >= 0:
            self.partner = self._forced_partner
        else:
            self.partner = np.random.randint(0, 2)
        self.step_count = 0
        return self._obs(), {}

    def _obs(self):
        if self.revealed:
            return np.array([1 - self.partner, self.partner], dtype=np.float32)
        return np.array([0.0], dtype=np.float32)

    def step(self, action):
        # Action 0 → cooperate with B, Action 1 → cooperate with C
        reward = 1.0 if int(action) == self.partner else -1.0
        self.step_count += 1
        done = self.step_count >= self.n_steps
        return self._obs(), reward, done, False, {}

    def set_partner(self, partner):
        """Force partner for gradient measurement."""
        self._forced_partner = partner
        self.partner = partner
        self.step_count = 0
