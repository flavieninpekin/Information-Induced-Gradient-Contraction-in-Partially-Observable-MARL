"""
Medium-complexity toy: Hidden Partner with Safe Fallback.

3 actions: {trust_A, trust_B, hedge}
2 partner types (A, B) — hidden when not revealed.
Agent observes: noisy signal from partner (μ=±1, σ=1) OR direct partner ID.

Hedge = safe fallback (always 0 reward). In hidden mode, agent may converge to hedge
because it can't reliably distinguish partners → SHORT path, LOW reward.
In revealed mode, agent learns trust_A/B matching → LONG path, HIGH reward.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class HiddenPartnerEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, revealed=False, n_steps=30):
        super().__init__()
        self.revealed = revealed
        self.n_steps = n_steps

        if revealed:
            # 2-dim: [is_A, is_B]
            self.observation_space = spaces.Box(0, 1, (2,), dtype=np.float32)
        else:
            # 1-dim noisy signal from partner: value ~ Gaussian(±1, 1)
            self.observation_space = spaces.Box(-5, 5, (1,), dtype=np.float32)

        self.action_space = spaces.Discrete(3)  # 0=trust_A, 1=trust_B, 2=hedge
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if not hasattr(self, '_forced_partner'):
            self._forced_partner = -1
        if self._forced_partner >= 0:
            self.partner = self._forced_partner
        else:
            self.partner = np.random.randint(0, 2)  # 0=A, 1=B
        self.step_count = 0
        return self._obs(), {}

    def _obs(self):
        if self.revealed:
            return np.array([1 - self.partner, self.partner], dtype=np.float32)
        # Very noisy signal: σ=3, μ=±1 → SNR≈0.33 → ~37% error rate
        mu = 1.0 if self.partner == 0 else -1.0
        signal = mu + np.random.randn() * 3.0
        return np.array([signal], dtype=np.float32)

    def step(self, action):
        """
        trust_A (0) + partner A → +2, + partner B → -2
        trust_B (1) + partner B → +2, + partner A → -2
        hedge (2) → always 0
        """
        action = int(action)
        if action == 2:  # hedge → dominates noisy guessing
            reward = 0.8
        elif action == self.partner:  # trust matching partner
            reward = 2.0
        else:  # trust mismatching partner
            reward = -2.0
        self.step_count += 1
        done = self.step_count >= self.n_steps
        return self._obs(), reward, done, False, {}

    def set_partner(self, partner):
        self._forced_partner = partner
        self.partner = partner
        self.step_count = 0
