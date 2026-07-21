"""
DQN + action masking for 510K environment.
Follows the toy DQN pattern with TD-loss gradient + kappa.

Uses a custom Q-network that masks invalid actions to -1e9.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces

# Add 510k-env to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '510k-env'))
from env.env_510k import FiveTenKEnv, MAX_ACTIONS
from env.game import GameMode

MASK_DIM = MAX_ACTIONS  # 300


class MaskedQNetwork(nn.Module):
    """Q-network with action masking: q[mask==0] = -1e9."""

    def __init__(self, obs_dim, mask_dim, n_actions, net_arch=[256, 256]):
        super().__init__()
        self.state_dim = obs_dim - mask_dim
        self.n_actions = n_actions
        layers = []
        prev = self.state_dim
        for h in net_arch:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.features = nn.Sequential(*layers)
        self.q_head = nn.Linear(prev, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        state = obs[:, :self.state_dim]
        mask = obs[:, self.state_dim:]
        x = self.features(state)
        q = self.q_head(x)
        q[mask == 0] = -1e9
        return q


class MaskedDQNPolicy:
    """Wrapper to make MaskedQNetwork work with SB3 DQN."""

    def __init__(self, obs_dim, mask_dim, n_actions, net_arch=[256, 256]):
        self.obs_dim = obs_dim
        self.mask_dim = mask_dim
        self.n_actions = n_actions
        self.net_arch = net_arch

    def make_q_net(self):
        return MaskedQNetwork(self.obs_dim, self.mask_dim, self.n_actions, self.net_arch)


class FiveTenKMaskedEnv(gym.Wrapper):
    """Wraps FiveTenKEnv to concatenate action_mask into observation."""

    def __init__(self, mode='single', num_players=4):
        raw = FiveTenKEnv(mode=mode, num_players=num_players)
        super().__init__(raw)
        self.obs_dim = raw.obs_dim
        self.mask_dim = MASK_DIM
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.obs_dim + self.mask_dim,), dtype=np.float32
        )
        self.action_space = raw.action_space

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        mask = info.get('action_mask', np.ones(self.mask_dim, dtype=np.float32))
        return np.concatenate([obs, mask.astype(np.float32)]), info

    def step(self, action):
        rng = np.random.default_rng()
        mask = self._get_action_mask()
        if mask[int(action)] == 0:
            valid = np.where(mask > 0)[0]
            action = rng.choice(valid) if len(valid) > 0 else 0
        obs, rew, done, trunc, info = self.env.step(int(action))
        mask_next = info.get('action_mask', np.ones(self.mask_dim, dtype=np.float32))
        return np.concatenate([obs, mask_next.astype(np.float32)]), rew, done, trunc, info

    def _get_action_mask(self):
        raw = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
        return raw._get_action_mask()
