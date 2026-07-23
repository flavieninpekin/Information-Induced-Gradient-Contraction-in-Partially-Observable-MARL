"""
Minimal discrete SAC for 510K with action masking.
Writes model checkpoints, computes kappa from actor gradient.
"""
import os, sys, json, time, multiprocessing
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '510k-env'))
from env.dqn_wrapper import FiveTenKMaskedEnv, MASK_DIM, MAX_ACTIONS


class Actor(nn.Module):
    """Policy network with action masking."""
    def __init__(self, obs_dim, mask_dim, n_actions, hidden=[256, 256]):
        super().__init__()
        self.state_dim = obs_dim - mask_dim
        self.n_actions = n_actions
        layers = []
        prev = self.state_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_actions)

    def forward(self, obs):
        state = obs[:, :self.state_dim]
        mask = obs[:, self.state_dim:]
        x = self.net(state)
        logits = self.head(x)
        logits[mask == 0] = -1e9
        return logits

    def get_action(self, obs, deterministic=False):
        logits = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        if deterministic:
            return probs.argmax(dim=-1), probs
        dist = torch.distributions.Categorical(probs)
        return dist.sample(), probs

    def log_prob(self, obs, action):
        logits = self.forward(obs)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs[range(len(action)), action]


class Critic(nn.Module):
    """Q-network with action masking."""
    def __init__(self, obs_dim, mask_dim, n_actions, hidden=[256, 256]):
        super().__init__()
        self.state_dim = obs_dim - mask_dim
        layers = []
        prev = self.state_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_actions)

    def forward(self, obs):
        state = obs[:, :self.state_dim]
        mask = obs[:, self.state_dim:]
        x = self.net(state)
        q = self.head(x)
        q[mask == 0] = -1e9
        return q


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buf = deque(maxlen=capacity)

    def add(self, obs, act, rew, next_obs, done, mask):
        self.buf.append((obs, act, rew, next_obs, done, mask))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        obs = torch.FloatTensor(np.array([b[0] for b in batch]))
        act = torch.tensor([b[1] for b in batch])
        rew = torch.FloatTensor([b[2] for b in batch])
        nxt = torch.FloatTensor(np.array([b[3] for b in batch]))
        don = torch.FloatTensor([float(b[4]) for b in batch])
        msk = torch.FloatTensor(np.array([b[5] for b in batch]))
        return obs, act, rew, nxt, don, msk

    def __len__(self):
        return len(self.buf)


class DiscreteSAC:
    """Discrete SAC with action masking and auto-tuned entropy."""

    def __init__(self, obs_dim, mask_dim, n_actions, lr=3e-4, gamma=0.99,
                 tau=0.005, alpha=0.1, device='cpu'):
        self.gamma = gamma
        self.tau = tau
        self.device = device

        self.actor = Actor(obs_dim, mask_dim, n_actions).to(device)
        self.critic1 = Critic(obs_dim, mask_dim, n_actions).to(device)
        self.critic2 = Critic(obs_dim, mask_dim, n_actions).to(device)
        self.target1 = Critic(obs_dim, mask_dim, n_actions).to(device)
        self.target2 = Critic(obs_dim, mask_dim, n_actions).to(device)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=lr)

        # Auto-tune alpha
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -np.log(n_actions) * 0.98
        self.alpha = alpha

        self.buffer = ReplayBuffer()

    @property
    def alpha_val(self):
        return self.log_alpha.exp().item()

    def select_action(self, obs, deterministic=False):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            act, _ = self.actor.get_action(obs_t, deterministic)
        return act.item()

    def update(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return

        obs, act, rew, nxt, don, msk = self.buffer.sample(batch_size)
        obs = obs.to(self.device); act = act.to(self.device)
        rew = rew.to(self.device); nxt = nxt.to(self.device)
        don = don.to(self.device)

        # Update critics
        with torch.no_grad():
            next_logits = self.actor(nxt)
            next_probs = F.softmax(next_logits, dim=-1)
            next_log_probs = F.log_softmax(next_logits, dim=-1)
            q1_next = self.target1(nxt)
            q2_next = self.target2(nxt)
            q_next = torch.min(q1_next, q2_next)
            v_next = (next_probs * (q_next - self.log_alpha.exp() * next_log_probs)).sum(dim=-1)
            target = rew + (1 - don) * self.gamma * v_next

        q1 = self.critic1(obs)[range(len(act)), act]
        q2 = self.critic2(obs)[range(len(act)), act]
        loss_c1 = F.mse_loss(q1, target)
        loss_c2 = F.mse_loss(q2, target)

        self.critic1_opt.zero_grad(); loss_c1.backward(); self.critic1_opt.step()
        self.critic2_opt.zero_grad(); loss_c2.backward(); self.critic2_opt.step()

        # Update actor
        logits = self.actor(obs)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            q1_a = self.critic1(obs)
            q2_a = self.critic2(obs)
            q_min = torch.min(q1_a, q2_a)
        actor_loss = (probs * (self.log_alpha.exp().detach() * log_probs - q_min)).sum(dim=-1).mean()

        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        # Update alpha
        with torch.no_grad():
            _, current_probs = self.actor.get_action(obs)
            current_log_probs = torch.log(current_probs + 1e-10)
        alpha_loss = -(self.log_alpha * (current_log_probs.sum(dim=-1).mean() + self.target_entropy)).mean()

        self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()

        # Soft update targets
        for tp, p in [(self.target1, self.critic1), (self.target2, self.critic2)]:
            for t_param, param in zip(tp.parameters(), p.parameters()):
                t_param.data.copy_(self.tau * param.data + (1 - self.tau) * t_param.data)

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
        }, path)

    def load(self, path, device='cpu'):
        ckpt = torch.load(path, map_location=device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic1.load_state_dict(ckpt['critic1'])
        self.critic2.load_state_dict(ckpt['critic2'])

    def actor_gradient(self, obs_batch, act_batch):
        """Policy gradient for a batch of (s,a) transitions (for kappa)."""
        obs = torch.FloatTensor(np.array(obs_batch)).to(self.device)
        act = torch.tensor(act_batch).to(self.device)
        logits = self.actor(obs)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            q1 = self.critic1(obs)
            q2 = self.critic2(obs)
            q_min = torch.min(q1, q2)
        loss = (probs * (self.log_alpha.exp().detach() * log_probs - q_min)).sum(dim=-1).mean()
        self.actor.zero_grad()
        loss.backward()
        gv = [p.grad.detach().clone().flatten() for p in self.actor.parameters() if p.grad is not None]
        return torch.cat(gv) if gv else torch.zeros(1)
