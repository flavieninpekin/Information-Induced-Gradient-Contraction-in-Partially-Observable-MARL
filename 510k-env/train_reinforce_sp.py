"""
Self-play REINFORCE for 510K. All 4 players share the same policy.
Uses obs_utils for any-player observation — enables true self-play.
"""
import os, sys, json, time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '510k-env'))
from env.game import Game, GameMode
from env.obs_utils import obs_for_player, action_mask_for_player

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models_510k_reinforce_sp')
KAPPA_DIR = os.path.join(os.path.dirname(__file__), '..', '510k_kappa_reinforce_sp')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(KAPPA_DIR, exist_ok=True)

MAX_ACTIONS = 300
MODES = ['single', 'dynamic']
SEEDS = list(range(41, 49))  # 8 seeds
TOTAL_EPISODES = 20000
GAMMA = 0.99
LR = 3e-4
N_PLAYERS = 4


class PolicyNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=[256, 256]):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h)); layers.append(nn.ReLU()); prev = h
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_actions)

    def forward(self, obs, mask):
        x = self.net(obs)
        logits = self.head(x)
        logits[mask == 0] = -1e9
        return logits

    def act(self, obs, mask, deterministic=False):
        obs_t = torch.FloatTensor(obs).unsqueeze(0) if obs.ndim == 1 else obs
        mask_t = torch.IntTensor(mask).unsqueeze(0) if mask.ndim == 1 else mask
        with torch.no_grad():
            logits = self.forward(obs_t, mask_t)
            probs = F.softmax(logits, dim=-1)
            if deterministic:
                return probs.argmax(dim=-1), probs
            dist = torch.distributions.Categorical(probs)
            a = dist.sample()
            return a, probs


def run_episode(game, policy, agent_id=0):
    """Run one self-play episode. Collect (obs, action, reward) for agent."""
    game.reset()
    olist, alist, rlist = [], [], []
    agent_score_before = game.player_510k_scores[agent_id]

    while not game.is_over:
        pid = game.current_player
        obs = obs_for_player(game, pid)
        mask = action_mask_for_player(game, pid)

        if pid == agent_id:
            a, _ = policy.act(obs, mask)
            action = a.item()
            olist.append(obs.copy())
            alist.append(action)
            rlist.append(0.0)  # intermediate reward = 0
        else:
            a, _ = policy.act(obs, mask)
            action = a.item()

        valid = game.get_valid_actions(pid)
        if action == 0 and game.can_pass(pid):
            game.pass_turn(pid)
        else:
            idx = action - 1
            if 0 <= idx < len(valid):
                game.play_cards(pid, valid[idx].cards)
            elif valid:
                game.play_cards(pid, np.random.choice(valid).cards)
            elif game.can_pass(pid):
                game.pass_turn(pid)

    # Terminal reward: score difference since start
    final_score = game.player_510k_scores[agent_id]
    final_reward = final_score - agent_score_before
    if rlist:
        rlist[-1] += final_reward

    return olist, alist, rlist


def run_episode_masked(game, policy, agent_id, partner_seed):
    """Run episode for kappa eval — deterministic, fixed seed."""
    np.random.seed(partner_seed)
    random.seed(partner_seed)
    game.reset()
    olist, alist, rlist = [], [], []
    agent_score_before = game.player_510k_scores[agent_id]

    while not game.is_over:
        pid = game.current_player
        obs = obs_for_player(game, pid)
        mask = action_mask_for_player(game, pid)

        if pid == agent_id:
            a, _ = policy.act(obs, mask, deterministic=True)
            action = a.item()
            olist.append(obs.copy())
            alist.append(action)
            rlist.append(0.0)
        else:
            a, _ = policy.act(obs, mask, deterministic=True)
            action = a.item()

        valid = game.get_valid_actions(pid)
        if action == 0 and game.can_pass(pid):
            game.pass_turn(pid)
        else:
            idx = action - 1
            if 0 <= idx < len(valid):
                game.play_cards(pid, valid[idx].cards)
            elif valid:
                game.play_cards(pid, np.random.choice(valid).cards)
            elif game.can_pass(pid):
                game.pass_turn(pid)

    final_reward = game.player_510k_scores[agent_id] - agent_score_before
    if rlist:
        rlist[-1] += final_reward
    return olist, alist, rlist


def compute_returns(rewards):
    G = 0; ret = []
    for r in reversed(rewards):
        G = r + GAMMA * G; ret.insert(0, G)
    return torch.FloatTensor(ret)


def reinforce_step(policy, optim, buffer):
    if not buffer: return 0
    total_loss = torch.tensor(0.0); total_r = 0.0
    for olist, alist, rlist in buffer:
        if not olist: continue
        obs_t = torch.FloatTensor(np.array(olist))
        acts_t = torch.tensor(alist)
        ret = compute_returns(rlist)
        logits = policy(obs_t, torch.ones(obs_t.shape[0], MAX_ACTIONS, dtype=torch.int32))
        log_probs = F.log_softmax(logits, dim=-1)
        sel_lp = log_probs[range(len(acts_t)), acts_t]
        loss = -(sel_lp * ret).mean()
        total_loss = total_loss + loss; total_r += sum(rlist)
    loss = total_loss / max(len(buffer), 1)
    optim.zero_grad(); loss.backward(); optim.step()
    return total_r / max(len(buffer), 1)


def train_one(mode, seed):
    fp = os.path.join(MODEL_DIR, f'reinforce_sp_{mode}_seed{seed}.pt')
    if os.path.exists(fp):
        print(f'SKIP {mode} seed{seed}')
        return

    print(f'TRAIN SP {mode} seed{seed}...'); sys.stdout.flush()

    game = Game(mode=GameMode(mode), num_players=N_PLAYERS)
    game.reset()
    obs_dim = obs_for_player(game, 0).shape[0]
    policy = PolicyNet(obs_dim, MAX_ACTIONS)
    optim = torch.optim.Adam(policy.parameters(), lr=LR)

    ep = 0; buffer = []; t0 = time.time(); next_rpt = 500
    while ep < TOTAL_EPISODES:
        o, a, r = run_episode(game, policy)
        buffer.append((o, a, r)); ep += 1
        if len(buffer) >= 10:
            reinforce_step(policy, optim, buffer); buffer = []
        if ep >= next_rpt:
            fps = ep / max(time.time() - t0, 1)
            print(f'  {mode} s{seed}: ep {ep}  {fps:.1f}ep/s'); sys.stdout.flush()
            next_rpt += 500

    torch.save({'policy': policy.state_dict(), 'obs_dim': obs_dim}, fp)
    print(f'  DONE {mode} seed{seed} in {time.time()-t0:.0f}s'); sys.stdout.flush()


def kappa_grad(policy, traj_A, traj_B):
    grads = []
    for traj in [traj_A, traj_B]:
        total_grad = None; n = 0
        for olist, alist, rlist in traj:
            if not olist: continue
            obs_t = torch.FloatTensor(np.array(olist))
            acts_t = torch.tensor(alist)
            ret = compute_returns(rlist)
            logits = policy(obs_t, torch.ones(obs_t.shape[0], MAX_ACTIONS, dtype=torch.int32))
            log_probs = F.log_softmax(logits, dim=-1)
            sel_lp = log_probs[range(len(acts_t)), acts_t]
            loss = -(sel_lp * ret).mean()
            policy.zero_grad(); loss.backward()
            gv = torch.cat([p.grad.detach().clone().flatten()
                          for p in policy.parameters() if p.grad is not None])
            total_grad = gv if total_grad is None else total_grad + gv; n += 1
        grads.append(total_grad / max(n, 1) if total_grad is not None else torch.zeros(1))
    gA, gB = grads
    avg = (gA + gB) / 2.0
    return (torch.norm(avg)**2 / max((torch.norm(gA)**2 + torch.norm(gB)**2) / 2.0, 1e-10)).item()


if __name__ == '__main__':
    import random

    for mode in MODES:
        for seed in SEEDS:
            train_one(mode, seed)

    results = {}
    for mode in MODES:
        results[mode] = {}
        for seed in SEEDS:
            fp = os.path.join(MODEL_DIR, f'reinforce_sp_{mode}_seed{seed}.pt')
            if not os.path.exists(fp): continue
            ckpt = torch.load(fp, map_location='cpu')
            policy = PolicyNet(ckpt['obs_dim'], MAX_ACTIONS)
            policy.load_state_dict(ckpt['policy']); policy.eval()

            game = Game(mode=GameMode(mode), num_players=N_PLAYERS)
            ta = [run_episode_masked(game, policy, 0, seed * 100 + 1) for _ in range(30)]
            tb = [run_episode_masked(game, policy, 0, seed * 100 + 2) for _ in range(30)]
            ra = np.mean([sum(rl) for _,_,rl in ta])
            rb = np.mean([sum(rl) for _,_,rl in tb])
            k = kappa_grad(policy, ta, tb)
            results[mode][f'seed{seed}'] = {'kappa': k, 'rA': ra, 'rB': rb}
            print(f'SP {mode} s{seed}: κ={k:.4f} rA={ra:.2f} rB={rb:.2f}')

    print(f'\n{"="*60}')
    print('SELF-PLAY REINFORCE KAPPA')
    print(f'{"="*60}')
    for mode in MODES:
        vals = [v['kappa'] for v in results[mode].values()]
        if vals:
            print(f'{mode}: mean={np.mean(vals):.4f} std={np.std(vals):.4f} '
                  f'seeds={[f"{v:.3f}" for v in vals]}')
    with open(os.path.join(KAPPA_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
