"""
Vanilla REINFORCE (policy gradient) for 510K with action masking.
No GAE, no value function, no clipping — pure Monte Carlo policy gradient.

Kappa computation mirrors toy_experiment.py pattern.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '510k-env'))
from env.dqn_wrapper import FiveTenKMaskedEnv, MASK_DIM, MAX_ACTIONS
from env.discrete_sac import Actor  # reuse the masked actor network

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models_510k_reinforce')
KAPPA_DIR = os.path.join(os.path.dirname(__file__), '..', '510k_kappa_reinforce')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(KAPPA_DIR, exist_ok=True)

MODES = ['single', 'dynamic']
SEEDS = list(range(41, 49))
TOTAL_EPISODES = 5000     # episodes per seed
GAMMA = 0.99
LR = 3e-4
BATCH_EPISODES = 10       # update every N episodes


def collect_episode(env, actor):
    """Run one episode, return (obs, act, rew) sequences."""
    obs, _ = env.reset()
    olist, alist, rlist = [], [], []
    done = False
    while not done:
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            logits = actor(obs_t)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()
        olist.append(obs.copy())
        alist.append(action)
        next_obs, reward, done, trunc, _ = env.step(action)
        rlist.append(reward)
        obs = next_obs
    return olist, alist, rlist


def compute_returns(rewards, gamma=GAMMA):
    """Discounted cumulative returns."""
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.FloatTensor(returns)


def reinforce_loss(actor, episodes):
    """
    Compute REINFORCE loss: -mean(log_prob * return).
    Returns loss (scalar) and detached metrics.
    """
    total_loss = torch.tensor(0.0)
    total_return = 0.0
    n_steps = 0

    for olist, alist, rlist in episodes:
        if not olist:
            continue
        obs_t = torch.FloatTensor(np.array(olist))
        acts_t = torch.tensor(alist)
        returns = compute_returns(rlist)

        logits = actor(obs_t)
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs[range(len(acts_t)), acts_t]

        loss = -(selected_log_probs * returns).mean()
        total_loss = total_loss + loss
        total_return += sum(rlist)
        n_steps += len(olist)

    return total_loss / max(len(episodes), 1), total_return / max(len(episodes), 1), n_steps


def train_one(mode, seed):
    fp = os.path.join(MODEL_DIR, f'reinforce_{mode}_seed{seed}.pt')
    if os.path.exists(fp):
        print(f'SKIP {mode} seed{seed}')
        return

    print(f'TRAIN REINFORCE {mode} seed{seed}...')
    sys.stdout.flush()

    env = FiveTenKMaskedEnv(mode=mode)
    obs_dim = env.observation_space.shape[0]
    actor = Actor(obs_dim, MASK_DIM, MAX_ACTIONS)
    optim = torch.optim.Adam(actor.parameters(), lr=LR)

    episode = 0
    buffer = []
    t0 = time.time()
    next_report = 500

    try:
        while episode < TOTAL_EPISODES:
            olist, alist, rlist = collect_episode(env, actor)
            buffer.append((olist, alist, rlist))
            episode += 1

            if len(buffer) >= BATCH_EPISODES:
                loss, avg_r, n_steps = reinforce_loss(actor, buffer)
                optim.zero_grad()
                loss.backward()
                optim.step()
                buffer = []

            if episode >= next_report:
                elapsed = time.time() - t0
                fps = episode / elapsed if elapsed > 0 else 0
                print(f'  {mode} s{seed}: ep {episode}  {fps:.1f}ep/s  r={avg_r:.2f}')
                sys.stdout.flush()
                next_report += 500

        torch.save({'actor': actor.state_dict()}, fp)
        elapsed = time.time() - t0
        print(f'  DONE {mode} seed{seed} in {elapsed:.0f}s ({elapsed/60:.1f}m)')
        sys.stdout.flush()
    except Exception as e:
        print(f'  CRASH {mode} seed{seed}: {e}')
        import traceback; traceback.print_exc()
        sys.stdout.flush()
    finally:
        env.close()


def rollout(model, env, n_eps=30):
    """Deterministic rollout, returns (obs, action, reward) sequences."""
    trajectories = []
    for ep in range(n_eps):
        obs, _ = env.reset()
        olist, alist, rlist = [], [], []
        done = False
        while not done:
            olist.append(obs.copy())
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                logits = model(obs_t)
            act = logits.argmax(dim=1).item()
            alist.append(act)
            obs, r, done, trunc, _ = env.step(act)
            rlist.append(r)
        trajectories.append((olist, alist, rlist))
    return trajectories


def grad_kappa(actor, traj_A, traj_B):
    """Compute REINFORCE gradient for each trajectory set, return kappa."""
    grads = []
    for traj in [traj_A, traj_B]:
        total_grad = None
        n = 0
        for olist, alist, rlist in traj:
            if not olist: continue
            obs_t = torch.FloatTensor(np.array(olist))
            acts_t = torch.tensor(alist)
            returns = compute_returns(rlist)
            logits = actor(obs_t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected_lp = log_probs[range(len(acts_t)), acts_t]
            loss = -(selected_lp * returns).mean()
            actor.zero_grad()
            loss.backward()
            gv = torch.cat([p.grad.detach().clone().flatten()
                          for p in actor.parameters() if p.grad is not None])
            total_grad = gv if total_grad is None else total_grad + gv
            n += 1
        grads.append(total_grad / max(n, 1) if total_grad is not None else torch.zeros(1))
    gA, gB = grads
    avg = (gA + gB) / 2.0
    return (torch.norm(avg)**2 / max((torch.norm(gA)**2 + torch.norm(gB)**2) / 2.0, 1e-10)).item()


if __name__ == '__main__':
    for mode in MODES:
        for seed in SEEDS:
            train_one(mode, seed)

    # Kappa
    results = {}
    for mode in MODES:
        results[mode] = {}
        for seed in SEEDS:
            fp = os.path.join(MODEL_DIR, f'reinforce_{mode}_seed{seed}.pt')
            if not os.path.exists(fp): continue
            ckpt = torch.load(fp, map_location='cpu')
            obs_dim = 112 + MASK_DIM
            actor = Actor(obs_dim, MASK_DIM, MAX_ACTIONS)
            actor.load_state_dict(ckpt['actor'])
            actor.eval()

            env_a = FiveTenKMaskedEnv(mode=mode)
            ta = rollout(actor, env_a); ra = np.mean([sum(rl) for _,_,rl in ta]); env_a.close()
            env_b = FiveTenKMaskedEnv(mode=mode)
            tb = rollout(actor, env_b); rb = np.mean([sum(rl) for _,_,rl in tb]); env_b.close()

            k = grad_kappa(actor, ta, tb)
            results[mode][f'seed{seed}'] = {'kappa': k, 'rA': ra, 'rB': rb}
            print(f'REINFORCE {mode} s{seed}: κ={k:.4f} rA={ra:.2f} rB={rb:.2f}')

    print(f'\n{"="*60}')
    print('REINFORCE KAPPA')
    print(f'{"="*60}')
    for mode in MODES:
        vals = [v['kappa'] for v in results[mode].values()]
        if vals:
            print(f'{mode}: mean={np.mean(vals):.4f} std={np.std(vals):.4f} '
                  f'seeds={[f"{v:.3f}" for v in vals]}')
    with open(os.path.join(KAPPA_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
