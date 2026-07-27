"""Toy DQN: 10 seeds HIDDEN vs REVEALED."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '510k-env'))
from stable_baselines3 import DQN
from env.toy_env import HiddenMatchingEnv

results = {'HIDDEN': [], 'REVEALED': []}
for seed in range(10):
    for name, revealed in [('HIDDEN', False), ('REVEALED', True)]:
        env = HiddenMatchingEnv(revealed=revealed, n_steps=20)
        model = DQN('MlpPolicy', env, learning_rate=1e-3, buffer_size=2000, learning_starts=200,
                    batch_size=32, tau=0.005, gamma=0.99, train_freq=4,
                    target_update_interval=100, exploration_fraction=0.3,
                    exploration_initial_eps=1.0, exploration_final_eps=0.02,
                    policy_kwargs=dict(net_arch=[32, 32]), verbose=0, seed=seed, device='cpu')
        model.learn(total_timesteps=10000, progress_bar=False)
        model.q_net.eval()

        # TD gradient kappa
        grads = []
        for partner in [0, 1]:
            env._forced_partner = partner
            all_obs, all_next, all_act, all_rew, all_done = [], [], [], [], []
            for _ in range(20):
                env.partner = partner; o, _ = env.reset(); done = False
                while not done:
                    ot = torch.FloatTensor(o).unsqueeze(0)
                    with torch.no_grad(): q = model.q_net(ot)
                    a = q.argmax(dim=1).item()
                    no, r, done, _, _ = env.step(a)
                    all_obs.append(o); all_next.append(no); all_act.append(a)
                    all_rew.append(r); all_done.append(float(done)); o = no

            if not all_obs:
                grads.append(torch.zeros(1))
                continue

            bo = torch.FloatTensor(np.array(all_obs))
            bn = torch.FloatTensor(np.array(all_next))
            ba = torch.tensor(all_act)
            br = torch.tensor(all_rew, dtype=torch.float32)
            bd = torch.tensor(all_done, dtype=torch.float32)

            with torch.no_grad():
                nq = model.q_net_target(bn).max(dim=1)[0]
                tg = br + (1.0 - bd) * model.gamma * nq
            qv = model.q_net(bo)[range(len(ba)), ba]
            loss = ((qv - tg) ** 2).mean()
            model.q_net.zero_grad(); loss.backward()
            gv = torch.cat([p.grad.detach().clone().flatten() for p in model.q_net.parameters() if p.grad is not None])
            grads.append(gv)

        gA, gB = grads; avg = (gA + gB) / 2.0
        k = (torch.norm(avg)**2 / max((torch.norm(gA)**2 + torch.norm(gB)**2) / 2.0, 1e-10)).item()
        results[name].append(k)
        print(f'Toy DQN {name} s{seed}: k={k:.4f}')

print(f'\nToy DQN HIDDEN:  mean={np.mean(results["HIDDEN"]):.4f} std={np.std(results["HIDDEN"]):.4f}')
print(f'Toy DQN REVEALED: mean={np.mean(results["REVEALED"]):.4f} std={np.std(results["REVEALED"]):.4f}')
direction = 'R>H' if np.mean(results['REVEALED']) > np.mean(results['HIDDEN']) else 'REVERSED'
print(f'Direction: {direction}')
