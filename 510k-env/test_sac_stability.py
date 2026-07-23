"""Quick SAC stability test."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '510k-env'))
from env.dqn_wrapper import FiveTenKMaskedEnv, MASK_DIM, MAX_ACTIONS
from env.discrete_sac import DiscreteSAC

env = FiveTenKMaskedEnv(mode='single')
sac = DiscreteSAC(env.observation_space.shape[0], MASK_DIM, MAX_ACTIONS, lr=3e-4, device='cpu')
obs, _ = env.reset()

for step in range(1000):
    action = sac.select_action(obs)
    next_obs, reward, done, trunc, info = env.step(action)
    mask = info.get('action_mask', np.ones(MASK_DIM, dtype=np.float32))
    sac.buffer.add(obs, action, reward, next_obs, done, mask)
    if len(sac.buffer) >= 64:
        sac.update(batch_size=64)
    obs = next_obs
    if done or trunc:
        obs, _ = env.reset()
    if step % 200 == 0:
        print(f'{step} steps OK')
        import sys; sys.stdout.flush()

print('ALL 1000 STEPS OK - SAC stable!')
