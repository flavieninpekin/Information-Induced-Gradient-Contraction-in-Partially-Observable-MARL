"""
Quick test of the Overcooked wrapper.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from overcooked_wrapper import OvercookedHiddenPartner

print('=== Testing SINGLE mode ===')
env = OvercookedHiddenPartner(layout_name='cramped_room', mode='single', horizon=50)
obs, _ = env.reset()
print(f'obs shape: {obs.shape}')
total_r = 0
for i in range(50):
    act = env.action_space.sample()
    obs, r, done, trunc, info = env.step(act)
    total_r += r
    if done:
        break
print(f'steps: {i+1}, total_reward: {total_r}')

print()
print('=== Testing DYNAMIC mode ===')
env2 = OvercookedHiddenPartner(layout_name='cramped_room', mode='dynamic',
                               horizon=100, switch_interval=10)
obs, _ = env2.reset()
partners_seen = set()
for i in range(100):
    act = env2.action_space.sample()
    obs, r, done, trunc, info = env2.step(act)
    partners_seen.add(info['partner_type'])
    if done:
        break
print(f'partner types seen: {partners_seen}')
print(f'switch_count: {info["partner_switch_count"]}')

print()
print('=== Testing STATIC mode ===')
env3 = OvercookedHiddenPartner(layout_name='cramped_room', mode='static', horizon=50)
for ep in range(3):
    obs, _ = env3.reset()
    ptype = None
    for _ in range(50):
        act = env3.action_space.sample()
        obs, r, done, trunc, info = env3.step(act)
        if ptype is None:
            ptype = info['partner_type']
        if done:
            break
    print(f'episode {ep}: partner={ptype}')

print()
print('All tests passed!')
