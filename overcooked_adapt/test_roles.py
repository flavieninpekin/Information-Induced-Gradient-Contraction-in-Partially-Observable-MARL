"""Quick test of role-based partners."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from overcooked_role_env import OvercookedRoleEnv, PARTNER_TYPES
import numpy as np

print("Testing chef partner...")
env = OvercookedRoleEnv(mode='single', horizon=100)
env._current_partner_idx = 0  # force chef
obs, _ = env.reset()
rewards = []
for _ in range(100):
    act = env.action_space.sample()
    obs, r, done, trunc, info = env.step(act)
    rewards.append(r)
    if done: break
print(f"  Reward: {sum(rewards):.1f}")

print("Testing waiter partner...")
env2 = OvercookedRoleEnv(mode='single', horizon=100)
env2._current_partner_idx = 1
obs, _ = env2.reset()
rewards = []
for _ in range(100):
    act = env2.action_space.sample()
    obs, r, done, trunc, info = env2.step(act)
    rewards.append(r)
    if done: break
print(f"  Reward: {sum(rewards):.1f}")

print("Testing chaos partner...")
env3 = OvercookedRoleEnv(mode='single', horizon=100)
env3._current_partner_idx = 2
obs, _ = env3.reset()
rewards = []
for _ in range(100):
    act = env3.action_space.sample()
    obs, r, done, trunc, info = env3.step(act)
    rewards.append(r)
    if done: break
print(f"  Reward: {sum(rewards):.1f}")

print("Testing DYNAMIC mode...")
env4 = OvercookedRoleEnv(mode='dynamic', horizon=200, switch_interval=20)
obs, _ = env4.reset()
seen = set()
for _ in range(200):
    act = env4.action_space.sample()
    obs, r, done, trunc, info = env4.step(act)
    seen.add(info['partner_type'])
    if done: break
print(f"  Partner types seen: {seen}")

print("\nAll partner types working!")
env.close(); env2.close(); env3.close(); env4.close()
