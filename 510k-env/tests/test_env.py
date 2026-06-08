import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
from env.env_510k import FiveTenKEnv


class TestEnvReset:
    def test_reset_returns_obs_and_info(self):
        env = FiveTenKEnv(mode='single')
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
        assert 'action_mask' in info

    def test_obs_shape(self):
        env = FiveTenKEnv(mode='single')
        obs, _ = env.reset()
        assert obs.shape == (116,)

    def test_action_mask_exists(self):
        env = FiveTenKEnv(mode='single')
        _, info = env.reset()
        mask = info['action_mask']
        assert mask.shape == (300,)
        assert mask.sum() >= 1  # At least pass

    def test_env_seed(self):
        env = FiveTenKEnv(mode='single')
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        assert np.array_equal(obs1, obs2)


class TestEnvStep:
    def test_step_single(self):
        env = FiveTenKEnv(mode='single')
        obs, info = env.reset(seed=0)
        mask = info['action_mask']
        valid = np.where(mask)[0]
        action = np.random.choice(valid)
        obs2, reward, done, truncated, info2 = env.step(action)
        assert isinstance(obs2, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert not done  # Should not end in one step

    def test_step_static(self):
        env = FiveTenKEnv(mode='static')
        obs, info = env.reset(seed=0)
        mask = info['action_mask']
        valid = np.where(mask)[0]
        action = np.random.choice(valid)
        obs2, reward, done, truncated, info2 = env.step(action)
        assert isinstance(reward, float)

    def test_step_dynamic(self):
        env = FiveTenKEnv(mode='dynamic')
        obs, info = env.reset(seed=0)
        mask = info['action_mask']
        valid = np.where(mask)[0]
        action = np.random.choice(valid)
        obs2, reward, done, truncated, info2 = env.step(action)
        assert isinstance(reward, float)


class Test3PlayerMode:
    def test_reset_3p(self):
        env = FiveTenKEnv(num_players=3)
        obs, info = env.reset()
        assert obs.shape == (116,)
        assert len(env.game.players) == 3
        sizes = [len(p.hand) for p in env.game.players]
        # During reset, opponents may auto-play any pattern (1-18 cards)
        assert sum(sizes) >= 36 and sum(sizes) <= 54
        assert all(s >= 0 for s in sizes)

    def test_3p_game_completes(self):
        env = FiveTenKEnv(num_players=3)
        obs, info = env.reset(seed=42)
        for _ in range(500):
            mask = info['action_mask']
            valid = np.where(mask)[0]
            action = np.random.choice(valid)
            obs, reward, done, truncated, info = env.step(action)
            if done:
                return
        pytest.fail("3p game did not complete in 500 steps")


class TestFullEpisodes:
    def test_single_mode_completes(self):
        env = FiveTenKEnv(mode='single')
        obs, info = env.reset(seed=42)
        total = 0
        for _ in range(500):
            mask = info['action_mask']
            valid = np.where(mask)[0]
            action = np.random.choice(valid)
            obs, reward, done, truncated, info = env.step(action)
            total += reward
            if done:
                break
        # Reward should be non-zero at end
        assert total != 0

    def test_static_mode_completes(self):
        env = FiveTenKEnv(mode='static')
        obs, info = env.reset(seed=42)
        for _ in range(1000):
            mask = info['action_mask']
            valid = np.where(mask)[0]
            if len(valid) == 0:
                break
            action = np.random.choice(valid)
            obs, reward, done, truncated, info = env.step(action)
            if done:
                return  # Game completed
        pytest.fail("Game did not complete in 1000 steps")

    def test_dynamic_mode_completes(self):
        env = FiveTenKEnv(mode='dynamic')
        obs, info = env.reset(seed=42)
        for _ in range(1000):
            mask = info['action_mask']
            valid = np.where(mask)[0]
            if len(valid) == 0:
                break
            action = np.random.choice(valid)
            obs, reward, done, truncated, info = env.step(action)
            if done:
                return  # Game completed
        pytest.fail("Game did not complete in 1000 steps")

    def test_multiple_episodes(self):
        env = FiveTenKEnv(mode='single')
        for ep in range(5):
            obs, info = env.reset(seed=ep)
            for _ in range(500):
                mask = info['action_mask']
                valid = np.where(mask)[0]
                action = np.random.choice(valid)
                obs, reward, done, truncated, info = env.step(action)
                if done:
                    break
