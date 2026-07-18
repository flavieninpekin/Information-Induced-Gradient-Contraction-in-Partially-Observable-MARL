"""
MAPPO wrapper env: provides Dict observation {local, global} for centralized critic.
"""
import numpy as np
from gymnasium import spaces, Wrapper
from env.obs_utils import obs_for_player
from env.game import Game, GameMode


class MAPPOEnv(Wrapper):
    """Wraps FiveTenKEnv to provide Dict observation {local, global}.

    local: agent's own observation (112-dim)
    global: all 4 players' observations concatenated (448-dim)
    """

    def __init__(self, mode='single', num_players=4):
        from env.env_510k import FiveTenKEnv
        base_env = FiveTenKEnv(mode=mode, num_players=num_players)
        super().__init__(base_env)

        local_dim = base_env.observation_space.shape[0]
        global_dim = local_dim * 4

        self.observation_space = spaces.Dict({
            'local': spaces.Box(low=0, high=1, shape=(local_dim,), dtype=np.float32),
            'global': spaces.Box(low=0, high=1, shape=(global_dim,), dtype=np.float32),
        })
        self.action_space = base_env.action_space

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._make_dict_obs(), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return self._make_dict_obs(), reward, done, truncated, info

    def _make_dict_obs(self):
        game = self.env.game
        if game is None:
            local = np.zeros(self.env.observation_space.shape[0], dtype=np.float32)
            global_ = np.zeros(self.env.observation_space.shape[0] * 4, dtype=np.float32)
        else:
            local = obs_for_player(game, self.env.agent_id)
            global_ = np.concatenate([
                obs_for_player(game, i) for i in range(game.num_players)
            ])
        return {'local': local.astype(np.float32), 'global': global_.astype(np.float32)}

    def _get_action_mask(self):
        return self.env._get_action_mask()

    def _get_info(self):
        return self.env._get_info()

    def set_policy_bot(self, model):
        self.env.set_policy_bot(model)

    def get_attr(self, name):
        if name == '_get_action_mask':
            return lambda: self._get_action_mask()
        if name == '_get_info':
            return lambda: self._get_info()
        return getattr(self, name, None)
