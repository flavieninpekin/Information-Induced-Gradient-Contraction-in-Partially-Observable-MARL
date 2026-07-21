"""
Overcooked wrapper v3 - STATIC vs DYNAMIC.

Partner roles:
  chef    - cooks soup; agent should DELIVER
  waiter  - delivers soup; agent should COOK

STATIC: partner type OBSERVABLE (99-dim obs = 96 state + 3 one-hot)
DYNAMIC: partner type HIDDEN (96-dim obs), switches mid-episode

κ hypothesis: DYNAMIC κ < STATIC κ
"""
import numpy as np
import gymnasium as gym

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action

from partner_agents import ChefAgent, WaiterAgent

PARTNER_TYPES = ['chef', 'waiter']
DEFAULT_LAYOUT = 'cramped_room'


class OvercookedV3Env(gym.Env):
    """
    STATIC: partner type visible as extra obs dims
    DYNAMIC: partner type hidden, switches mid-episode
    """

    metadata = {'render_modes': []}

    def __init__(self, layout_name=DEFAULT_LAYOUT, mode='static',
                 horizon=400, switch_interval=30, seed=None):
        super().__init__()
        self.layout_name = layout_name
        self.mode = mode
        self.horizon = horizon
        self.switch_interval = switch_interval
        self.seed_val = seed

        self._partner_idx = 0
        self._switch_timer = 0
        self._steps = 0
        self._force_partner = None

        self._build_mdp_and_env()
        self._build_pool()

        self.action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))
        dummy = self._get_obs()
        self.observation_space = gym.spaces.Box(
            low=0, high=5, shape=dummy.shape, dtype=np.float32
        )

    def _build_mdp_and_env(self):
        self.mdp = OvercookedGridworld.from_layout_name(self.layout_name)
        self.base_env = OvercookedEnv.from_mdp(self.mdp, horizon=self.horizon)

    def _build_pool(self):
        self.pool = {
            'chef': ChefAgent(self.base_env.mlam),
            'waiter': WaiterAgent(self.base_env.mlam),
        }

    @property
    def ptype(self):
        return PARTNER_TYPES[self._partner_idx]

    def _get_obs(self):
        base = self.mdp.featurize_state(self.base_env.state, self.base_env.mlam)[0]
        base = base.astype(np.float32)
        if self.mode == 'static':
            onehot = np.zeros(len(PARTNER_TYPES), dtype=np.float32)
            onehot[self._partner_idx] = 1.0
            return np.concatenate([base, onehot])
        return base

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed_val = seed
        self.base_env.reset()
        for a in self.pool.values():
            a.reset(); a.set_agent_index(1); a.set_mdp(self.mdp)

        if self._force_partner is not None:
            self._partner_idx = PARTNER_TYPES.index(self._force_partner)
        else:
            self._partner_idx = np.random.randint(len(PARTNER_TYPES))
        self._switch_timer = 0
        self._steps = 0
        return self._get_obs(), {}

    def step(self, action):
        p_act = Action.INDEX_TO_ACTION[int(action)]
        other = self.pool[self.ptype].action(self.base_env.state)[0]
        joint = (p_act, other)
        _, r, done, info = self.base_env.step(joint)

        self._steps += 1
        if self._force_partner is None:
            self._switch_timer += 1
            if self.mode == 'dynamic' and self._switch_timer >= self.switch_interval:
                self._partner_idx = (self._partner_idx + 1) % len(PARTNER_TYPES)
                self._switch_timer = 0

        trunc = self._steps >= self.horizon
        if trunc: done = True
        info['partner_type'] = self.ptype
        return self._get_obs(), r, done, trunc, info

    def close(self):
        pass
