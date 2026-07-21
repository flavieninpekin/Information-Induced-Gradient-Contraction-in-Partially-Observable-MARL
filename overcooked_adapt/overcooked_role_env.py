"""
Overcooked wrapper v2 - hidden partner ROLE.

Partner roles (require DIFFERENT agent strategies):
  chef    - cooks onions into soup; agent should DELIVER
  waiter  - delivers finished soups; agent should COOK
  chaos   - random+obstructive; agent must do everything alone

Modes:
  SINGLE  - fixed chaos partner; agent learns solo strategy
  DYNAMIC - hidden switching chef/waiter/chaos mid-episode

κ hypothesis: DYNAMIC < SINGLE because conflicting strategies cause
gradient contraction.
"""
import numpy as np
import gymnasium as gym

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import AgentPair
from overcooked_ai_py.mdp.actions import Action

from partner_agents import ChefAgent, WaiterAgent, ChaosAgent

PARTNER_TYPES = ['chef', 'waiter', 'chaos']
DEFAULT_LAYOUT = 'cramped_room'


class OvercookedRoleEnv(gym.Env):
    """
    Single-agent Overcooked where partner has a hidden role.
    Agent observes only the state, not the partner's role label.
    """

    metadata = {'render_modes': []}

    def __init__(self,
                 layout_name=DEFAULT_LAYOUT,
                 mode='single',
                 horizon=400,
                 switch_interval=40,
                 seed=None):
        super().__init__()

        self.layout_name = layout_name
        self.mode = mode
        self.horizon = horizon
        self.switch_interval = switch_interval
        self.seed_val = seed

        self._build_mdp_and_env()
        self._build_partner_pool()

        self.action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))
        dummy_obs = self._get_obs()
        self.observation_space = gym.spaces.Box(
            low=0, high=5, shape=dummy_obs.shape, dtype=np.float32
        )

        self._current_partner_idx = 0
        self._steps_since_switch = 0
        self._step_count = 0

    def _build_mdp_and_env(self):
        self.mdp = OvercookedGridworld.from_layout_name(self.layout_name)
        self.base_env = OvercookedEnv.from_mdp(self.mdp, horizon=self.horizon)

    def _build_partner_pool(self):
        self.partner_pool = {
            'chef': ChefAgent(self.base_env.mlam),
            'waiter': WaiterAgent(self.base_env.mlam),
            'chaos': ChaosAgent(),
        }

    @property
    def current_partner_type(self):
        if self.mode == 'single':
            return 'chaos'
        return PARTNER_TYPES[self._current_partner_idx]

    def _select_partner(self):
        if self.mode == 'single':
            return
        if self.mode == 'dynamic':
            if self._steps_since_switch >= self.switch_interval:
                self._current_partner_idx = (
                    self._current_partner_idx + 1
                ) % len(PARTNER_TYPES)
                self._steps_since_switch = 0

    def _get_partner_action(self):
        agent = self.partner_pool[self.current_partner_type]
        action, _ = agent.action(self.base_env.state)
        return action

    def _get_obs(self):
        obs_list = self.mdp.featurize_state(self.base_env.state, self.base_env.mlam)
        return obs_list[0].astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed_val = seed

        self.base_env.reset()

        for agent in self.partner_pool.values():
            agent.reset()
            agent.set_agent_index(1)
            agent.set_mdp(self.mdp)

        if self.mode == 'dynamic':
            self._current_partner_idx = np.random.randint(len(PARTNER_TYPES))
        else:
            self._current_partner_idx = 0

        self._steps_since_switch = 0
        self._step_count = 0

        return self._get_obs(), {}

    def step(self, action):
        agent_action = Action.INDEX_TO_ACTION[int(action)]
        other_action = self._get_partner_action()
        joint_action = (agent_action, other_action)

        next_state, reward, done, info = self.base_env.step(joint_action)

        self._step_count += 1
        self._steps_since_switch += 1

        if self.mode == 'dynamic':
            self._select_partner()

        truncated = self._step_count >= self.horizon
        if truncated:
            done = True

        info['partner_type'] = self.current_partner_type

        return self._get_obs(), reward, done, truncated, info

    def close(self):
        pass
