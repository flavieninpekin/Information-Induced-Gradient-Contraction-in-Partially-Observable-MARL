"""
Overcooked Gymnasium wrapper with hidden partner type.

SINGLE mode: partner uses one fixed strategy throughout all episodes.
DYNAMIC mode: partner's strategy switches mid-episode at random intervals.
STATIC mode: partner strategy fixed within episode but can vary across episodes.

The hidden "relationship" is the partner's behavioural type:
  - greedy: heuristic human-model agent (somewhat cooperative)
  - random: entirely random actions (uncooperative)
  - noop: stays in place (minimal)
"""
import numpy as np
import gymnasium as gym

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import RandomAgent, StayAgent, GreedyHumanModel, AgentFromPolicy, AgentPair
from overcooked_ai_py.mdp.actions import Action

PARTNER_TYPES = ['greedy', 'random', 'noop']
DEFAULT_LAYOUT = 'cramped_room'


class OvercookedHiddenPartner(gym.Env):
    """
    Single-agent view of Overcooked where the partner's identity is hidden.

    The controlled agent is always P0. The partner (P1) is controlled by
    a predefined agent whose type may be hidden and may switch during play.
    """

    metadata = {'render_modes': []}

    def __init__(self,
                 layout_name: str = DEFAULT_LAYOUT,
                 mode: str = 'single',
                 horizon: int = 400,
                 partner_types: list | None = None,
                 switch_interval: int = 50,
                 seed: int | None = None):
        """
        Args:
            layout_name: Overcooked layout name (e.g. 'cramped_room', 'coordination_ring')
            mode: 'single', 'static', or 'dynamic'
            horizon: max steps per episode
            partner_types: list of partner type strings for DYNAMIC mode (default: ['greedy', 'random'])
            switch_interval: steps between partner switches in DYNAMIC mode
        """
        super().__init__()

        self.layout_name = layout_name
        self.mode = mode
        self.horizon = horizon
        self.seed_val = seed

        if partner_types is None:
            partner_types = ['greedy', 'random']
        self.partner_type_names = partner_types
        self.switch_interval = switch_interval

        self._build_mdp_and_env()
        self._build_partner_agents()

        # 6 actions: N, S, E, W, STAY, INTERACT
        self.action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))
        # Observation: featurized state for P0
        dummy_obs = self._get_obs()
        self.observation_space = gym.spaces.Box(
            low=0, high=5, shape=dummy_obs.shape, dtype=np.float32
        )

        # Current partner info (hidden from agent)
        self._current_partner_idx = 0
        self._partner_agent = None
        self._steps_since_switch = 0
        self._step_count = 0

    def _build_mdp_and_env(self):
        self.mdp = OvercookedGridworld.from_layout_name(self.layout_name)
        self.base_env = OvercookedEnv.from_mdp(self.mdp, horizon=self.horizon)

    def _build_partner_agents(self):
        """Create partner agent instances for each type."""
        self.partner_pool = {}
        for ptype in self.partner_type_names:
            if ptype not in PARTNER_TYPES:
                raise ValueError(f"Unknown partner type: {ptype}. Choose from {PARTNER_TYPES}")
            self.partner_pool[ptype] = self._make_agent(ptype)

    def _make_agent(self, ptype: str):
        if ptype == 'random':
            return RandomAgent(all_actions=True)
        elif ptype == 'noop':
            return StayAgent()
        elif ptype == 'greedy':
            return GreedyHumanModel(self.base_env.mlam)
        else:
            raise ValueError(f"Unknown partner type: {ptype}")

    def _select_partner(self):
        """Pick the partner agent for the current timestep."""
        if self.mode == 'single':
            # Always use the first partner type
            self._current_partner_idx = 0
        elif self.mode == 'static':
            # Pick once per episode (already done in reset)
            pass
        elif self.mode == 'dynamic':
            if self._steps_since_switch >= self.switch_interval:
                self._current_partner_idx = 1 - self._current_partner_idx
                self._steps_since_switch = 0
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    @property
    def current_partner_type(self):
        return self.partner_type_names[self._current_partner_idx]

    def _get_partner_action(self):
        """Get action from current partner agent for P1."""
        p1_agent = self.partner_pool[self.current_partner_type]
        action, _ = p1_agent.action(self.base_env.state)
        return action

    def _get_obs(self):
        obs_list = self.mdp.featurize_state(self.base_env.state, self.base_env.mlam)
        return obs_list[0].astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed_val = seed

        self.base_env.reset()

        # Re-init all partner agents for the new episode
        for ptype, agent in self.partner_pool.items():
            agent.reset()
            agent.set_agent_index(1)
            agent.set_mdp(self.mdp)

        if self.mode == 'static':
            self._current_partner_idx = np.random.randint(len(self.partner_type_names))

        self._select_partner()
        self._steps_since_switch = 0
        self._step_count = 0

        return self._get_obs(), {}

    def step(self, action):
        # Convert integer action to Overcooked Action enum
        agent_action = Action.INDEX_TO_ACTION[int(action)]

        # Get partner action (P1)
        other_action = self._get_partner_action()

        # Build joint action: P0 = our agent, P1 = partner
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
        info['partner_switch_count'] = (
            self._step_count // self.switch_interval if self.mode == 'dynamic' else 0
        )

        return self._get_obs(), reward, done, truncated, info

    def close(self):
        pass
