from typing import Dict, List, Optional, Tuple, Any
import random

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .card import Card, Rank, Suit, card_to_id
from .patterns import Pattern, PatternType, detect_pattern, get_valid_plays, can_beat
from .game import Game, GameMode
from .scorer import Scorer
from .obs_utils import obs_for_player, action_mask_for_player


MAX_ACTIONS = 300


class FiveTenKEnv(gym.Env):
    """510K Gymnasium environment.
    
    Single-agent: the agent is always player 0. Other players use a random bot.
    """
    
    metadata = {'render_modes': ['human', 'ansi']}

    def __init__(self, mode: str = 'single', num_players: int = 4,
                 render_mode: Optional[str] = None):
        super().__init__()
        self.mode = GameMode(mode) if mode != '3p' else GameMode.SINGLE
        self.num_players = num_players
        self.include_jokers = (num_players == 3)
        self.n_cards = 54 if self.include_jokers else 52
        self.agent_id = 0
        self.render_mode = render_mode

        obs_dim = self.n_cards * 2 + 1 + 4 + 1 + 1 + 1
        if self.mode == GameMode.OBVIOUS:
            obs_dim += 4  # teammate one-hot
        self.obs_dim = obs_dim
        self.action_space = spaces.Discrete(MAX_ACTIONS)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.game: Optional[Game] = None
        self._bot_fn = self._random_bot
        self._model_bot = None  # Optional[MaskablePPO] for self-play
        self.history: List[dict] = []

    def set_policy_bot(self, model):
        """Enable self-play: P1-P3 use the same policy as P0."""
        self._model_bot = model

    def _get_obs(self) -> np.ndarray:
        n = self.n_cards
        hand = np.zeros(n, dtype=np.float32)
        if self.game:
            for c in self.game.players[self.agent_id].hand:
                hand[card_to_id(c)] = 1.0

        last_play = np.zeros(n, dtype=np.float32)
        last_type = np.float32(0.0)
        if self.game and self.game.last_trick and self.game.last_trick.pattern:
            for c in self.game.last_trick.cards:
                last_play[card_to_id(c)] = 1.0
            last_type = np.float32(self.game.last_trick.pattern.type.value)

        hand_sizes = np.zeros(4, dtype=np.float32)
        if self.game:
            for i, p in enumerate(self.game.players):
                hand_sizes[i] = len(p.hand)

        cp = np.float32(self.game.current_player if self.game else 0)
        pc = np.float32(self.game.pass_count if self.game else 0)
        score = np.float32(self.game.player_510k_scores[self.agent_id] if self.game else 0.0)

        obs = np.concatenate([hand, last_play, [last_type], hand_sizes, [cp], [pc], [score]])
        if self.mode == GameMode.OBVIOUS:
            team_bits = np.zeros(4, dtype=np.float32)
            if self.game and self.game.red_a_team is not None:
                for i in range(4):
                    if i != self.agent_id and i in self.game.red_a_team:
                        team_bits[i] = 1.0
            obs = np.concatenate([obs, team_bits])
        return obs

    def _get_action_mask(self) -> np.ndarray:
        mask = np.zeros(MAX_ACTIONS, dtype=np.int64)
        if self.game is None or self.game.current_player != self.agent_id:
            mask[0] = 1  # Only pass available
            return mask

        valid = self.game.get_valid_actions(self.agent_id)
        mask[0] = 1  # Pass is always available (if allowed)
        for i, p in enumerate(valid):
            if i + 1 < MAX_ACTIONS:
                mask[i + 1] = 1
        return mask

    def _get_info(self) -> dict:
        return {'action_mask': self._get_action_mask()}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.game = Game(mode=self.mode, num_players=self.num_players,
                         include_jokers=self.include_jokers)

        # Auto-play until it's agent's turn
        while not self.game.is_over and self.game.current_player != self.agent_id:
            self._auto_play_next()

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self.game is None or self.game.is_over:
            return self.reset()[0], 0.0, True, False, {}

        pid = self.game.current_player
        assert pid == self.agent_id, f"Expected agent turn, got player {pid}"

        # Execute agent action (fallback to auto-play if invalid)
        patterns = self.game.get_valid_actions(pid)
        valid_card_sets = [p.cards for p in patterns]
        action_taken = False

        if action == 0 and self.game.can_pass(pid):
            action_taken = self.game.pass_turn(pid)
        else:
            idx = action - 1
            if 0 <= idx < len(valid_card_sets):
                action_taken = self.game.play_cards(pid, valid_card_sets[idx])

        if not action_taken:
            if valid_card_sets:
                chosen = random.choice(patterns)
                self.game.play_cards(pid, chosen.cards)
            elif self.game.can_pass(pid):
                self.game.pass_turn(pid)

        # Auto-play opponents until it's agent's turn again
        while not self.game.is_over and self.game.current_player != self.agent_id:
            self._auto_play_next()

        done = self.game.is_over
        reward = self._compute_reward() if done else 0.0

        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, done, False, info

    def _auto_play_next(self):
        pid = self.game.current_player
        actions = self.game.get_valid_actions(pid)
        if not actions:
            self.game.pass_turn(pid)
            return
        if self._model_bot is not None and pid != self.agent_id:
            chosen = self._policy_bot_act(pid, actions)
        else:
            chosen = self._bot_fn(pid, actions)
        self.game.play_cards(pid, chosen.cards)

    def _policy_bot_act(self, player_idx: int, actions: List[Pattern]) -> Pattern:
        obs = obs_for_player(self.game, player_idx)
        # If the model expects Dict obs (e.g., MAPPO), build it
        from gymnasium import spaces as gspaces
        if isinstance(self._model_bot.observation_space, gspaces.Dict):
            global_obs = np.concatenate([
                obs_for_player(self.game, i) for i in range(self.game.num_players)
            ])
            obs = {'local': obs.astype(np.float32), 'global': global_obs.astype(np.float32)}
        mask = action_mask_for_player(self.game, player_idx)
        try:
            action, _ = self._model_bot.predict(obs, action_masks=mask, deterministic=False)
        except Exception:
            return random.choice(actions)
        idx = int(action) - 1
        if 0 <= idx < len(actions):
            return actions[idx]
        return random.choice(actions)

    def _random_bot(self, player_idx: int, actions: List[Pattern]) -> Pattern:
        return random.choice(actions)

    def _compute_reward(self) -> float:
        if not self.game:
            return 0.0
        scorer = Scorer(self.game)
        rewards = scorer.compute_rewards()
        return rewards.get(self.agent_id, 0.0)

    def render(self):
        if self.render_mode == 'human' or self.render_mode == 'ansi':
            if not self.game:
                return "Game not started"
            lines = []
            for i, p in enumerate(self.game.players):
                marker = " <<<" if i == self.game.current_player else ""
                status = " FINISHED" if p.finished else ""
                lines.append(f"P{i}: {len(p.hand)} cards{marker}{status}")
            if self.game.last_trick:
                cards = ' '.join(str(c) for c in self.game.last_trick.cards)
                lines.append(f"Last play (P{self.game.last_trick.player}): {cards}")
            lines.append(f"Pass count: {self.game.pass_count}")
            return '\n'.join(lines)
        return ''
