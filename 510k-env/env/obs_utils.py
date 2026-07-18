"""Standalone observation and action-mask builders for any player."""
from typing import List, Optional
import numpy as np
from .card import Card, card_to_id
from .game import Game, GameMode

MAX_ACTIONS = 300


def _n_cards(game: Game) -> int:
    return 54 if game.include_jokers else 52


def obs_for_player(game: Game, player_id: int) -> np.ndarray:
    """Build observation vector from a specific player's perspective."""
    n = _n_cards(game)
    hand = np.zeros(n, dtype=np.float32)
    for c in game.players[player_id].hand:
        hand[card_to_id(c)] = 1.0

    last_play = np.zeros(n, dtype=np.float32)
    last_type = np.float32(0.0)
    if game.last_trick and game.last_trick.pattern:
        for c in game.last_trick.cards:
            last_play[card_to_id(c)] = 1.0
        last_type = np.float32(game.last_trick.pattern.type.value)

    hand_sizes = np.zeros(4, dtype=np.float32)
    for i, p in enumerate(game.players):
        hand_sizes[i] = len(p.hand)

    cp = np.float32(game.current_player)
    pc = np.float32(game.pass_count)
    score = np.float32(game.player_510k_scores[player_id])

    obs = np.concatenate([hand, last_play, [last_type], hand_sizes, [cp], [pc], [score]])
    if game.mode == GameMode.OBVIOUS:
        team_bits = np.zeros(4, dtype=np.float32)
        if game.red_a_team is not None:
            for i in range(4):
                if i != player_id and i in game.red_a_team:
                    team_bits[i] = 1.0
        obs = np.concatenate([obs, team_bits])
    return obs


def action_mask_for_player(game: Game, player_id: int) -> np.ndarray:
    """Build action mask for a specific player."""
    mask = np.zeros(MAX_ACTIONS, dtype=np.int64)
    if game.current_player != player_id:
        mask[0] = 1
        return mask
    valid = game.get_valid_actions(player_id)
    mask[0] = 1
    for i in range(len(valid)):
        if i + 1 < MAX_ACTIONS:
            mask[i + 1] = 1
    return mask


def execute_action(game: Game, player_id: int, action: int) -> bool:
    """Execute a trained-model action index in the game. Returns True if valid."""
    patterns = game.get_valid_actions(player_id)
    valid_card_sets = [p.cards for p in patterns]

    if action == 0 and game.can_pass(player_id):
        return game.pass_turn(player_id)
    idx = action - 1
    if 0 <= idx < len(valid_card_sets):
        return game.play_cards(player_id, valid_card_sets[idx])
    return False
