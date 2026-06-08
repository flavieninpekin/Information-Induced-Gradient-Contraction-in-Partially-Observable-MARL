import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import random
from env.card import Card, Rank, Suit
from env.patterns import Pattern, PatternType, detect_pattern, get_valid_plays, can_beat
from env.game import Game, GameMode, TrickState


def random_bot_play(game: Game, player_idx: int) -> bool:
    """Simple random bot strategy."""
    actions = game.get_valid_actions(player_idx)
    if not actions:
        return game.pass_turn(player_idx)

    # Prioritize playing single cards when leading
    if game.last_trick is None:
        pattern = random.choice(actions)
    else:
        pattern = random.choice(actions)

    return game.play_cards(player_idx, pattern.cards)


def greedy_bot_play(game: Game, player_idx: int) -> bool:
    """Play the weakest valid action."""
    actions = game.get_valid_actions(player_idx)
    if not actions:
        return game.pass_turn(player_idx)

    # Sort by pattern type priority (weaker first) then main rank
    def sort_key(p: Pattern):
        type_order = {
            PatternType.SINGLE: 0, PatternType.PAIR: 1, PatternType.THREE: 2,
            PatternType.THREE_ONE: 3, PatternType.THREE_PAIR: 4,
            PatternType.STRAIGHT: 5, PatternType.CONSECUTIVE_PAIRS: 6,
            PatternType.AIRPLANE_NONE: 7, PatternType.AIRPLANE_SINGLE: 8,
            PatternType.AIRPLANE_PAIR: 9,
            PatternType.NONSUIT_510K: 10, PatternType.SUIT_510K: 11,
            PatternType.BOMB: 12, PatternType.JOKER_BOMB: 13,
            PatternType.RED_A_SINGLE: 14, PatternType.RED_A_PAIR: 15,
        }
        return (type_order.get(p.type, 99), p.main_rank)

    actions.sort(key=sort_key)
    chosen = actions[0]

    return game.play_cards(player_idx, chosen.cards)


def simulate_game(mode: GameMode = GameMode.SINGLE, seed: int = 42) -> Game:
    random.seed(seed)
    game = Game(mode=mode)
    max_rounds = 1000
    safety = 0

    while not game.is_over and safety < max_rounds:
        pid = game.current_player
        greedy_bot_play(game, pid)
        safety += 1

    return game


class TestGameInit:
    def test_game_creation(self):
        game = Game()
        assert game.num_players == 4
        assert game.started
        assert game.current_player >= 0

    def test_hand_sizes(self):
        game = Game()
        sizes = [len(p.hand) for p in game.players]
        assert sum(sizes) == 52
        assert all(s == 13 for s in sizes)

    def test_hand_sizes_3p(self):
        game = Game(num_players=3, include_jokers=True)
        sizes = [len(p.hand) for p in game.players]
        assert sum(sizes) == 54
        assert all(s == 18 for s in sizes)
        # Verify jokers exist
        all_cards = [c for p in game.players for c in p.hand]
        jokers = [c for c in all_cards if c.is_joker]
        assert len(jokers) == 2

    def test_starter_has_3d(self):
        game = Game()
        three_d = Card(Rank.THREE, Suit.DIAMOND)
        assert three_d in game.players[game.current_player].hand


class TestGamePlay:
    def test_single_player_game(self):
        game = simulate_game(GameMode.SINGLE, seed=42)
        assert len(game.finish_order) >= 1

    def test_static_game(self):
        game = simulate_game(GameMode.STATIC, seed=42)
        assert len(game.finish_order) >= 1

    def test_dynamic_game(self):
        game = simulate_game(GameMode.DYNAMIC, seed=43)
        assert len(game.finish_order) >= 1

    def test_3p_game(self):
        game = Game(num_players=3, include_jokers=True, mode=GameMode.SINGLE)
        max_rounds = 1000
        safety = 0
        while not game.is_over and safety < max_rounds:
            pid = game.current_player
            greedy_bot_play(game, pid)
            safety += 1
        assert len(game.finish_order) >= 1

    def test_valid_play(self):
        game = Game()
        pid = game.current_player
        hand = game.get_hand(pid)
        actions = game.get_valid_actions(pid)
        assert len(actions) > 0
        # Play the first action
        success = game.play_cards(pid, actions[0].cards)
        assert success
        # Hand should have fewer cards
        assert len(game.get_hand(pid)) < len(hand)

    def test_invalid_play_wrong_player(self):
        game = Game()
        wrong_pid = (game.current_player + 1) % 4
        hand = game.get_hand(wrong_pid)
        success = game.play_cards(wrong_pid, hand[:1])
        assert not success

    def test_pass_and_continue(self):
        game = Game()
        pid = game.current_player
        if game.can_pass(pid):
            # Should not be able to pass when leading
            assert False, "Leader should not be able to pass"

    def test_trick_rotation(self):
        game = Game()
        initial_pid = game.current_player
        # Play one card
        actions = game.get_valid_actions(initial_pid)
        game.play_cards(initial_pid, actions[0].cards)
        # Next player should be different
        assert game.current_player != initial_pid


class TestMultipleGames:
    def test_many_single_games(self):
        for seed in range(20):
            game = simulate_game(GameMode.SINGLE, seed=seed)
            assert len(game.finish_order) >= 1, f"Seed {seed} failed"

    def test_many_static_games(self):
        for seed in range(20):
            game = simulate_game(GameMode.STATIC, seed=seed)
            assert len(game.finish_order) >= 1, f"Seed {seed} failed"

    def test_many_dynamic_games(self):
        for seed in range(20):
            game = simulate_game(GameMode.DYNAMIC, seed=seed)
            assert len(game.finish_order) >= 1, f"Seed {seed} failed"
