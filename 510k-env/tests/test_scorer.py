import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from env.card import Card, Rank, Suit
from env.game import Game, GameMode
from env.scorer import Scorer


def make_game_with_finish_order(mode: GameMode, finish_order):
    game = Game(mode=mode)
    game.finish_order = finish_order
    return game


class TestScorerSingle:
    def test_winner_gets_30(self):
        game = make_game_with_finish_order(GameMode.SINGLE, [0, 1, 2, 3])
        scorer = Scorer(game)
        rewards = scorer.compute_rewards()
        assert rewards[0] == 30.0
        assert rewards[1] == -10.0
        assert rewards[2] == -10.0
        assert rewards[3] == -10.0

    def test_winner_gets_30_p2(self):
        game = make_game_with_finish_order(GameMode.SINGLE, [2, 0, 1, 3])
        scorer = Scorer(game)
        rewards = scorer.compute_rewards()
        assert rewards[2] == 30.0
        assert rewards[0] == -10.0


class TestScorerStatic:
    def test_team0_wins_both_finished(self):
        game = make_game_with_finish_order(GameMode.STATIC, [0, 2])
        scorer = Scorer(game)
        rewards = scorer.compute_rewards()
        assert rewards[0] == 15.0
        assert rewards[2] == 15.0
        assert rewards[1] == -15.0
        assert rewards[3] == -15.0

    def test_team1_wins_both_finished(self):
        game = make_game_with_finish_order(GameMode.STATIC, [1, 3])
        scorer = Scorer(game)
        rewards = scorer.compute_rewards()
        assert rewards[1] == 15.0
        assert rewards[3] == 15.0
        assert rewards[0] == -15.0
        assert rewards[2] == -15.0


class TestScorerDynamic:
    def test_dynamic_produces_nonzero(self):
        game = Game(mode=GameMode.DYNAMIC)
        scorer = Scorer(game)
        game.finish_order = [0, 2]
        rewards = scorer.compute_rewards()
        # Each player's reward = team base (±30) + 510K score
        for i in range(4):
            assert rewards[i] != 0
