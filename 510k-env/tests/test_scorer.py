import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from env.game import Game, GameMode
from env.scorer import Scorer


def make_game(mode=GameMode.SINGLE):
    g = Game(mode=mode)
    g.finish_order = []
    return g


class TestScorerSingle:
    def test_winner_gets_30(self):
        game = make_game()
        game.finish_order = [0]
        game.player_510k_scores = [0, 0, 0, 0]
        rewards = Scorer(game).compute_rewards()
        assert rewards[0] == 30.0
        assert rewards[1] == -10.0
        assert rewards[2] == -10.0
        assert rewards[3] == -10.0

    def test_single_with_510k(self):
        game = make_game()
        game.finish_order = [2]
        game.player_510k_scores = [10, 5, 20, 3]
        rewards = Scorer(game).compute_rewards()
        # winner P2: +30 base + 20 510K = 50
        # others: -10 base + own 510K
        assert rewards[2] == 50.0
        assert rewards[0] == 0.0   # -10 + 10
        assert rewards[1] == -5.0  # -10 + 5


class TestScorerTeam:
    def test_user_example(self):
        """P0=25 P1=15 P2=20 P3=20, finish P0→P1→P3 (P2 unfinished)."""
        game = make_game(GameMode.STATIC)
        game.finish_order = [0, 1, 3]
        game.player_510k_scores = [25, 15, 20, 20]
        rewards = Scorer(game).compute_rewards()
        # Individual after bonuses: P0=40, P1=15, P2=5, P3=20
        # Team0=45, Team1=35, Team0 wins
        assert rewards[0] == 90.0   # 45 * 2
        assert rewards[2] == 90.0   # 45 * 2
        assert rewards[1] == 35.0   # 35 * 1
        assert rewards[3] == 35.0   # 35 * 1

    def test_team1_wins(self):
        """Team1 has higher total."""
        game = make_game(GameMode.STATIC)
        game.finish_order = [0, 1, 3, 2]
        game.player_510k_scores = [10, 40, 5, 30]
        # Indiv: P0=25(+15), P1=40, P2=-10(??), P3=30
        # Wait: finish all 4, so last = P2, first = P0
        # P0=10+15=25, P1=40, P3=30, P2=5-15=-10
        # Team0(P0+P2): 25+(-10)=15, Team1(P1+P3): 40+30=70
        # Team1 wins
        rewards = Scorer(game).compute_rewards()
        assert rewards[1] == 140.0  # 70 * 2
        assert rewards[3] == 140.0
        assert rewards[0] == 15.0   # 15 * 1
        assert rewards[2] == 15.0

    def test_tie_team0_wins(self):
        game = make_game(GameMode.STATIC)
        game.finish_order = [0, 2]
        game.player_510k_scores = [0, 0, 0, 0]
        rewards = Scorer(game).compute_rewards()
        # P0 first+15=15, P2 finished (2nd, not last), P1/P3 unfinished both -15
        # Indiv: P0=15, P2=0, P1=-15, P3=-15
        # Team0=15, Team1=-30, Team0 wins
        # Team0×2: P0=30, P2=30  Team1×1: P1=-30, P3=-30
        assert rewards[0] == 30.0
        assert rewards[2] == 30.0
        assert rewards[1] == -30.0
        assert rewards[3] == -30.0


class TestScorerDynamic:
    def test_dynamic_team_scoring(self):
        game = Game(mode=GameMode.DYNAMIC)
        red_team = game.red_a_team
        if red_team is None:
            pytest.skip("No red A team formed")
        # All players finish, red-team members first
        game.finish_order = list(red_team) + [i for i in range(4) if i not in red_team]
        game.player_510k_scores = [100 if i in red_team else 0 for i in range(4)]
        rewards = Scorer(game).compute_rewards()
        # Red team has 510K advantage → should produce positive reward for them
        red_rewards = [rewards[i] for i in red_team]
        non_red_rewards = [rewards[i] for i in range(4) if i not in red_team]
        assert sum(red_rewards) > sum(non_red_rewards)
