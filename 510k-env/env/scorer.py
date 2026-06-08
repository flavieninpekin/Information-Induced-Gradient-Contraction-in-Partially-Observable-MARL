from typing import Dict, List, Optional, Set
from .card import Card, Rank
from .game import Game, GameMode


class Scorer:
    def __init__(self, game: Game):
        self.game = game

    def compute_rewards(self) -> Dict[int, float]:
        mode = self.game.mode
        if mode == GameMode.SINGLE:
            rewards = self._score_single()
        elif mode == GameMode.STATIC:
            rewards = self._score_team(teams={0: 0, 1: 1, 2: 0, 3: 1})
        elif mode == GameMode.DYNAMIC:
            red_team = self._determine_red_a_team()
            if red_team is None:
                rewards = self._score_single()
            else:
                team_map = {i: (0 if i in red_team else 1) for i in range(self.game.num_players)}
                rewards = self._score_team(teams=team_map)
        else:
            rewards = {i: 0.0 for i in range(self.game.num_players)}
        # Add 510K scores on top of the base reward
        for i in range(self.game.num_players):
            rewards[i] += float(self.game.player_510k_scores[i])
        return rewards

    def _score_single(self) -> Dict[int, float]:
        rewards = {i: 0.0 for i in range(self.game.num_players)}
        if not self.game.finish_order:
            return rewards
        winner = self.game.finish_order[0]
        rewards[winner] = 10.0 * (self.game.num_players - 1)
        for i in range(self.game.num_players):
            if i != winner:
                rewards[i] = -10.0
        return rewards

    def _score_team(self, teams: Dict[int, int]) -> Dict[int, float]:
        rewards = {i: 0.0 for i in range(self.game.num_players)}
        if not self.game.finish_order:
            return rewards

        finished = set(self.game.finish_order)
        winning_team = None
        for team_id in (0, 1):
            members = [i for i in range(self.game.num_players) if teams.get(i) == team_id]
            if all(m in finished for m in members):
                winning_team = team_id
                break

        if winning_team is None:
            winning_team = teams.get(self.game.finish_order[0], 0)

        for i in range(self.game.num_players):
            if teams.get(i) == winning_team:
                rewards[i] = 15.0
            else:
                rewards[i] = -15.0
        return rewards

    def _determine_red_a_team(self) -> Optional[Set[int]]:
        if self.game.red_a_team is not None:
            return self.game.red_a_team
        red_team = set()
        for i, p in enumerate(self.game.players):
            for c in p.hand:
                if c.rank != Rank.ACE:
                    continue
                if c.is_red:
                    red_team.add(i)
                    break
                ace_count = sum(1 for cc in p.hand if cc.rank == Rank.ACE)
                if c.is_black and ace_count == 4:
                    red_team.add(i)
                    break
        return red_team if len(red_team) in (1, 2) else None
