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
            for i in range(self.game.num_players):
                rewards[i] += float(self.game.player_510k_scores[i])
        elif mode == GameMode.STATIC:
            rewards = self._score_team(teams={0: 0, 1: 1, 2: 0, 3: 1})
        elif mode in (GameMode.DYNAMIC, GameMode.OBVIOUS):
            red_team = self._determine_red_a_team()
            if red_team is None:
                rewards = self._score_single()
                for i in range(self.game.num_players):
                    rewards[i] += float(self.game.player_510k_scores[i])
            else:
                team_map = {i: (0 if i in red_team else 1) for i in range(self.game.num_players)}
                rewards = self._score_team(teams=team_map)
        else:
            rewards = {i: 0.0 for i in range(self.game.num_players)}
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
        n = self.game.num_players
        # 1) Individual score = 510K points
        individual = [float(self.game.player_510k_scores[i]) for i in range(n)]

        # 2) Finish-position bonuses
        if self.game.finish_order:
            first = self.game.finish_order[0]
            individual[first] += 15.0
            # Last = last finisher, or any unfinished player
            finished_set = set(self.game.finish_order)
            unfinished = [i for i in range(n) if i not in finished_set]
            if unfinished:
                for i in unfinished:
                    individual[i] -= 15.0
            elif len(self.game.finish_order) > 1:
                individual[self.game.finish_order[-1]] -= 15.0

        # 3) Sum per team
        team_total = {0: 0.0, 1: 0.0}
        for i in range(n):
            tid = teams.get(i)
            if tid is not None:
                team_total[tid] += individual[i]

        # 4) Winning team (higher total)
        winning_team = 0 if team_total[0] >= team_total[1] else 1

        # 5) Reward = team_total × multiplier (×2 for winners, ×1 for losers)
        mult = {winning_team: 2, 1 - winning_team: 1}
        rewards = {}
        for i in range(n):
            tid = teams.get(i)
            rewards[i] = team_total[tid] * mult[tid]

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
