from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field
from enum import Enum

from .card import Card, Rank, Suit, deal_cards
from .patterns import Pattern, PatternType, detect_pattern, get_valid_plays, can_beat


class GameMode(Enum):
    SINGLE = 'single'
    STATIC = 'static'
    DYNAMIC = 'dynamic'
    OBVIOUS = 'obvious'  # DYNAMIC rules + known team info (ablation)


@dataclass
class PlayerState:
    hand: List[Card]
    finished: bool = False
    finish_order: int = -1


@dataclass
class TrickState:
    cards: List[Card] = field(default_factory=list)
    player: int = -1
    pattern: Optional[Pattern] = None


@dataclass
class Game:
    mode: GameMode = GameMode.SINGLE
    num_players: int = 4
    include_jokers: bool = False
    players: List[PlayerState] = field(init=False)
    current_player: int = 0
    last_trick: Optional[TrickState] = None
    pass_count: int = 0
    round_count: int = 0
    finish_order: List[int] = field(default_factory=list)
    leader: int = -1
    started: bool = False
    red_a_team: Optional[set] = None
    player_510k_scores: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    trick_pending_score: int = 0
    actions_log: List[dict] = field(default_factory=list)

    def __post_init__(self):
        self.reset()

    def reset(self):
        hands = deal_cards(self.num_players, include_jokers=self.include_jokers)
        self.players = [PlayerState(hand=h) for h in hands]
        self.current_player = self._find_starter()
        self.last_trick = None
        self.pass_count = 0
        self.round_count = 0
        self.finish_order = []
        self.leader = self.current_player
        self.started = True
        self.player_510k_scores = [0, 0, 0, 0]
        self.trick_pending_score = 0
        if self.mode in (GameMode.DYNAMIC, GameMode.OBVIOUS):
            self.red_a_team = self._compute_red_a_team()
        else:
            self.red_a_team = None

        self.actions_log = [dict(
            action='deal',
            hands=[list(h) for h in hands],
            starter=self.current_player,
            mode=self.mode.value,
            red_a_team=self.red_a_team,
        )]

    def _find_starter(self) -> int:
        three_diamond = Card(Rank.THREE, Suit.DIAMOND)
        for i, p in enumerate(self.players):
            if three_diamond in p.hand:
                return i
        return 0

    @property
    def is_over(self) -> bool:
        if self.mode in (GameMode.STATIC, GameMode.DYNAMIC, GameMode.OBVIOUS):
            # Check if any team has both players finished
            return len(self.finish_order) >= 2 and self._is_team_finished()
        return len(self.finish_order) >= 1

    def _is_team_finished(self) -> bool:
        if len(self.finish_order) < 2:
            return False
        if self.mode == GameMode.STATIC:
            finished_set = set(self.finish_order)
            if (0 in finished_set and 2 in finished_set) or (1 in finished_set and 3 in finished_set):
                return True
        elif self.mode in (GameMode.DYNAMIC, GameMode.OBVIOUS):
            if self.red_a_team is None:
                return False
            finished_set = set(self.finish_order)
            if self.red_a_team.issubset(finished_set):
                return True
            non_red_team = {i for i in range(self.num_players)} - self.red_a_team
            if non_red_team.issubset(finished_set):
                return True
        return False

    def _compute_red_a_team(self) -> Optional[set]:
        red_team = set()
        for i, p in enumerate(self.players):
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

    def _can_black_be_red(self) -> bool:
        # Check if any player has all 4 aces (allows black A to be treated as red A)
        for i, p in enumerate(self.players):
            if self._has_all_four_aces(i):
                return True
        return False

    def get_valid_actions(self, player_idx: int) -> List[Pattern]:
        player = self.players[player_idx]
        if player.finished:
            return []

        lt = self.last_trick.pattern if self.last_trick else None
        is_reda = self.mode in (GameMode.DYNAMIC, GameMode.OBVIOUS)
        if lt is None:
            return get_valid_plays(player.hand, None, is_reda)
        return get_valid_plays(player.hand, lt, is_reda)

    def can_pass(self, player_idx: int) -> bool:
        if self.players[player_idx].finished:
            return False
        if self.last_trick is None or self.last_trick.player == player_idx:
            return False
        if self._active_player_count() <= 1:
            return False
        return True

    def play_cards(self, player_idx: int, cards: List[Card]) -> bool:
        is_reda = self.mode in (GameMode.DYNAMIC, GameMode.OBVIOUS)
        if player_idx != self.current_player:
            return False
        player = self.players[player_idx]
        if player.finished:
            return False

        pattern = detect_pattern(cards, is_reda)
        if pattern is None:
            return False

        # Verify all cards are in hand
        if not all(c in player.hand for c in cards):
            return False

        # Verify beats last trick
        if self.last_trick is not None and self.last_trick.player != player_idx:
            lt = self.last_trick.pattern
            if not can_beat(pattern, lt, is_reda):
                return False

        # Execute play
        for c in cards:
            player.hand.remove(c)
        self.last_trick = TrickState(cards=cards, player=player_idx, pattern=pattern)
        self.pass_count = 0
        self.leader = player_idx

        # Accumulate 510K scores from played cards
        for c in cards:
            if c.rank == Rank.FIVE:
                self.trick_pending_score += 5
            elif c.rank == Rank.TEN:
                self.trick_pending_score += 10
            elif c.rank == Rank.KING:
                self.trick_pending_score += 10

        # Check if player finished
        finished_now = False
        if not player.hand:
            player.finished = True
            player.finish_order = len(self.finish_order)
            self.finish_order.append(player_idx)
            finished_now = True

        self.actions_log.append(dict(
            player=player_idx,
            action='play',
            cards=cards,
            pattern_type=pattern.type.name,
            hand_size=len(player.hand),
            finished=finished_now,
        ))

        self._advance_turn()
        return True

    def pass_turn(self, player_idx: int) -> bool:
        if player_idx != self.current_player:
            return False
        if not self.can_pass(player_idx):
            return False

        self.pass_count += 1

        self.actions_log.append(dict(
            player=player_idx,
            action='pass',
            cards=[],
            pattern_type='PASS',
            hand_size=len(self.players[player_idx].hand),
            finished=False,
        ))

        self._advance_turn()
        return True

    def _advance_turn(self):
        self.round_count += 1

        if self.is_over:
            return

        # Check if all remaining players passed → trick ends
        if self.pass_count >= self._active_player_count() - 1:
            self.player_510k_scores[self.leader] += self.trick_pending_score
            self.actions_log.append(dict(
                action='trick_end',
                winner=self.leader,
                score=self.trick_pending_score,
                cards=list(self.last_trick.cards) if self.last_trick else [],
                pattern_type=self.last_trick.pattern.type.name if self.last_trick else 'NONE',
            ))
            self.trick_pending_score = 0
            self.last_trick = None
            self.pass_count = 0
            self.current_player = self.leader
            if self.players[self.current_player].finished:
                self.current_player = self._next_active(self.current_player)
            return

        next_p = self._next_active(self.current_player)
        self.current_player = next_p

    def _next_active(self, from_player: int) -> int:
        p = (from_player + 1) % self.num_players
        safety = 0
        while self.players[p].finished:
            p = (p + 1) % self.num_players
            safety += 1
            if safety > self.num_players:
                return from_player  # All players finished
        return p

    def _active_player_count(self) -> int:
        return sum(1 for p in self.players if not p.finished)

    def get_hand(self, player_idx: int) -> List[Card]:
        return list(self.players[player_idx].hand)

    def get_hand_size(self, player_idx: int) -> int:
        return len(self.players[player_idx].hand)
