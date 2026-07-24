from enum import IntEnum
from dataclasses import dataclass
from typing import List, Optional, Dict, Set, Tuple
from itertools import combinations

from .card import Card, Rank, Suit


class PatternType(IntEnum):
    SINGLE = 1
    PAIR = 2
    THREE = 3
    THREE_ONE = 4
    THREE_PAIR = 5
    STRAIGHT = 6
    CONSECUTIVE_PAIRS = 7
    AIRPLANE_NONE = 8
    AIRPLANE_SINGLE = 9
    AIRPLANE_PAIR = 10
    BOMB = 11
    NONSUIT_510K = 12
    SUIT_510K = 13
    JOKER_BOMB = 14
    RED_A_SINGLE = 15
    RED_A_PAIR = 16


PATTERN_NAMES = {
    PatternType.SINGLE: '单张',
    PatternType.PAIR: '对子',
    PatternType.THREE: '三条',
    PatternType.THREE_ONE: '三带一',
    PatternType.THREE_PAIR: '三带一对',
    PatternType.STRAIGHT: '顺子',
    PatternType.CONSECUTIVE_PAIRS: '连对',
    PatternType.AIRPLANE_NONE: '飞机',
    PatternType.AIRPLANE_SINGLE: '飞机带单',
    PatternType.AIRPLANE_PAIR: '飞机带对',
    PatternType.BOMB: '炸弹',
    PatternType.NONSUIT_510K: '510K（异花）',
    PatternType.SUIT_510K: '510K（同花）',
    PatternType.JOKER_BOMB: '王炸',
    PatternType.RED_A_SINGLE: '红A单张',
    PatternType.RED_A_PAIR: '红A对',
}


@dataclass
class Pattern:
    type: PatternType
    main_rank: Rank
    cards: List[Card]
    length: int = 1

    def __post_init__(self):
        cards = list(self.cards)
        cards.sort(key=lambda c: (c.rank.value, c.suit.value))
        object.__setattr__(self, 'cards', cards)

    @property
    def name(self) -> str:
        return PATTERN_NAMES.get(self.type, '未知')

    def __str__(self) -> str:
        return f"[{self.name}] {' '.join(str(c) for c in self.cards)}"

    def __repr__(self) -> str:
        return str(self)


def _get_rank_counts(cards: List[Card]) -> Dict[Rank, int]:
    counts: Dict[Rank, int] = {}
    for c in cards:
        counts[c.rank] = counts.get(c.rank, 0) + 1
    return counts


def _is_consecutive(ranks: List[Rank]) -> bool:
    if len(ranks) < 2:
        return True
    values = [r.value for r in ranks]
    return all(values[i+1] - values[i] == 1 for i in range(len(values) - 1))


def detect_pattern(cards: List[Card], dynamic_mode: bool = False) -> Optional[Pattern]:
    if not cards:
        return None

    n = len(cards)
    sorted_cards = sorted(cards, key=lambda c: c.rank)
    rank_counts = _get_rank_counts(sorted_cards)

    # --- Joker bomb ---
    if n == 2:
        has_small = any(c.rank == Rank.SMALL_JOKER for c in sorted_cards)
        has_big = any(c.rank == Rank.BIG_JOKER for c in sorted_cards)
        if has_small and has_big:
            return Pattern(PatternType.JOKER_BOMB, Rank.BIG_JOKER, cards)

    # --- 510K ---
    if n == 3:
        ranks_found = sorted([c.rank for c in sorted_cards], key=lambda r: r.value)
        if len(ranks_found) == 3 and ranks_found == [Rank.FIVE, Rank.TEN, Rank.KING]:
            suits = [c.suit for c in sorted_cards]
            if len(set(suits)) == 1:
                return Pattern(PatternType.SUIT_510K, Rank.KING, cards)
            else:
                return Pattern(PatternType.NONSUIT_510K, Rank.KING, cards)

    # --- Bomb ---
    if n == 4:
        for rank, cnt in rank_counts.items():
            if cnt == 4:
                return Pattern(PatternType.BOMB, rank, cards)

    # --- Dynamic mode: Red A pair ---
    if dynamic_mode and n == 2:
        red_as = [c for c in sorted_cards if c.rank == Rank.ACE and c.is_red]
        if len(red_as) == 2:
            return Pattern(PatternType.RED_A_PAIR, Rank.ACE, cards)

    # --- Dynamic mode: Red A single ---
    if dynamic_mode and n == 1:
        card = sorted_cards[0]
        if card.rank == Rank.ACE and card.is_red:
            return Pattern(PatternType.RED_A_SINGLE, Rank.ACE, cards)

    # --- Single ---
    if n == 1:
        return Pattern(PatternType.SINGLE, sorted_cards[0].rank, cards)

    # --- Pair ---
    if n == 2 and len(rank_counts) == 1:
        rank = list(rank_counts.keys())[0]
        return Pattern(PatternType.PAIR, rank, cards)

    # --- Three ---
    if n == 3 and len(rank_counts) == 1:
        rank = list(rank_counts.keys())[0]
        return Pattern(PatternType.THREE, rank, cards)

    # --- Three + one ---
    if n == 4:
        counts = sorted(rank_counts.values())
        if counts == [1, 3]:
            main_rank = [r for r, cnt in rank_counts.items() if cnt == 3][0]
            return Pattern(PatternType.THREE_ONE, main_rank, cards)

    # --- Three + pair ---
    if n == 5:
        counts = sorted(rank_counts.values())
        if counts == [2, 3]:
            main_rank = [r for r, cnt in rank_counts.items() if cnt == 3][0]
            return Pattern(PatternType.THREE_PAIR, main_rank, cards)

    # --- Straight ---
    if n >= 5 and all(cnt == 1 for cnt in rank_counts.values()):
        ranks = sorted(rank_counts.keys(), key=lambda r: r.value)
        if all(Rank.THREE.value <= r.value <= Rank.ACE.value for r in ranks):
            if _is_consecutive(ranks):
                return Pattern(PatternType.STRAIGHT, max(ranks, key=lambda r: r.value), cards, length=n)

    # --- Consecutive pairs ---
    if n >= 6 and n % 2 == 0 and all(cnt == 2 for cnt in rank_counts.values()):
        pair_ranks = sorted(rank_counts.keys(), key=lambda r: r.value)
        if all(Rank.THREE.value <= r.value <= Rank.ACE.value for r in pair_ranks):
            if _is_consecutive(pair_ranks):
                return Pattern(PatternType.CONSECUTIVE_PAIRS, max(pair_ranks, key=lambda r: r.value), cards,
                               length=len(pair_ranks))

    # --- Airplane ---
    triple_ranks = sorted([r for r, cnt in rank_counts.items() if cnt >= 3], key=lambda r: r.value)
    if len(triple_ranks) >= 2:
        if all(Rank.THREE.value <= r.value <= Rank.ACE.value for r in triple_ranks):
            if _is_consecutive(triple_ranks):
                num_triples = len(triple_ranks)
                triple_cards_used = num_triples * 3
                kicker_count = n - triple_cards_used

                if kicker_count == 0:
                    return Pattern(PatternType.AIRPLANE_NONE, max(triple_ranks, key=lambda r: r.value), cards,
                                   length=num_triples)

                if kicker_count == num_triples:
                    return Pattern(PatternType.AIRPLANE_SINGLE, max(triple_ranks, key=lambda r: r.value), cards,
                                   length=num_triples)

                if kicker_count == num_triples * 2:
                    remaining_counts = dict(rank_counts)
                    for r in triple_ranks:
                        remaining_counts[r] -= 3
                    remaining_counts = {r: cnt for r, cnt in remaining_counts.items() if cnt > 0}
                    if all(cnt == 2 for cnt in remaining_counts.values()):
                        return Pattern(PatternType.AIRPLANE_PAIR, max(triple_ranks, key=lambda r: r.value), cards,
                                       length=num_triples)

    return None


# --- Hierarchy for bomb-like patterns ---
_BOMB_HIERARCHY = {
    PatternType.NONSUIT_510K: 0,
    PatternType.BOMB: 1,
    PatternType.SUIT_510K: 2,
    PatternType.JOKER_BOMB: 3,
    PatternType.RED_A_PAIR: 4,
}

_BOMB_LIKE = set(_BOMB_HIERARCHY.keys())

_REGULAR_TYPES = {
    PatternType.SINGLE, PatternType.PAIR, PatternType.THREE,
    PatternType.THREE_ONE, PatternType.THREE_PAIR,
    PatternType.STRAIGHT, PatternType.CONSECUTIVE_PAIRS,
    PatternType.AIRPLANE_NONE, PatternType.AIRPLANE_SINGLE, PatternType.AIRPLANE_PAIR,
    PatternType.RED_A_SINGLE,
}

_CONSEQ_TYPES = {
    PatternType.STRAIGHT, PatternType.CONSECUTIVE_PAIRS,
    PatternType.AIRPLANE_NONE, PatternType.AIRPLANE_SINGLE, PatternType.AIRPLANE_PAIR,
}


def can_beat(play: Pattern, last_play: Pattern, dynamic_mode: bool = False) -> bool:
    pt, lt = play.type, last_play.type

    # Dynamic mode: red A single beats any single
    if dynamic_mode and pt == PatternType.RED_A_SINGLE and lt == PatternType.SINGLE:
        return True

    # Dynamic mode: red A pair beats any bomb-like pattern
    if dynamic_mode and pt == PatternType.RED_A_PAIR:
        if lt in _BOMB_LIKE:
            return True
        if lt == PatternType.SINGLE:
            return play.main_rank > last_play.main_rank

    # Same regular type
    if pt == lt:
        if pt in _CONSEQ_TYPES:
            return play.length == last_play.length and play.main_rank > last_play.main_rank
        elif pt in {PatternType.SINGLE, PatternType.PAIR, PatternType.THREE,
                     PatternType.THREE_ONE, PatternType.THREE_PAIR}:
            return play.main_rank > last_play.main_rank
        elif pt == PatternType.BOMB:
            return play.main_rank > last_play.main_rank
        elif pt in {PatternType.NONSUIT_510K, PatternType.SUIT_510K, PatternType.JOKER_BOMB}:
            return False

    # Bomb-like beats regular
    if lt not in _BOMB_LIKE and pt in _BOMB_LIKE:
        return True

    # Bomb-like vs bomb-like (different types)
    if lt in _BOMB_LIKE and pt in _BOMB_LIKE:
        if dynamic_mode and pt == PatternType.RED_A_PAIR:
            return True
        if _BOMB_HIERARCHY[pt] > _BOMB_HIERARCHY[lt]:
            return True
        if _BOMB_HIERARCHY[pt] == _BOMB_HIERARCHY[lt] and pt == PatternType.BOMB and lt == PatternType.BOMB:
            return play.main_rank > last_play.main_rank

    return False


def _find_consecutive_sequences(rank_values: List[int], min_len: int = 2,
                                 max_val: int = Rank.ACE.value) -> List[List[int]]:
    if not rank_values:
        return []
    rank_values = sorted(set(rank_values))
    rank_values = [v for v in rank_values if v <= max_val]
    sequences = []
    i = 0
    while i < len(rank_values):
        j = i
        while j + 1 < len(rank_values) and rank_values[j+1] - rank_values[j] == 1:
            j += 1
        if j - i + 1 >= min_len:
            sequences.append(rank_values[i:j+1])
        i = j + 1
    return sequences


def generate_all_patterns(hand: List[Card], dynamic_mode: bool = False) -> List[Pattern]:
    patterns: List[Pattern] = []
    rank_counts = _get_rank_counts(hand)
    sorted_hand = sorted(hand, key=lambda c: c.rank)

    # Card groups by rank
    cards_by_rank: Dict[Rank, List[Card]] = {}
    for c in sorted_hand:
        cards_by_rank.setdefault(c.rank, []).append(c)

    # --- Singles ---
    if dynamic_mode:
        for c in sorted_hand:
            if c.rank == Rank.ACE and c.is_red:
                patterns.append(Pattern(PatternType.RED_A_SINGLE, Rank.ACE, [c]))
            else:
                patterns.append(Pattern(PatternType.SINGLE, c.rank, [c]))
    else:
        for c in sorted_hand:
            patterns.append(Pattern(PatternType.SINGLE, c.rank, [c]))

    # --- Pairs ---
    for rank, carlist in cards_by_rank.items():
        if len(carlist) >= 2:
            patterns.append(Pattern(PatternType.PAIR, rank, carlist[:2]))

    # --- Triples ---
    for rank, carlist in cards_by_rank.items():
        if len(carlist) >= 3:
            patterns.append(Pattern(PatternType.THREE, rank, carlist[:3]))

    # --- Three + one ---
    for main_rank, main_cards in cards_by_rank.items():
        if len(main_cards) >= 3:
            triple = main_cards[:3]
            for kick_rank, kick_cards in cards_by_rank.items():
                if kick_rank != main_rank:
                    patterns.append(Pattern(PatternType.THREE_ONE, main_rank, triple + list(kick_cards)[:1]))

    # --- Three + pair ---
    for main_rank, main_cards in cards_by_rank.items():
        if len(main_cards) >= 3:
            triple = main_cards[:3]
            for kick_rank, kick_cards in cards_by_rank.items():
                if kick_rank != main_rank and len(kick_cards) >= 2:
                    patterns.append(Pattern(PatternType.THREE_PAIR, main_rank, triple + list(kick_cards)[:2]))

    # --- Bombs ---
    for rank, carlist in cards_by_rank.items():
        if len(carlist) >= 4:
            patterns.append(Pattern(PatternType.BOMB, rank, carlist[:4]))

    # --- 510K (all combos) ---
    five_cards = cards_by_rank.get(Rank.FIVE, [])
    ten_cards = cards_by_rank.get(Rank.TEN, [])
    king_cards = cards_by_rank.get(Rank.KING, [])
    for f in five_cards:
        for t in ten_cards:
            for k in king_cards:
                suits = {f.suit, t.suit, k.suit}
                if len(suits) == 1:
                    patterns.append(Pattern(PatternType.SUIT_510K, Rank.KING, [f, t, k]))
                else:
                    patterns.append(Pattern(PatternType.NONSUIT_510K, Rank.KING, [f, t, k]))

    # --- Joker bomb ---
    small_jokers = cards_by_rank.get(Rank.SMALL_JOKER, [])
    big_jokers = cards_by_rank.get(Rank.BIG_JOKER, [])
    if small_jokers and big_jokers:
        patterns.append(Pattern(PatternType.JOKER_BOMB, Rank.BIG_JOKER, [small_jokers[0], big_jokers[0]]))

    # --- Dynamic mode: Red A pair ---
    if dynamic_mode:
        red_aces = [c for c in sorted_hand if c.rank == Rank.ACE and c.is_red]
        if len(red_aces) >= 2:
            patterns.append(Pattern(PatternType.RED_A_PAIR, Rank.ACE, red_aces[:2]))

    # --- Straights ---
    ranks_available = [r for r in Rank if r.value <= Rank.ACE.value and
                       r in rank_counts and rank_counts[r] >= 1]
    rank_values = sorted([r.value for r in ranks_available])
    for length in range(5, len(rank_values) + 1):
        for i in range(len(rank_values) - length + 1):
            seq = rank_values[i:i+length]
            if all(seq[j+1] - seq[j] == 1 for j in range(length - 1)):
                cards_used = []
                for rv in seq:
                    rank = Rank(rv)
                    cards_used.append(cards_by_rank[rank][0])
                patterns.append(Pattern(PatternType.STRAIGHT, Rank(seq[-1]), cards_used, length=length))

    # --- Consecutive pairs ---
    pair_ranks = [r for r in Rank if r.value <= Rank.ACE.value and
                  r in rank_counts and rank_counts[r] >= 2]
    pair_values = sorted([r.value for r in pair_ranks])
    for length in range(3, len(pair_values) + 1):
        for i in range(len(pair_values) - length + 1):
            seq = pair_values[i:i+length]
            if all(seq[j+1] - seq[j] == 1 for j in range(length - 1)):
                cards_used = []
                for rv in seq:
                    rank = Rank(rv)
                    cards_used.extend(cards_by_rank[rank][:2])
                patterns.append(Pattern(PatternType.CONSECUTIVE_PAIRS, Rank(seq[-1]),
                                        cards_used, length=length))

    # --- Airplanes (no kickers) ---
    triple_ranks = [r for r in Rank if r.value <= Rank.ACE.value and
                    r in rank_counts and rank_counts[r] >= 3]
    triple_values = sorted([r.value for r in triple_ranks])
    for length in range(2, len(triple_values) + 1):
        for i in range(len(triple_values) - length + 1):
            seq = triple_values[i:i+length]
            if all(seq[j+1] - seq[j] == 1 for j in range(length - 1)):
                cards_used = []
                for rv in seq:
                    rank = Rank(rv)
                    cards_used.extend(cards_by_rank[rank][:3])
                patterns.append(Pattern(PatternType.AIRPLANE_NONE, Rank(seq[-1]),
                                        cards_used, length=length))

    # --- Airplanes with single kickers ---
    for length in range(2, len(triple_values) + 1):
        for i in range(len(triple_values) - length + 1):
            seq = triple_values[i:i+length]
            if all(seq[j+1] - seq[j] == 1 for j in range(length - 1)):
                triple_cards = []
                for rv in seq:
                    rank = Rank(rv)
                    triple_cards.extend(cards_by_rank[rank][:3])
                # Find kickers: must be different from triple ranks, available singles
                used_ranks = set(seq)
                kicker_candidates = [c for c in sorted_hand if c.rank.value not in used_ranks
                                     and c not in triple_cards]
                # Also consider extra cards from ranks that have > 3 for the triple
                for rv in seq:
                    rank = Rank(rv)
                    if len(cards_by_rank[rank]) > 3:
                        for extra_c in cards_by_rank[rank][3:]:
                            if extra_c not in triple_cards:
                                kicker_candidates.append(extra_c)
                if len(kicker_candidates) >= length:
                    for combo in combinations(kicker_candidates, length):
                        patterns.append(Pattern(PatternType.AIRPLANE_SINGLE, Rank(seq[-1]),
                                                triple_cards + list(combo), length=length))

    # --- Airplanes with pair kickers ---
    all_pair_candidates = []
    for rank, carlist in cards_by_rank.items():
        if rank.value <= Rank.ACE.value and len(carlist) >= 2:
            all_pair_candidates.append(carlist[:2])
        if rank.value in triple_values and len(carlist) >= 5:
            all_pair_candidates.append(carlist[3:5])

    for length in range(2, len(triple_values) + 1):
        for i in range(len(triple_values) - length + 1):
            seq = triple_values[i:i+length]
            if all(seq[j+1] - seq[j] == 1 for j in range(length - 1)):
                triple_cards = []
                for rv in seq:
                    rank = Rank(rv)
                    triple_cards.extend(cards_by_rank[rank][:3])
                valid_pairs = []
                for p in all_pair_candidates:
                    if p[0] not in triple_cards and p[1] not in triple_cards:
                        valid_pairs.append(p)
                if len(valid_pairs) >= length:
                    for combo in combinations(valid_pairs, length):
                        flat_kickers = [c for pair in combo for c in pair]
                        patterns.append(Pattern(PatternType.AIRPLANE_PAIR, Rank(seq[-1]),
                                                triple_cards + flat_kickers, length=length))

    return patterns


def get_valid_plays(hand: List[Card], last_play: Optional[Pattern] = None,
                     dynamic_mode: bool = False) -> List[Pattern]:
    if not hand:
        return []

    all_patterns = generate_all_patterns(hand, dynamic_mode)

    if last_play is None:
        return all_patterns

    return [p for p in all_patterns if can_beat(p, last_play, dynamic_mode)]
