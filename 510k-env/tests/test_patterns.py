import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from env.card import Card, Rank, Suit
from env.patterns import detect_pattern, can_beat, generate_all_patterns, get_valid_plays, PatternType


C = Card


class TestDetectPattern:
    def test_single(self):
        p = detect_pattern([C(Rank.THREE, Suit.SPADE)])
        assert p is not None
        assert p.type == PatternType.SINGLE
        assert p.main_rank == Rank.THREE

    def test_pair(self):
        p = detect_pattern([C(Rank.THREE, Suit.SPADE), C(Rank.THREE, Suit.HEART)])
        assert p.type == PatternType.PAIR
        assert p.main_rank == Rank.THREE

    def test_three(self):
        p = detect_pattern([C(Rank.FIVE, Suit.SPADE), C(Rank.FIVE, Suit.HEART), C(Rank.FIVE, Suit.CLUB)])
        assert p.type == PatternType.THREE
        assert p.main_rank == Rank.FIVE

    def test_three_one(self):
        p = detect_pattern([C(Rank.THREE, Suit.SPADE), C(Rank.THREE, Suit.HEART),
                           C(Rank.THREE, Suit.CLUB), C(Rank.FOUR, Suit.SPADE)])
        assert p.type == PatternType.THREE_ONE
        assert p.main_rank == Rank.THREE

    def test_three_pair(self):
        p = detect_pattern([C(Rank.THREE, Suit.SPADE), C(Rank.THREE, Suit.HEART),
                           C(Rank.THREE, Suit.CLUB), C(Rank.FOUR, Suit.SPADE),
                           C(Rank.FOUR, Suit.HEART)])
        assert p.type == PatternType.THREE_PAIR
        assert p.main_rank == Rank.THREE

    def test_straight_5(self):
        p = detect_pattern([C(Rank.THREE, Suit.SPADE), C(Rank.FOUR, Suit.HEART),
                           C(Rank.FIVE, Suit.CLUB), C(Rank.SIX, Suit.DIAMOND),
                           C(Rank.SEVEN, Suit.SPADE)])
        assert p.type == PatternType.STRAIGHT
        assert p.main_rank == Rank.SEVEN
        assert p.length == 5

    def test_straight_invalid_with_2(self):
        # Straight cannot contain 2
        p = detect_pattern([C(Rank.TEN, Suit.SPADE), C(Rank.JACK, Suit.HEART),
                           C(Rank.QUEEN, Suit.CLUB), C(Rank.KING, Suit.DIAMOND),
                           C(Rank.ACE, Suit.SPADE), C(Rank.TWO, Suit.HEART)])
        # 10 J Q K A is 5 straight, plus 2 makes it invalid
        # Since the ranks are 10, J, Q, K, A, 2 - these don't form a straight
        # because A(14) to 2(15) is consecutive but 2 is not allowed in straight
        assert p is None

    def test_bomb(self):
        p = detect_pattern([C(Rank.THREE, Suit.SPADE), C(Rank.THREE, Suit.HEART),
                           C(Rank.THREE, Suit.CLUB), C(Rank.THREE, Suit.DIAMOND)])
        assert p.type == PatternType.BOMB
        assert p.main_rank == Rank.THREE

    def test_suit_510k(self):
        p = detect_pattern([C(Rank.FIVE, Suit.SPADE), C(Rank.TEN, Suit.SPADE),
                           C(Rank.KING, Suit.SPADE)])
        assert p.type == PatternType.SUIT_510K

    def test_nonsuit_510k(self):
        p = detect_pattern([C(Rank.FIVE, Suit.SPADE), C(Rank.TEN, Suit.HEART),
                           C(Rank.KING, Suit.CLUB)])
        assert p.type == PatternType.NONSUIT_510K

    def test_pair_of_aces(self):
        p = detect_pattern([C(Rank.ACE, Suit.HEART), C(Rank.ACE, Suit.DIAMOND)])
        assert p.type == PatternType.PAIR
        assert p.main_rank == Rank.ACE

    def test_consecutive_pairs(self):
        p = detect_pattern([C(Rank.THREE, Suit.SPADE), C(Rank.THREE, Suit.HEART),
                           C(Rank.FOUR, Suit.SPADE), C(Rank.FOUR, Suit.HEART),
                           C(Rank.FIVE, Suit.SPADE), C(Rank.FIVE, Suit.HEART)])
        assert p.type == PatternType.CONSECUTIVE_PAIRS
        assert p.length == 3

    def test_airplane_none(self):
        p = detect_pattern([C(Rank.THREE, Suit.SPADE), C(Rank.THREE, Suit.HEART),
                           C(Rank.THREE, Suit.CLUB), C(Rank.FOUR, Suit.SPADE),
                           C(Rank.FOUR, Suit.HEART), C(Rank.FOUR, Suit.CLUB)])
        assert p.type == PatternType.AIRPLANE_NONE
        assert p.length == 2

    def test_airplane_singles(self):
        p = detect_pattern([C(Rank.THREE, Suit.SPADE), C(Rank.THREE, Suit.HEART),
                           C(Rank.THREE, Suit.CLUB), C(Rank.FOUR, Suit.SPADE),
                           C(Rank.FOUR, Suit.HEART), C(Rank.FOUR, Suit.CLUB),
                           C(Rank.FIVE, Suit.SPADE), C(Rank.SIX, Suit.HEART)])
        assert p.type == PatternType.AIRPLANE_SINGLE
        assert p.length == 2

    def test_airplane_pairs(self):
        p = detect_pattern([C(Rank.THREE, Suit.SPADE), C(Rank.THREE, Suit.HEART),
                           C(Rank.THREE, Suit.CLUB), C(Rank.FOUR, Suit.SPADE),
                           C(Rank.FOUR, Suit.HEART), C(Rank.FOUR, Suit.CLUB),
                           C(Rank.FIVE, Suit.SPADE), C(Rank.FIVE, Suit.HEART),
                           C(Rank.SIX, Suit.SPADE), C(Rank.SIX, Suit.HEART)])
        assert p.type == PatternType.AIRPLANE_PAIR
        assert p.length == 2

    def test_red_a_single(self):
        p = detect_pattern([C(Rank.ACE, Suit.HEART)], dynamic_mode=True)
        assert p.type == PatternType.RED_A_SINGLE

    def test_red_a_pair(self):
        p = detect_pattern([C(Rank.ACE, Suit.HEART), C(Rank.ACE, Suit.DIAMOND)],
                          dynamic_mode=True)
        assert p.type == PatternType.RED_A_PAIR

    def test_black_a_not_red(self):
        # Black A is not red A in normal circumstances
        p = detect_pattern([C(Rank.ACE, Suit.SPADE)])
        assert p.type == PatternType.SINGLE
        assert p.main_rank == Rank.ACE

    def test_invalid_empty(self):
        assert detect_pattern([]) is None

    def test_invalid_2cards_diff(self):
        assert detect_pattern([C(Rank.THREE, Suit.SPADE), C(Rank.FOUR, Suit.SPADE)]) is None


class TestCanBeat:
    def test_single_higher(self):
        s5 = detect_pattern([C(Rank.FIVE, Suit.SPADE)])
        s6 = detect_pattern([C(Rank.SIX, Suit.SPADE)])
        assert can_beat(s6, s5)
        assert not can_beat(s5, s6)

    def test_pair_higher(self):
        p3 = detect_pattern([C(Rank.THREE, Suit.SPADE), C(Rank.THREE, Suit.HEART)])
        p4 = detect_pattern([C(Rank.FOUR, Suit.SPADE), C(Rank.FOUR, Suit.HEART)])
        assert can_beat(p4, p3)

    def test_bomb_beats_single(self):
        bomb = detect_pattern([C(Rank.THREE, Suit.SPADE), C(Rank.THREE, Suit.HEART),
                              C(Rank.THREE, Suit.CLUB), C(Rank.THREE, Suit.DIAMOND)])
        single = detect_pattern([C(Rank.FIVE, Suit.SPADE)])
        assert can_beat(bomb, single)

    def test_suit510k_beats_bomb(self):
        s510k = detect_pattern([C(Rank.FIVE, Suit.SPADE), C(Rank.TEN, Suit.SPADE),
                               C(Rank.KING, Suit.SPADE)])
        bomb = detect_pattern([C(Rank.THREE, Suit.SPADE), C(Rank.THREE, Suit.HEART),
                              C(Rank.THREE, Suit.CLUB), C(Rank.THREE, Suit.DIAMOND)])
        assert can_beat(s510k, bomb)

    def test_bomb_beats_nonsuit510k(self):
        bomb = detect_pattern([C(Rank.THREE, Suit.SPADE), C(Rank.THREE, Suit.HEART),
                              C(Rank.THREE, Suit.CLUB), C(Rank.THREE, Suit.DIAMOND)])
        ns510k = detect_pattern([C(Rank.FIVE, Suit.SPADE), C(Rank.TEN, Suit.HEART),
                                C(Rank.KING, Suit.CLUB)])
        assert can_beat(bomb, ns510k)
        assert not can_beat(ns510k, bomb)

    def test_same_rank_bomb(self):
        bomb3 = detect_pattern([C(Rank.THREE, Suit.SPADE), C(Rank.THREE, Suit.HEART),
                               C(Rank.THREE, Suit.CLUB), C(Rank.THREE, Suit.DIAMOND)])
        bomb5 = detect_pattern([C(Rank.FIVE, Suit.SPADE), C(Rank.FIVE, Suit.HEART),
                               C(Rank.FIVE, Suit.CLUB), C(Rank.FIVE, Suit.DIAMOND)])
        assert can_beat(bomb5, bomb3)

    def test_same_type_regular(self):
        t3 = detect_pattern([C(Rank.THREE, Suit.SPADE), C(Rank.THREE, Suit.HEART),
                            C(Rank.THREE, Suit.CLUB)])
        t4 = detect_pattern([C(Rank.FOUR, Suit.SPADE), C(Rank.FOUR, Suit.HEART),
                            C(Rank.FOUR, Suit.CLUB)])
        assert can_beat(t4, t3)

    def test_straight_same_length(self):
        s34567 = detect_pattern([C(Rank.THREE, Suit.SPADE), C(Rank.FOUR, Suit.HEART),
                                C(Rank.FIVE, Suit.CLUB), C(Rank.SIX, Suit.DIAMOND),
                                C(Rank.SEVEN, Suit.SPADE)])
        s45678 = detect_pattern([C(Rank.FOUR, Suit.SPADE), C(Rank.FIVE, Suit.HEART),
                                C(Rank.SIX, Suit.CLUB), C(Rank.SEVEN, Suit.DIAMOND),
                                C(Rank.EIGHT, Suit.SPADE)])
        assert can_beat(s45678, s34567)
        assert not can_beat(s34567, s45678)

    def test_straight_diff_length(self):
        s34567 = detect_pattern([C(Rank.THREE, Suit.SPADE), C(Rank.FOUR, Suit.HEART),
                                C(Rank.FIVE, Suit.CLUB), C(Rank.SIX, Suit.DIAMOND),
                                C(Rank.SEVEN, Suit.SPADE)])
        s345678 = detect_pattern([C(Rank.THREE, Suit.SPADE), C(Rank.FOUR, Suit.HEART),
                                 C(Rank.FIVE, Suit.CLUB), C(Rank.SIX, Suit.DIAMOND),
                                 C(Rank.SEVEN, Suit.SPADE), C(Rank.EIGHT, Suit.HEART)])
        # Different lengths, cannot beat
        assert not can_beat(s345678, s34567)

    def test_red_a_single_beats_king(self):
        ra = detect_pattern([C(Rank.ACE, Suit.HEART)], dynamic_mode=True)
        king = detect_pattern([C(Rank.KING, Suit.SPADE)])
        assert can_beat(ra, king, dynamic_mode=True)

    def test_red_a_pair_beats_suit510k(self):
        ra_pair = detect_pattern([C(Rank.ACE, Suit.HEART), C(Rank.ACE, Suit.DIAMOND)],
                                dynamic_mode=True)
        s510k = detect_pattern([C(Rank.FIVE, Suit.SPADE), C(Rank.TEN, Suit.SPADE),
                               C(Rank.KING, Suit.SPADE)])
        assert can_beat(ra_pair, s510k, dynamic_mode=True)


class TestGenerateValidPlays:
    def test_get_valid_plays_leading(self):
        hand = [C(Rank.THREE, Suit.SPADE), C(Rank.FOUR, Suit.HEART),
                C(Rank.FIVE, Suit.CLUB)]
        plays = get_valid_plays(hand)
        assert len(plays) > 0

    def test_get_valid_plays_single(self):
        hand = [C(Rank.FIVE, Suit.SPADE), C(Rank.SIX, Suit.HEART),
                C(Rank.THREE, Suit.CLUB), C(Rank.THREE, Suit.SPADE)]
        last = detect_pattern([C(Rank.FOUR, Suit.SPADE)])
        plays = get_valid_plays(hand, last)
        assert any(p.type == PatternType.SINGLE and p.main_rank == Rank.FIVE for p in plays)
        assert any(p.type == PatternType.SINGLE and p.main_rank == Rank.SIX for p in plays)
        assert not any(p.type == PatternType.SINGLE and p.main_rank == Rank.THREE for p in plays)

    def test_pass_not_included(self):
        """Pass is not a pattern - it's handled at game level"""
        hand = [C(Rank.THREE, Suit.SPADE)]
        last = detect_pattern([C(Rank.KING, Suit.SPADE)])
        plays = get_valid_plays(hand, last)
        # No valid play (3 < K and no bomb)
        assert len(plays) == 0
