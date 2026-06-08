from enum import IntEnum
from dataclasses import dataclass
from typing import List, Optional
import random


class Suit(IntEnum):
    SPADE = 0
    HEART = 1
    CLUB = 2
    DIAMOND = 3

    def __str__(self):
        return ['♠', '♥', '♣', '♦'][self.value]

    def __repr__(self):
        return str(self)


class Rank(IntEnum):
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14
    TWO = 15
    SMALL_JOKER = 16
    BIG_JOKER = 17

    def __str__(self):
        return {3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
                10: '10', 11: 'J', 12: 'Q', 13: 'K', 14: 'A', 15: '2',
                16: '小王', 17: '大王'}[self.value]

    def __repr__(self):
        return str(self)


@dataclass(frozen=True, order=True)
class Card:
    rank: Rank
    suit: Suit

    def __str__(self):
        if self.rank in (Rank.SMALL_JOKER, Rank.BIG_JOKER):
            return str(self.rank)
        return f"{self.rank}{self.suit}"

    def __repr__(self):
        return str(self)

    @property
    def is_red(self) -> bool:
        return self.suit in (Suit.HEART, Suit.DIAMOND)

    @property
    def is_black(self) -> bool:
        return self.suit in (Suit.SPADE, Suit.CLUB)

    @property
    def is_joker(self) -> bool:
        return self.rank in (Rank.SMALL_JOKER, Rank.BIG_JOKER)


def create_deck(include_jokers: bool = False) -> List[Card]:
    cards = []
    for rank in Rank:
        if rank == Rank.SMALL_JOKER:
            if include_jokers:
                cards.append(Card(rank, Suit.SPADE))
        elif rank == Rank.BIG_JOKER:
            if include_jokers:
                cards.append(Card(rank, Suit.HEART))
        else:
            for suit in Suit:
                cards.append(Card(rank, suit))
    return cards


def deal_cards(num_players: int = 4, include_jokers: bool = False) -> List[List[Card]]:
    deck = create_deck(include_jokers)
    random.shuffle(deck)
    hands = [[] for _ in range(num_players)]
    for i, card in enumerate(deck):
        hands[i % num_players].append(card)
    for hand in hands:
        hand.sort()
    return hands


def card_to_id(card: Card) -> int:
    if card.rank == Rank.SMALL_JOKER:
        return 52
    if card.rank == Rank.BIG_JOKER:
        return 53
    rank_idx = card.rank.value - 3
    suit_idx = card.suit.value
    return rank_idx * 4 + suit_idx


def id_to_card(card_id: int) -> Card:
    if card_id == 52:
        return Card(Rank.SMALL_JOKER, Suit.SPADE)
    if card_id == 53:
        return Card(Rank.BIG_JOKER, Suit.HEART)
    rank_idx = card_id // 4
    suit_idx = card_id % 4
    return Card(Rank(rank_idx + 3), Suit(suit_idx))


def cards_to_ids(cards: List[Card]) -> List[int]:
    return [card_to_id(c) for c in cards]


def ids_to_cards(ids: List[int]) -> List[Card]:
    return [id_to_card(i) for i in ids]
