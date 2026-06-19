"""Feature extractor for IRL — decision-relevant features for 5-10-K.

All features are:
  1) Mode-independent (computed identically in SINGLE/STATIC/DYNAMIC)
  2) Decision-relevant (inform the question "what should I play now?")
  3) Non-redundant (each captures a distinct strategic dimension)

Feature set:
  φ₁ MyScore      — my accumulated 510K score / 150        [How am I doing?]
  φ₂ MyHandSize   — my cards remaining / 13 (inverted)      [How close to done?]
  φ₃ MyStrength   — average rank of my cards [3→0, 2→1]    [How strong is my hand?]
  φ₄ TrickScore   — 510K points at stake in this trick /100 [What's the risk/reward?]
"""
import numpy as np
from .card import Rank
from .game import Game


FEATURE_NAMES = ['MyScore', 'MyHandSize', 'MyStrength', 'TrickScore']
FEATURE_DIM = len(FEATURE_NAMES)
FEATURE_BINS = 4


def _avg_rank_normalized(hand) -> float:
    """Average rank (3→0.0, 2→1.0). Jokers treated as 2."""
    if not hand:
        return 0.0
    total = 0
    for c in hand:
        r = c.rank.value
        if r >= 16:
            r = 15
        total += r
    avg = total / len(hand)
    return (avg - 3.0) / 12.0


def extract_features(game: Game, player_id: int) -> np.ndarray:
    hand = game.players[player_id].hand
    max_hand = 18 if game.num_players == 3 else 13

    f = np.zeros(FEATURE_DIM, dtype=np.float32)

    # φ₁: MyScore — how many 510K points I've accumulated
    f[0] = min(float(game.player_510k_scores[player_id]) / 150.0, 1.0)

    # φ₂: MyHandSize — inverted: 0 = 13 cards, 1 = empty
    f[1] = (max_hand - len(hand)) / max_hand

    # φ₃: MyStrength — average card rank
    f[2] = _avg_rank_normalized(hand)

    # φ₄: TrickScore — 510K points at stake right now
    f[3] = min(game.trick_pending_score / 100.0, 1.0)

    return f


def discretize_state(features: np.ndarray, bins: int = FEATURE_BINS) -> int:
    idx = 0
    for d in range(FEATURE_DIM):
        val = np.clip(features[d], 0.0, 1.0)
        bin_id = min(int(val * bins), bins - 1)
        idx += bin_id * (bins ** d)
    return idx


def state_to_features(state_idx: int, bins: int = FEATURE_BINS) -> np.ndarray:
    features = np.zeros(FEATURE_DIM, dtype=np.float32)
    remaining = state_idx
    for d in range(FEATURE_DIM):
        bin_id = remaining % bins
        remaining //= bins
        features[d] = (bin_id + 0.5) / bins
    return features
