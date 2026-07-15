"""Feature extractor for IRL — 7 features (5 individual + 2 cooperative/interaction).

  φ₁-φ₅: Individual features (unchanged)
  φ₆: ScoreSpread    — score inequality across players  [cooperative potential]
  φ₇: SuppressionGap — rank margin when following leader [competition intensity]
"""
import numpy as np
from .card import Rank
from .game import Game


FEATURE_NAMES = ['MyScore', 'MyHandSize', 'MyStrength',
                 'TrickScore', 'PassCount', 'ScoreSpread', 'SuppressionGap']
FEATURE_DIM = len(FEATURE_NAMES)
FEATURE_BINS = 3


def _avg_rank_normalized(hand) -> float:
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
    n = game.num_players
    max_hand = 13 if n == 4 else 18

    f = np.zeros(FEATURE_DIM, dtype=np.float32)

    # φ₁: MyScore
    f[0] = min(float(game.player_510k_scores[player_id]) / 150.0, 1.0)

    # φ₂: MyHandSize
    f[1] = (max_hand - len(hand)) / max_hand

    # φ₃: MyStrength
    f[2] = _avg_rank_normalized(hand)

    # φ₄: TrickScore
    f[3] = min(game.trick_pending_score / 100.0, 1.0)

    # φ₅: PassCount
    active = game._active_player_count()
    f[4] = game.pass_count / max(active - 1, 1)

    # φ₆: ScoreSpread — standard deviation of 510K scores / 50
    scores = [game.player_510k_scores[i] for i in range(n)]
    f[5] = min(np.std(scores) / 50.0, 1.0)

    # φ₇: SuppressionGap — when NOT leading, rank gap between my best card
    # and the leader's best card (normalized). 0 = same rank, 1 = max gap.
    if game.last_trick is not None and game.last_trick.pattern is not None \
            and game.last_trick.player != player_id:
        leader_best = max(c.rank.value for c in game.last_trick.cards)
        my_best = max((c.rank.value for c in hand), default=leader_best)
        gap = max(my_best - leader_best, 0)
        f[6] = gap / 12.0  # max rank gap is 12 (3 to 15)
    # else: leader or no trick → gap = 0

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
