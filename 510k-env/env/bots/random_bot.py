import random
from typing import List, Optional

from env.patterns import Pattern


def random_bot_play(actions: List[Pattern], hand_size: int) -> Pattern:
    return random.choice(actions)
