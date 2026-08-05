"""Process-wide random state, mirroring torch.manual_seed()."""

from __future__ import annotations

import random
import threading

_lock = threading.Lock()
_generator = random.Random()


def manual_seed(seed: int) -> int:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("hytorch.manual_seed: seed must be an integer")
    with _lock:
        _generator.seed(seed)
    return seed


def shuffled(values):
    result = list(values)
    with _lock:
        _generator.shuffle(result)
    return result
