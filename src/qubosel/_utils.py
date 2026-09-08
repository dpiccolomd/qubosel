"""Small shared helpers."""

from __future__ import annotations

import numpy as np

SEED_MAX = 2**32 - 1


def check_random_state(seed: int | np.random.Generator | None) -> np.random.Generator:
    """Return a :class:`numpy.random.Generator` for ``seed``.

    ``None`` gives fresh entropy, an ``int`` seeds a new generator, and an
    existing generator is returned unchanged.
    """
    if isinstance(seed, np.random.Generator):
        return seed
    if seed is None or isinstance(seed, (int, np.integer)):
        return np.random.default_rng(seed)
    raise TypeError(f"seed must be None, int or numpy Generator, got {type(seed)!r}")


def child_seeds(rng: np.random.Generator | int | None, n: int) -> list[int]:
    """Draw ``n`` independent integer seeds from ``rng`` in a fixed order."""
    gen = check_random_state(rng)
    return [int(s) for s in gen.integers(0, SEED_MAX, size=n, dtype=np.int64)]
