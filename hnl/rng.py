"""Deterministic random-number utilities.

All stochastic components of the package consume ``numpy.random.Generator``
objects created here.  ``SeedSequence.spawn`` guarantees that sub-streams are
independent and that the mapping ``seed -> streams`` is version-stable.
"""

from __future__ import annotations

from typing import List

import numpy as np


def make_rng(seed: int, stream: int = 0) -> np.random.Generator:
    """Return a generator for ``(seed, stream)``.

    The spawn key keeps streams reproducible even if the caller adds new
    streams later.
    """
    ss = np.random.SeedSequence(seed, spawn_key=(int(stream),))
    return np.random.default_rng(ss)


def spawn_rngs(seed: int, count: int) -> List[np.random.Generator]:
    """Return ``count`` independent generators derived from ``seed``."""
    if count < 0:
        raise ValueError("count must be non-negative")
    return [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(count)]


def seeds_for(seed: int, count: int) -> List[int]:
    """Deterministic integer seeds, useful for tabular manifests."""
    return [
        int(np.random.SeedSequence(seed, spawn_key=(i,)).generate_state(1)[0])
        for i in range(count)
    ]
