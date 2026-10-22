"""Counter-based random generator shared with the R package (ssdiff).

This module mirrors a copy in the R package (ssdiff); the two must change
together.

Stream definition: value ``i`` (``i = 0, 1, 2, ...``) is splitmix64's output
``mix(seed + (i + 1) * 0x9E3779B97F4A7C15 mod 2^64)``, i.e. exactly the
sequence of the standard sequential splitmix64 (Steele, Lea & Flood 2014,
"Fast splittable pseudorandom number generators", OOPSLA) started at
``state = seed``. Because value ``i`` depends only on ``(seed, i)``, draws
made in chunks are identical to one large draw.
"""

from __future__ import annotations

import operator

import numpy as np

_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_MAX_SEED = 2**53  # R passes the seed as a double, exact below 2**53


def _check_seed(seed) -> int:
    if isinstance(seed, bool):
        raise ValueError(f"seed must be an int, got {type(seed).__name__}")
    try:
        seed = operator.index(seed)
    except TypeError:
        raise ValueError(f"seed must be an int, got {type(seed).__name__}") from None
    if not 0 <= seed < _MAX_SEED:
        raise ValueError(f"seed must satisfy 0 <= seed < 2**53, got {seed}")
    return seed


def raw_at(seed: int, start: int, n: int) -> np.ndarray:
    """Raw 64-bit outputs at counter positions ``start .. start + n - 1``.

    Parameters
    ----------
    seed : int
        Seed, ``0 <= seed < 2**53``.
    start : int
        First counter position (``>= 0``).
    n : int
        Number of values (``>= 0``).

    Returns
    -------
    ndarray of shape (n,), dtype uint64
        ``mix(seed + (i + 1) * GAMMA mod 2^64)`` for each position ``i``.
    """
    seed = _check_seed(seed)
    if start < 0 or n < 0:
        raise ValueError("start and n must be non-negative")
    i = np.arange(start + 1, start + n + 1, dtype=np.uint64)
    # uint64 array arithmetic wraps mod 2^64, which is what splitmix64 needs.
    with np.errstate(over="ignore"):
        z = np.uint64(seed) + i * _GAMMA
        z = (z ^ (z >> np.uint64(30))) * _M1
        z = (z ^ (z >> np.uint64(27))) * _M2
        return z ^ (z >> np.uint64(31))


def uniform_at(seed: int, start: int, n: int) -> np.ndarray:
    """Uniform doubles in [0, 1) at counter positions ``start .. start + n - 1``.

    Each value is ``(raw >> 11) * 2**-53``.

    Parameters
    ----------
    seed : int
        Seed, ``0 <= seed < 2**53``.
    start : int
        First counter position (``>= 0``).
    n : int
        Number of values (``>= 0``).

    Returns
    -------
    ndarray of shape (n,), dtype float64
    """
    return (raw_at(seed, start, n) >> np.uint64(11)).astype(np.float64) * 2.0**-53


class Rng:
    """Counter-based generator; draws advance an internal counter.

    Parameters
    ----------
    seed : int
        Seed, ``0 <= seed < 2**53``. A Python or NumPy int; ``bool`` is rejected.
    """

    def __init__(self, seed: int) -> None:
        self._seed = _check_seed(seed)
        self._counter = 0

    def uniform(self, n: int) -> np.ndarray:
        """Draw *n* uniforms in [0, 1) and advance the counter by *n*."""
        out = uniform_at(self._seed, self._counter, n)
        self._counter += n
        return out

    def integers(self, n: int) -> int:
        """One integer in ``[0, n)``: ``floor(u * n)`` for a single uniform ``u``."""
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        return int(np.floor(self.uniform(1)[0] * n))

    def permutation(self, n: int) -> np.ndarray:
        """Random permutation of ``0 .. n - 1``.

        Defined as the stable argsort of *n* uniforms. R parity: ``order()``
        is stable, so ``order(u)`` gives the same permutation.
        """
        return np.argsort(self.uniform(n), kind="stable").astype(np.intp)

    def choice(self, weights, size: int) -> np.ndarray:
        """Weighted draw with replacement of *size* indices into *weights*.

        Parameters
        ----------
        weights : array-like of shape (k,)
            Non-negative weights with a positive sum.
        size : int
            Number of draws.

        Returns
        -------
        ndarray of shape (size,), dtype intp
            ``searchsorted(cumsum(weights), u * total, side="right")`` for
            uniforms ``u``. R parity: ``np.cumsum`` and R ``cumsum`` both sum
            left to right, and ``findInterval(x, c)`` equals
            ``searchsorted(c, x, side="right")``.
        """
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim != 1 or w.size == 0 or not np.all(w >= 0) or not w.sum() > 0:
            raise ValueError("weights must be a non-empty 1-D array, all >= 0, with positive sum")
        c = np.cumsum(w)
        u = self.uniform(size)
        idx = np.searchsorted(c, u * c[-1], side="right")
        return np.minimum(idx, len(w) - 1).astype(np.intp)
