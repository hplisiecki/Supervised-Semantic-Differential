"""Tests for ssdiff/utils/rng.py (stream is shared with the R package, ssdiff)."""

from __future__ import annotations

import numpy as np
import pytest

from ssdiff.utils.rng import Rng, raw_at, uniform_at

_M = 2**64 - 1


def _splitmix64(seed: int, n: int) -> list[int]:
    """Plain sequential splitmix64 with Python ints."""
    state = seed
    out = []
    for _ in range(n):
        state = (state + 0x9E3779B97F4A7C15) & _M
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M
        out.append(z ^ (z >> 31))
    return out


RAW_VECTORS = {
    0: [16294208416658607535, 7960286522194355700, 487617019471545679,
        17909611376780542444, 1961750202426094747],
    2137: [17522378519535027745, 364756094709023243, 8236747292942766741,
           8881960951598843736, 8369509784638573016],
    1234567: [6457827717110365317, 3203168211198807973, 9817491932198370423,
              4593380528125082431, 16408922859458223821],
}
UNIFORM_2137 = [0.9498900428996606, 0.019773467515542542, 0.44651496546113223,
                0.48149206798274413, 0.45371203455719133]


@pytest.mark.parametrize("seed", [0, 2137, 1234567, 2**53 - 1])
def test_raw_matches_sequential_splitmix64(seed):
    assert raw_at(seed, 0, 50).tolist() == _splitmix64(seed, 50)


@pytest.mark.parametrize("seed", list(RAW_VECTORS))
def test_raw_hardcoded_vectors(seed):
    assert raw_at(seed, 0, 5).tolist() == RAW_VECTORS[seed]
    assert _splitmix64(seed, 5) == RAW_VECTORS[seed]


def test_uniform_hardcoded_vector():
    assert uniform_at(2137, 0, 5).tolist() == UNIFORM_2137


def test_chunked_uniform_at_equals_single_call():
    whole = uniform_at(99, 0, 1000)
    parts = np.concatenate([uniform_at(99, s, 250) for s in range(0, 1000, 250)])
    np.testing.assert_array_equal(whole, parts)
    assert whole.min() >= 0.0 and whole.max() < 1.0


def test_rng_uniform_advances_counter():
    a, b = Rng(7), Rng(7)
    np.testing.assert_array_equal(
        np.concatenate([a.uniform(3), a.uniform(3)]), b.uniform(6)
    )


def test_permutation_and_integers():
    assert Rng(2137).permutation(10).tolist() == [1, 5, 2, 4, 3, 9, 8, 6, 7, 0]
    rng = Rng(2137)
    assert [rng.integers(10) for _ in range(5)] == [9, 0, 4, 4, 4]
    with pytest.raises(ValueError):
        Rng(1).integers(0)
    assert Rng(1).permutation(100).dtype == np.intp


def test_choice_respects_zero_weights_and_range():
    rng = Rng(3)
    idx = rng.choice([0.0, 2.0, 0.0, 1.0, 0.0], 5000)
    assert idx.dtype == np.intp
    assert set(idx.tolist()) <= {1, 3}
    assert set(idx.tolist()) == {1, 3}
    with pytest.raises(ValueError):
        rng.choice([0.0, 0.0], 3)
    with pytest.raises(ValueError):
        rng.choice([1.0, -1.0], 3)


@pytest.mark.parametrize("seed", [-1, 2**53, True, 1.0, "1"])
def test_seed_validation(seed):
    with pytest.raises(ValueError):
        Rng(seed)
    with pytest.raises(ValueError):
        uniform_at(seed, 0, 1)
