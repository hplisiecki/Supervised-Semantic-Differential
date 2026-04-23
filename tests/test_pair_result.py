"""Unit tests for PairResult — the group-contrast leaf."""
import numpy as np
import pytest


def _minimal_container(x, groups):
    """Build a tiny object with `_x` and `_groups` that PairResult can bind to."""
    class _C:
        pass
    c = _C()
    c._x = x
    c._groups = groups
    return c


def test_pair_result_beta_is_centroid_diff():
    from ssdiff.results.group_result import PairResult

    x = np.array([
        [1.0, 0.0],  # g1
        [2.0, 0.0],  # g1
        [0.0, 1.0],  # g2
        [0.0, 3.0],  # g2
    ])
    groups = np.array(["g1", "g1", "g2", "g2"])
    c = _minimal_container(x, groups)

    pr = PairResult(
        container=c, g1="g1", g2="g2",
        embeddings=None, corpus=None,
        lexicon=None, window=3, sif_a=1e-3, lang="pl",
    )
    # centroid(g1) - centroid(g2) = (1.5, 0) - (0, 2) = (1.5, -2)
    np.testing.assert_allclose(pr.beta, [1.5, -2.0], rtol=1e-12)


def test_pair_result_x_slices_container():
    from ssdiff.results.group_result import PairResult

    x = np.arange(12).reshape(4, 3).astype(float)
    groups = np.array(["g1", "g1", "g2", "g2"])
    c = _minimal_container(x, groups)

    pr = PairResult(
        container=c, g1="g1", g2="g2",
        embeddings=None, corpus=None,
        lexicon=None, window=3, sif_a=1e-3, lang="pl",
    )
    assert pr.x.shape == (4, 3)  # both groups → all 4 rows
    # groups_mask should be all True for a 2-group container
    assert pr.groups_mask.sum() == 4


def test_pair_result_no_test_attribute():
    from ssdiff.results.group_result import PairResult

    x = np.ones((4, 2))
    groups = np.array(["g1", "g1", "g2", "g2"])
    c = _minimal_container(x, groups)

    pr = PairResult(
        container=c, g1="g1", g2="g2",
        embeddings=None, corpus=None,
        lexicon=None, window=3, sif_a=1e-3, lang="pl",
    )
    assert not hasattr(pr, "test")


def test_pair_result_contrast_string():
    from ssdiff.results.group_result import PairResult

    x = np.ones((4, 2))
    groups = np.array(["g1", "g1", "g2", "g2"])
    c = _minimal_container(x, groups)

    pr = PairResult(
        container=c, g1="g1", g2="g2",
        embeddings=None, corpus=None,
        lexicon=None, window=3, sif_a=1e-3, lang="pl",
    )
    assert pr.contrast == "g1_g2"


def test_pair_result_x_shares_memory_with_container():
    """Leaf x must not duplicate container._x — shared via mask slice."""
    from ssdiff.results.group_result import PairResult

    x = np.arange(12).reshape(4, 3).astype(float)
    groups = np.array(["g1", "g1", "g2", "g2"])
    c = _minimal_container(x, groups)
    pr = PairResult(
        container=c, g1="g1", g2="g2",
        embeddings=None, corpus=None,
        lexicon=None, window=3, sif_a=1e-3, lang="pl",
    )
    # Fancy indexing with a boolean mask does allocate (numpy semantics),
    # but the allocation must happen at most once (cached).
    x1 = pr.x
    x2 = pr.x
    assert x1 is x2


def test_pair_result_pickling_drops_container():
    """__getstate__ must blank _container and _x_cache so the pickle is
    small and pair leaves don't carry back-refs. GroupResult.__setstate__
    (Task 6) will re-wire them."""
    import pickle
    from ssdiff.results.group_result import PairResult

    x = np.arange(12).reshape(4, 3).astype(float)
    groups = np.array(["g1", "g1", "g2", "g2"])
    c = _minimal_container(x, groups)
    pr = PairResult(
        container=c, g1="g1", g2="g2",
        embeddings=None, corpus=None,
        lexicon=None, window=3, sif_a=1e-3, lang="pl",
    )

    state = pr.__getstate__()
    assert state["_container"] is None
    assert state["_x_cache"] is None
    # beta / groups_mask are preserved
    np.testing.assert_allclose(state["beta"], pr.beta)
    np.testing.assert_array_equal(state["groups_mask"], pr.groups_mask)
