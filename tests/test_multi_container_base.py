"""Sanity tests for the _MultiContainer / _ShimView bases.

Exercises the container without the pair-specific semantics added by
GroupResult: plain-hashable keys, default str/repr hooks, dict-only shim.
"""
import numpy as np
import pytest


def _make_leaf(seed=0, lang="pl"):
    from ssdiff.results.single_result import _SingleResult
    rng = np.random.RandomState(seed)
    return _SingleResult(
        x=rng.randn(10, 4), beta=rng.randn(4),
        embeddings=None, corpus=None,
        lexicon=None, window=3, sif_a=1e-3, lang=lang,
    )


def test_container_dict_access():
    from ssdiff.results.multi_container import _MultiContainer

    leaves = {"a": _make_leaf(0), "b": _make_leaf(1)}
    c = _MultiContainer.__new__(_MultiContainer)
    c._cache = {}
    c._leaves = leaves

    assert c["a"] is leaves["a"]
    assert c["b"] is leaves["b"]
    with pytest.raises(KeyError):
        c["z"]
    assert list(c.keys()) == ["a", "b"]
    assert len(c) == 2


def test_container_beta_dict_is_plain_dict():
    from ssdiff.results.multi_container import _MultiContainer

    leaves = {"a": _make_leaf(0), "b": _make_leaf(1)}
    c = _MultiContainer.__new__(_MultiContainer)
    c._cache = {}
    c._leaves = leaves

    betas = c.beta
    assert set(betas) == {"a", "b"}
    assert isinstance(betas, dict)
    np.testing.assert_array_equal(betas["a"], leaves["a"].beta)
    np.testing.assert_array_equal(betas["b"], leaves["b"].beta)


def test_shim_view_is_dict_only_no_flat_iter():
    from ssdiff.results.multi_container import _MultiContainer, _ShimView

    class _FakeView:
        def __init__(self, name): self.name = name

    leaves = {"k1": _FakeView("w1"), "k2": _FakeView("w2")}
    c = _MultiContainer.__new__(_MultiContainer)
    c._cache = {}
    c._leaves = {}  # unused here

    shim = _ShimView(
        leaves=leaves, view_name="words", container=c,
    )
    assert shim["k1"].name == "w1"
    assert shim.keys() == ["k1", "k2"]
    assert len(shim) == 2
