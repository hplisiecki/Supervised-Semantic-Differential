"""Sanity tests for the _SingleResult leaf base class.

These tests construct a minimal _SingleResult directly (no subclass) and
verify it exposes the expected gradient-derived attrs/views without any
y/fit_info machinery.
"""
import numpy as np
import pytest


def test_single_result_basic_attrs():
    from ssdiff.results.single_result import _SingleResult

    x = np.random.RandomState(0).randn(20, 8).astype(float)
    beta = np.random.RandomState(1).randn(8).astype(float)

    r = _SingleResult(
        x=x, beta=beta,
        embeddings=None, corpus=None,
        lexicon=None, window=3, sif_a=1e-3, lang="pl",
    )

    assert r.x is x
    assert np.array_equal(r.beta, beta)
    np.testing.assert_allclose(
        r.gradient, beta / np.linalg.norm(beta), rtol=1e-12,
    )
    assert r.beta_norm == pytest.approx(float(np.linalg.norm(beta)))
    assert r.lexicon == set()
    assert r.window == 3
    assert r.sif_a == 1e-3
    assert r.lang == "pl"


def test_single_result_alignment_scores_cached():
    from ssdiff.results.single_result import _SingleResult

    x = np.random.RandomState(0).randn(10, 5).astype(float)
    beta = np.array([1.0, 0, 0, 0, 0])
    r = _SingleResult(
        x=x, beta=beta, embeddings=None, corpus=None,
        lexicon=None, window=3, sif_a=1e-3, lang="pl",
    )

    a1 = r.alignment_scores
    a2 = r.alignment_scores
    assert a1 is a2  # cached reference
    # should be cosine of each x row with unit(beta)
    x_norms = np.linalg.norm(x, axis=1, keepdims=True)
    expected = (x / np.maximum(x_norms, 1e-12) @ r.gradient).ravel()
    np.testing.assert_allclose(a1, expected, rtol=1e-12)


def test_single_result_words_requires_embeddings():
    from ssdiff.results.single_result import _SingleResult

    x = np.random.RandomState(0).randn(10, 5).astype(float)
    beta = np.array([1.0, 0, 0, 0, 0])
    r = _SingleResult(
        x=x, beta=beta, embeddings=None, corpus=None,
        lexicon=None, window=3, sif_a=1e-3, lang="pl",
    )
    with pytest.raises(RuntimeError, match="embeddings"):
        _ = r.words


def test_single_result_snippets_requires_corpus():
    from ssdiff.results.single_result import _SingleResult

    x = np.random.RandomState(0).randn(10, 5).astype(float)
    beta = np.array([1.0, 0, 0, 0, 0])
    r = _SingleResult(
        x=x, beta=beta, embeddings=object(), corpus=None,
        lexicon=None, window=3, sif_a=1e-3, lang="pl",
    )
    with pytest.raises(RuntimeError, match="corpus"):
        _ = r.snippets


def test_single_result_clusters_is_eager_wrapper():
    """`.clusters` must be an eager ClustersView instance, not a property
    backed by a private slot — keeps caching discipline consistent."""
    from ssdiff.results.continuous_result import ClustersView
    from ssdiff.results.single_result import _SingleResult

    x = np.random.RandomState(0).randn(10, 5).astype(float)
    beta = np.array([1.0, 0, 0, 0, 0])
    r = _SingleResult(
        x=x, beta=beta, embeddings=object(), corpus=None,
        lexicon=None, window=3, sif_a=1e-3, lang="pl",
    )
    assert isinstance(r.clusters, ClustersView)
    assert r.clusters is r.clusters  # stable identity


def test_single_result_clusters_for_requires_embeddings():
    from ssdiff.results.single_result import _SingleResult

    x = np.random.RandomState(0).randn(10, 5).astype(float)
    beta = np.array([1.0, 0, 0, 0, 0])
    r = _SingleResult(
        x=x, beta=beta, embeddings=None, corpus=None,
        lexicon=None, window=3, sif_a=1e-3, lang="pl",
    )
    with pytest.raises(RuntimeError, match="embeddings"):
        r._clusters_for("pos")


def test_single_result_cluster_snippets_requires_embeddings():
    from ssdiff.results.single_result import _SingleResult

    x = np.random.RandomState(0).randn(10, 5).astype(float)
    beta = np.array([1.0, 0, 0, 0, 0])
    r = _SingleResult(
        x=x, beta=beta, embeddings=None, corpus=None,
        lexicon=None, window=3, sif_a=1e-3, lang="pl",
    )
    with pytest.raises(RuntimeError, match="embeddings"):
        r.cluster_snippets(side="pos")
