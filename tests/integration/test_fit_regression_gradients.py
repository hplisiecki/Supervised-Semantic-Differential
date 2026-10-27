"""fit_regression(ssd_gradients=N): N rotated gradients from the regression's own model."""
from __future__ import annotations

import pickle

import numpy as np
import pytest

from ssdiff import SSD
from ssdiff.utils.math import standardize
from tests.conftest import _make_large_docs

DOCS = _make_large_docs(150, np.random.default_rng(0))


@pytest.fixture(scope="module")
def ssd(tiny_kv_large, lexicon):
    return SSD(DOCS, embeddings=tiny_kv_large, lexicon=lexicon)


@pytest.fixture(scope="module")
def data(ssd):
    X, has_vector, _ = ssd._vectors()
    rng = np.random.default_rng(1)
    y = np.full(len(DOCS), np.nan)
    y[has_vector] = X[:, 0] * 3 + X[:, 1] - X[:, 5] + 0.1 * rng.normal(size=len(X))
    return {"y": y, "age": rng.normal(size=len(DOCS)),
            "hi": (y > np.nanmedian(y)).astype(float),
            "group": rng.choice(["a", "b"], size=len(DOCS))}


@pytest.fixture(scope="module")
def one(ssd, data):
    return ssd.fit_regression(data["y"], fixed_k=10)


@pytest.fixture(scope="module")
def three(ssd, data):
    return ssd.fit_regression(data["y"], fixed_k=10, ssd_gradients=3)


def _span_projector(A):
    Q = np.linalg.qr(A)[0]
    return Q @ Q.T


def test_one_gradient_is_the_standard_fit(ssd, data, one):
    res = ssd.fit_regression(data["y"], fixed_k=10, ssd_gradients=1)
    assert res.keys() == [] and res.pls_info is None
    assert (res.fit_info.ssd_gradients, res.fit_info.rotate) == (1, None)
    assert res.fit_args["ssd_gradients"] == 1
    np.testing.assert_array_equal(res.beta, one.beta)


def test_standard_fit_unchanged(one, three):
    assert three.stats.pvalue == one.stats.pvalue
    assert three.stats.r2 == one.stats.r2
    np.testing.assert_array_equal(three.beta, one.beta)
    np.testing.assert_array_equal(three.predict(DOCS), one.predict(DOCS))
    assert (three.fit_info.ssd_gradients, three.fit_info.rotate) == (3, "words")
    assert three.keys() == ["dim-1", "dim-2", "dim-3"]


def test_leaves_span_the_block_pieces_that_sum_to_beta(three):
    comps, w = three.pca_components, three.pca_weights
    # K = 10 split into 3 consecutive blocks: 4 / 3 / 3, highest variance first.
    pieces = np.column_stack([comps[b].T @ w[b] for b in ([0, 1, 2, 3], [4, 5, 6], [7, 8, 9])])
    X_scale = standardize(three.x)[2]
    np.testing.assert_allclose(pieces.sum(axis=1) / X_scale, three.beta, atol=1e-12)
    W = pieces / np.linalg.norm(pieces, axis=0)
    np.testing.assert_allclose(W.T @ W, np.eye(3), atol=1e-12)
    B = np.column_stack([three[k].beta * X_scale for k in three.keys()])
    np.testing.assert_allclose(_span_projector(B), _span_projector(W), atol=1e-8)


def test_words_rotation_orders_by_positive_score_corr(three):
    corr = [three[k].stats.score_corr for k in three.keys()]
    assert all(c > 0 for c in corr)
    assert corr == sorted(corr, reverse=True)
    assert sum(three[k].stats.cov_share for k in three.keys()) == pytest.approx(1.0)


@pytest.mark.parametrize("outcome, kwargs", [
    ("y ~ age", {}),
    ("hi ~ age", {"family": "binomial"}),
])
def test_with_a_formula_or_another_family(ssd, data, outcome, kwargs):
    plain = ssd.fit_regression(outcome, data=data, fixed_k=6, **kwargs)
    res = ssd.fit_regression(outcome, data=data, fixed_k=6, ssd_gradients=2, **kwargs)
    assert res.keys() == ["dim-1", "dim-2"]
    assert res.stats.pvalue == plain.stats.pvalue
    np.testing.assert_array_equal(res.beta, plain.beta)


@pytest.mark.parametrize("outcome, kwargs, match", [
    ("y", dict(fixed_k=4, ssd_gradients=5), "more than the K=4"),
    ("y", dict(fixed_k=4, ssd_gradients=0), "int >= 1"),
    ("y", dict(fixed_k=4, ssd_gradients=1.5), "int >= 1"),
    ("y ~ group*ssd()", dict(fixed_k=4, ssd_gradients=2), "one plain ssd"),
    ("y ~ age:ssd()", dict(fixed_k=4, ssd_gradients=2), "one plain ssd"),
])
def test_guards(ssd, data, outcome, kwargs, match):
    with pytest.raises(ValueError, match=match):
        ssd.fit_regression(outcome, data=data, **kwargs)


def test_pickle_round_trip_relinks_leaves(three):
    back = pickle.loads(pickle.dumps(three))
    leaf = back["dim-2"]
    assert leaf._container is back
    np.testing.assert_array_equal(leaf.x, back.x)
    assert [w.word for w in leaf.words] == [w.word for w in three["dim-2"].words]
