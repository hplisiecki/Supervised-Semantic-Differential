"""fit_pls(ssd_gradients=N): N rotated gradients beside an unchanged standard fit."""
from __future__ import annotations

import pickle

import numpy as np
import plskit
import pytest

from ssdiff import SSD, Corpus
from ssdiff.utils.math import standardize
from tests.conftest import _make_large_docs

FIT = dict(k=2, n_splits=10, random_state=0)


@pytest.fixture(scope="module")
def ssd(tiny_kv_large, lexicon):
    docs = _make_large_docs(150, np.random.default_rng(0))
    corpus = Corpus(docs, pretokenized=True, lang="pl")
    probe = SSD(tiny_kv_large, corpus, np.zeros(len(docs)), lexicon)
    rng = np.random.default_rng(1)
    y = np.full(len(docs), np.nan)
    y[probe._keep_mask] = probe.x[:, 0] * 3 + probe.x[:, 1] + 0.1 * rng.normal(size=probe.n_kept)
    data = {"age": rng.normal(size=len(docs))}
    return SSD(tiny_kv_large, corpus, y, lexicon, data=data)


@pytest.fixture(scope="module")
def one(ssd):
    return ssd.fit_pls(**FIT)


@pytest.fixture(scope="module")
def three(ssd):
    return ssd.fit_pls(**FIT, ssd_gradients=3)


def _span_projector(A):
    Q = np.linalg.qr(A)[0]
    return Q @ Q.T


def test_three_leaves_span_the_k3_weights(ssd, three):
    assert three.keys() == ["dim-1", "dim-2", "dim-3"]
    Xs, _, X_scale = standardize(ssd.x)
    ys = standardize(ssd.y.reshape(-1, 1))[0].ravel()
    W = plskit.pls1_fit(Xs, ys, k=3, pre_standardized=True, seed=0).W
    B = np.column_stack([three[k].beta * X_scale for k in three.keys()])
    np.testing.assert_allclose(_span_projector(B), _span_projector(W), atol=1e-8)


def test_standard_fit_unchanged(ssd, one, three):
    assert three.stats.pvalue == one.stats.pvalue
    assert three.stats.r2 == one.stats.r2
    np.testing.assert_array_equal(three.beta, one.beta)
    np.testing.assert_array_equal(three.predict(ssd.corpus), one.predict(ssd.corpus))
    assert three.fit_info.n_components == one.fit_info.n_components
    assert (three.fit_info.ssd_gradients, three.fit_info.rotate) == (3, "words")
    assert one.keys() == [] and one.pls_info is None


def test_words_rotation_orders_by_positive_score_corr(three):
    corr = [three[k].stats.score_corr for k in three.keys()]
    assert all(c > 0 for c in corr)
    assert corr == sorted(corr, reverse=True)
    assert sum(three[k].stats.cov_share for k in three.keys()) == pytest.approx(1.0)


def test_rotate_none_keeps_nipals_weights(ssd):
    res = ssd.fit_pls(**FIT, ssd_gradients=3, rotate=None)
    Xs, _, X_scale = standardize(ssd.x)
    ys = standardize(ssd.y.reshape(-1, 1))[0].ravel()
    W = plskit.pls1_fit(Xs, ys, k=3, pre_standardized=True, seed=0).W
    for i, k in enumerate(res.keys()):
        np.testing.assert_allclose(res[k].beta * X_scale, W[:, i], atol=1e-12)
    assert res["dim-2"].stats.cov_share == pytest.approx(0.0, abs=1e-12)


def test_with_formula(ssd):
    plain = ssd.fit_pls("~ age", **FIT)
    res = ssd.fit_pls("~ age", **FIT, ssd_gradients=3)
    assert res.keys() == ["dim-1", "dim-2", "dim-3"]
    assert res.stats.pvalue == plain.stats.pvalue
    np.testing.assert_array_equal(res.beta, plain.beta)


@pytest.mark.parametrize("kwargs, match", [
    (dict(ssd_gradients=3, rotate="outcomes"), "rotate must be"),
    (dict(ssd_gradients=11), r"\[1, 10\]"),
    (dict(ssd_gradients=0), r"\[1, 10\]"),
])
def test_guards(ssd, kwargs, match):
    with pytest.raises(ValueError, match=match):
        ssd.fit_pls(**FIT, **kwargs)


def test_pickle_round_trip_relinks_leaves(three):
    back = pickle.loads(pickle.dumps(three))
    leaf = back["dim-2"]
    assert leaf._container is back
    np.testing.assert_array_equal(leaf.x, back.x)
    assert [w.word for w in leaf.words] == [w.word for w in three["dim-2"].words]
