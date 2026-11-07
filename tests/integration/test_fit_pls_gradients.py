"""fit_pls(ssd_gradients=N): N rotated parts of beta beside an unchanged standard fit."""
from __future__ import annotations

import pickle

import numpy as np
import plskit
import pytest

from ssdiff import SSD
from ssdiff.backends.pls import word_target
from ssdiff.utils.math import standardize
from tests.conftest import _make_large_docs, assert_leaves_sum_to_beta

FIT = dict(k=2, n_splits=10, random_state=0)
DOCS = _make_large_docs(150, np.random.default_rng(0))


@pytest.fixture(scope="module")
def ssd(tiny_kv_large, lexicon):
    return SSD(DOCS, embeddings=tiny_kv_large, lexicon=lexicon)


@pytest.fixture(scope="module")
def data(ssd):
    X, has_vector, _ = ssd._vectors()
    rng = np.random.default_rng(1)
    y = np.full(len(DOCS), np.nan)
    y[has_vector] = X[:, 0] * 3 + X[:, 1] + 0.1 * rng.normal(size=len(X))
    return {"y": y, "age": rng.normal(size=len(DOCS))}


@pytest.fixture(scope="module")
def one(ssd, data):
    return ssd.fit_pls(data["y"], **FIT)


@pytest.fixture(scope="module")
def three(ssd, data):
    return ssd.fit_pls(data["y"], **FIT, ssd_gradients=3)


def test_standard_fit_unchanged(one, three):
    assert three.stats.pvalue == one.stats.pvalue
    assert three.stats.r2 == one.stats.r2
    np.testing.assert_array_equal(three.beta, one.beta)
    np.testing.assert_array_equal(three.predict(DOCS), one.predict(DOCS))
    assert three.fit_info.n_components == one.fit_info.n_components
    assert (three.fit_info.ssd_gradients, three.fit_info.rotate) == (3, "docs")
    assert one.keys() == [] and one.pls_info is None


def test_leaves_sum_to_beta(three):
    assert_leaves_sum_to_beta(three)


def test_leaves_sum_to_beta_with_formula(ssd, data):
    assert_leaves_sum_to_beta(ssd.fit_pls("y ~ age", data=data, **FIT, ssd_gradients=3))


def test_leaves_sum_to_beta_with_auto_k(ssd, data):
    res = ssd.fit_pls(data["y"], k="auto", k_max=3, n_splits=10, random_state=0, ssd_gradients=3)
    assert_leaves_sum_to_beta(res)


def test_leaves_sum_to_beta_at_the_rank(ssd, data, three):
    rank = int(np.linalg.matrix_rank(standardize(three.x)[0]))
    res = ssd.fit_pls(data["y"], **FIT, ssd_gradients=rank)
    assert len(res.keys()) == rank
    assert_leaves_sum_to_beta(res)


@pytest.mark.parametrize("k", [1, 5])
def test_leaves_match_profiles(ssd, data, k):
    res = ssd.fit_pls(data["y"], k=k, n_splits=10, random_state=0, test=False, ssd_gradients=3)
    Xs, _, X_scale = standardize(res.x)
    ys = standardize(res.y.reshape(-1, 1))[0].ravel()
    m = plskit.pls1_fit(Xs, ys, k=k, pre_standardized=True, seed=0)
    f = plskit.profiles(m, Xs, 3).factors
    for i, key in enumerate(res.keys()):
        np.testing.assert_allclose(res[key].beta * X_scale, f.F[:, i], atol=1e-12)


def test_factors_match_plskit_factors(ssd, data):
    res = ssd.fit_pls(data["y"], k=2, n_splits=10, random_state=0, test=False,
                      ssd_gradients=3, gradients="factors")
    Xs, _, X_scale = standardize(res.x)
    ys = standardize(res.y.reshape(-1, 1))[0].ravel()
    m = plskit.pls1_fit(Xs, ys, k=2, pre_standardized=True, seed=0)
    f = plskit.factors(m, Xs, 3)
    for i, key in enumerate(res.keys()):
        np.testing.assert_allclose(res[key].beta * X_scale, f.F[:, i], atol=1e-12)
    assert_leaves_sum_to_beta(res)
    assert res.fit_info.rotate is None and next(iter(res.pls_info))["rotate"] is None
    assert res.fit_args["gradients"] == "factors"


def _plain_fit(res, k):
    Xs, X_mean, X_scale = standardize(res.x)
    ys = standardize(res.y.reshape(-1, 1))[0].ravel()
    return plskit.pls1_fit(Xs, ys, k=k, pre_standardized=True, seed=0), X_mean, X_scale


def test_components_raw_are_beta_on_each_pls_weight(ssd, data):
    res = ssd.fit_pls(data["y"], k=3, n_splits=10, random_state=0, test=False,
                      ssd_gradients=3, gradients="components_raw")
    m, _, X_scale = _plain_fit(res, 3)
    for i, key in enumerate(res.keys()):
        w = m.W[:, i]
        np.testing.assert_allclose(res[key].beta * X_scale, w * (w @ m.coef), atol=1e-12)
    assert_leaves_sum_to_beta(res)
    assert res.fit_info.rotate is None


def test_components_are_beta_on_each_word_rotated_weight(ssd, data):
    res = ssd.fit_pls(data["y"], k=3, n_splits=10, random_state=0, test=False,
                      ssd_gradients=3, gradients="components")
    m, X_mean, X_scale = _plain_fit(res, 3)
    L = word_target(ssd.embeddings.vectors, m.W, X_mean, X_scale)
    W_rot = plskit.rotate(m.W, method="varimax", L=L).W_rot
    parts = [W_rot[:, j] * (W_rot[:, j] @ m.coef) for j in range(3)]
    parts.sort(key=np.linalg.norm, reverse=True)
    for key, part in zip(res.keys(), parts):
        np.testing.assert_allclose(res[key].beta * X_scale, part, atol=1e-12)
    assert_leaves_sum_to_beta(res)
    assert res.fit_info.rotate == "words"


def test_leaves_are_in_descending_size(three):
    X_scale = standardize(three.x)[2]
    sizes = [np.linalg.norm(three[k].beta * X_scale) for k in three.keys()]
    assert sizes == sorted(sizes, reverse=True)
    assert sum(three[k].stats.cov_share for k in three.keys()) == pytest.approx(1.0)


def test_pls_info_has_no_order_or_signs(three):
    pi = next(iter(three.pls_info))
    assert (pi["rotate"], pi["order"], pi["signs"]) == ("docs", None, None)
    assert "docs" in repr(three.pls_info)


def test_leaves_keep_the_outcome_pole_names(ssd, data):
    res = ssd.fit_pls("y", data=data, **FIT, ssd_gradients=3)
    assert {res[k]._pole_names() for k in res.keys()} == {("higher y", "lower y")}


def test_rotate_is_gone(ssd, data):
    with pytest.raises(TypeError, match="rotate"):
        ssd.fit_pls(data["y"], **FIT, rotate="words")


def test_with_formula(ssd, data):
    plain = ssd.fit_pls("y ~ age", data=data, **FIT)
    res = ssd.fit_pls("y ~ age", data=data, **FIT, ssd_gradients=3)
    assert res.keys() == ["dim-1", "dim-2", "dim-3"]
    assert res.stats.pvalue == plain.stats.pvalue
    np.testing.assert_array_equal(res.beta, plain.beta)


@pytest.mark.parametrize("kwargs, match", [
    (dict(ssd_gradients=11), r"\[1, 10\]"),
    (dict(ssd_gradients=0), r"\[1, 10\]"),
    (dict(ssd_gradients=3, gradients="rotated"), "'components_raw'"),
    (dict(ssd_gradients=3, gradients="components"), "equal to the fit's k=2"),
    (dict(gradients="components_raw"), "equal to the fit's k=2"),
])
def test_guards(ssd, data, kwargs, match):
    with pytest.raises(ValueError, match=match):
        ssd.fit_pls(data["y"], **FIT, **kwargs)


def test_pickle_round_trip_relinks_leaves(three):
    back = pickle.loads(pickle.dumps(three))
    leaf = back["dim-2"]
    assert leaf._container is back
    np.testing.assert_array_equal(leaf.x, back.x)
    assert [w.word for w in leaf.words] == [w.word for w in three["dim-2"].words]
