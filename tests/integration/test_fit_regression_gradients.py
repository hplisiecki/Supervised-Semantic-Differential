"""fit_regression(ssd_gradients=N): N rotated parts of the regression's own beta."""
from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np
import plskit
import pytest

from ssdiff import SSD
from ssdiff.ssd import _regression_gradients
from ssdiff.utils.math import standardize
from tests.conftest import _make_large_docs, assert_leaves_sum_to_beta

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
    assert (three.fit_info.ssd_gradients, three.fit_info.rotate) == (3, "docs")
    assert three.keys() == ["dim-1", "dim-2", "dim-3"]


def test_leaves_sum_to_beta(three):
    assert_leaves_sum_to_beta(three)


@pytest.mark.parametrize("outcome, kwargs", [
    ("y ~ age", {}),
    ("hi ~ age", {"family": "binomial"}),
])
def test_leaves_sum_to_beta_with_a_formula_or_another_family(ssd, data, outcome, kwargs):
    assert_leaves_sum_to_beta(ssd.fit_regression(outcome, data=data, fixed_k=6, ssd_gradients=2, **kwargs))


def test_leaves_match_factors_and_rotate(ssd, data):
    res = ssd.fit_regression(data["y"], fixed_k=6, ssd_gradients=3)
    Xs, _, X_scale = standardize(res.x)
    V = res.pca_components.T
    f = plskit.factors(V @ res.pca_weights, Xs @ V @ V.T, 3, pre_standardized=True)
    f = plskit.rotate(f, L=Xs @ f.W)
    for i, key in enumerate(res.keys()):
        np.testing.assert_allclose(res[key].beta * X_scale, f.F[:, i], atol=1e-12)


def test_factors_match_plskit_factors(ssd, data):
    res = ssd.fit_regression(data["y"], fixed_k=6, ssd_gradients=3, gradients="factors")
    Xs, _, X_scale = standardize(res.x)
    V = res.pca_components.T
    f = plskit.factors(V @ res.pca_weights, Xs @ V @ V.T, 3, pre_standardized=True)
    for i, key in enumerate(res.keys()):
        np.testing.assert_allclose(res[key].beta * X_scale, f.F[:, i], atol=1e-12)
    assert_leaves_sum_to_beta(res)
    assert res.fit_info.rotate is None and next(iter(res.pls_info))["rotate"] is None
    assert res.fit_args["gradients"] == "factors"


@pytest.mark.parametrize("K, n", [(3, 3), (6, 3), (6, 6), (9, 2)])
def test_split_is_along_the_K_components(ssd, data, K, n):
    res = ssd.fit_regression(data["y"], fixed_k=K)
    Xs = standardize(res.x)[0]
    V, w = res.pca_components.T, res.pca_weights
    f = plskit.factors(V @ w, Xs @ V @ V.T, n, pre_standardized=True)
    c = V.T @ f.F
    block = np.abs(c).argmax(axis=1)
    assert np.all(np.diff(block) >= 0) and block[-1] == n - 1
    np.testing.assert_allclose(np.abs(c).sum(axis=1), np.abs(w), atol=1e-12)
    np.testing.assert_allclose(c[np.arange(K), block], w, atol=1e-12)


def test_leaves_are_in_descending_size(three):
    X_scale = standardize(three.x)[2]
    sizes = [np.linalg.norm(three[k].beta * X_scale) for k in three.keys()]
    assert sizes == sorted(sizes, reverse=True)
    assert sum(three[k].stats.cov_share for k in three.keys()) == pytest.approx(1.0)


def _parts():
    rng = np.random.default_rng(0)
    Xs = standardize(rng.normal(size=(30, 4)))[0]
    return SimpleNamespace(components=np.linalg.svd(Xs, full_matrices=False)[2][:3], Xs=Xs,
                           ys=standardize(rng.normal(size=(30, 1)))[0].ravel(), X_scale=np.ones(4))


def test_aliased_coefficient_names_the_component():
    with pytest.raises(ValueError, match="component 2"):
        _regression_gradients(_parts(), np.array([1.0, np.nan, 1.0]), 2, "profiles")


def test_zero_block_is_a_plskit_error():
    with pytest.raises(plskit.PlsKitError) as err:
        _regression_gradients(_parts(), np.array([1.0, 0.0, 0.0]), 3, "profiles")
    assert err.value.code == "invalid_input"


def test_component_beyond_the_rank_is_a_plskit_error():
    rng = np.random.default_rng(0)
    Xs = standardize(rng.normal(size=(30, 2)) @ rng.normal(size=(2, 4)))[0]
    # Rank 2: plskit takes at most rank + 1 parts, the + 1 for beta's part outside the row space.
    p = SimpleNamespace(components=np.linalg.svd(Xs, full_matrices=False)[2], Xs=Xs,
                        ys=standardize(rng.normal(size=(30, 1)))[0].ravel(), X_scale=np.ones(4))
    with pytest.raises(plskit.PlsKitError) as err:
        _regression_gradients(p, np.ones(4), 4, "profiles")
    assert err.value.code == "invalid_input"


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
    ("y", dict(fixed_k=4, ssd_gradients=2, gradients="rotated"), "'profiles' or 'factors'"),
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


def test_module_fit_regression_passes_gradients(ssd, data, three):
    import ssdiff
    res = ssdiff.fit_regression("y ~ ssd(essay)", data, fixed_k=10, ssd_gradients=3, essay=ssd)
    assert res.keys() == three.keys() and res.fit_args["ssd_gradients"] == 3
    for key in three.keys():
        np.testing.assert_allclose(res[key].beta, three[key].beta, rtol=1e-10)
    with pytest.raises(ValueError, match="one plain ssd"):
        ssdiff.fit_regression("y ~ ssd(a) + ssd(b)", data, fixed_k=4, ssd_gradients=2, a=ssd, b=ssd)
