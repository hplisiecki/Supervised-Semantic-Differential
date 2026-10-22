"""fit_multipls: PLS3 (several outcomes) fit, rotation modes, guards, rerun."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from ssdiff import SSD
from ssdiff.embeddings import Embeddings
from tests.conftest import _make_large_docs

MODES = (None, "words", "outcomes")


def _std(a):
    return (a - a.mean(0)) / a.std(0)


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


@pytest.fixture(scope="module")
def planted(tiny_kv_large, lexicon):
    docs = _make_large_docs(200, np.random.default_rng(0))
    ssd = SSD(tiny_kv_large, docs, lexicon)
    Xs = _std(ssd._vectors()[0])
    rng = np.random.default_rng(1)
    # PLS3 recovers Xs'Xs d, which points along d only for an eigenvector of
    # Xs'Xs; this X is far from isotropic, so plant along the top two.
    E = np.linalg.eigh(Xs.T @ Xs)[1]
    d1, d2 = E[:, -1], E[:, -2]
    t1, t2 = Xs @ d1, Xs @ d2
    Y = np.column_stack([2 * t1, 2 * t1, t2, t2]) + 0.3 * rng.normal(size=(len(t1), 4))

    def make(cols):
        full = np.full((len(docs), cols.shape[1]), np.nan)
        full[ssd._vectors()[1]] = cols  # rows with no doc vector stay NaN
        return full

    return {
        "ssd": ssd, "d1": d1, "d2": d2, "Y": Y, "make": make,
        "Yfull": make(Y), "docs": docs, "emb": tiny_kv_large, "lexicon": lexicon,
    }


@pytest.fixture(scope="module")
def fits(planted):
    ssd = planted["ssd"]
    return {
        mode: ssd.fit_multipls(planted["Yfull"], ssd_gradients=2, rotate=mode, n_perm=99, n_splits=10)
        for mode in MODES
    }


def _fit_with(planted, Y_kept, **kwargs):
    return planted["ssd"].fit_multipls(planted["make"](Y_kept), **kwargs)


# ---------------------------------------------------------------------------
# Fit and rotation
# ---------------------------------------------------------------------------

def test_planted_directions_recovered(fits, planted):
    U_rot = fits[None].U_rot
    assert abs(_cos(U_rot[:, 0], planted["d1"])) > 0.9
    assert abs(_cos(U_rot[:, 1], planted["d2"])) > 0.9


@pytest.mark.parametrize("mode", MODES)
def test_rotated_saliences_orthonormal(fits, mode):
    res = fits[mode]
    np.testing.assert_allclose(res.U_rot.T @ res.U_rot, np.eye(2), atol=1e-8)
    np.testing.assert_allclose(res.V_rot.T @ res.V_rot, np.eye(2), atol=1e-8)


def test_cov_explained_and_pvalue_rotation_independent(fits):
    ref = fits[None].stats
    for mode in ("words", "outcomes"):
        s = fits[mode].stats
        assert s.cov_explained == pytest.approx(ref.cov_explained, rel=1e-10)
        assert s.pvalue == ref.pvalue


@pytest.mark.parametrize("mode", MODES)
def test_cov_share(fits, planted, mode):
    res = fits[mode]
    Xs, Ys = _std(planted["ssd"]._vectors()[0]), _std(planted["Y"])
    total = np.linalg.norm(Xs.T @ Ys) ** 2
    shares = np.array([res[f"dim-{i + 1}"].stats.cov_share for i in range(2)])
    assert shares.sum() == pytest.approx(1.0)
    for i in range(2):
        c_i = np.linalg.norm(Ys.T @ Xs @ res.U_rot[:, i]) ** 2
        assert shares[i] * res.stats.cov_explained == pytest.approx(c_i / total)


def test_outcome_groups_rows(planted):
    res = planted["ssd"].fit_multipls(
        planted["Yfull"], ssd_gradients=2, rotate="outcomes", outcome_groups={"g1": ["y1", "y2"]},
        n_perm=19, n_splits=5,
    )
    rows = list(res["dim-1"].outcomes)
    assert len(rows) == 5
    by_name = {r.outcome: r for r in rows}
    assert by_name["y1"].group == "g1"
    assert by_name["y3"].group is None
    assert by_name["g1"].group == "g1"
    assert by_name["g1"].loading == pytest.approx(
        (by_name["y1"].loading + by_name["y2"].loading) / 2
    )


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def test_guard_1d_y(planted):
    y = planted["make"](planted["Y"][:, :1]).ravel()
    with pytest.raises(ValueError, match="fit_pls"):
        planted["ssd"].fit_multipls(y, ssd_gradients=1)


def test_guard_single_column_2d_y(planted):
    with pytest.raises(ValueError, match="fit_pls"):
        _fit_with(planted, planted["Y"][:, :1], ssd_gradients=1)


@pytest.mark.parametrize("kwargs, match", [
    (dict(ssd_gradients=0), r"\[1, 4\]"),
    (dict(ssd_gradients=5), r"\[1, 4\]"),
    (dict(ssd_gradients=3, rotate="outcomes",
          outcome_groups={"a": ["y1", "y2"], "b": ["y3", "y4"]}), "target rows"),
    (dict(ssd_gradients=2, outcome_groups={"a": ["y1", "y2"]}), "requires rotate='outcomes'"),
    (dict(ssd_gradients=2, rotate="words", outcome_groups={"a": ["y1", "y2"]}),
     "requires rotate='outcomes'"),
    (dict(ssd_gradients=2, rotate="outcomes", outcome_groups={"a": ["y1", "nope"]}),
     "unknown outcome"),
    (dict(ssd_gradients=2, rotate="outcomes", outcome_groups={"a": ["y1"], "b": ["y1", "y2"]}),
     "two groups"),
    (dict(ssd_gradients=2, rotate="outcomes", outcome_groups={"a": []}), "empty"),
    (dict(ssd_gradients=2, rotate="varimax"), "rotate must be"),
])
def test_guards_value_error(planted, kwargs, match):
    with pytest.raises(ValueError, match=match):
        planted["ssd"].fit_multipls(planted["Yfull"], **kwargs)


def test_guard_constant_column(planted):
    Y = planted["Y"].copy()
    Y[:, 2] = 3.0
    with pytest.raises(ValueError, match="y3"):
        _fit_with(planted, Y, ssd_gradients=2)


def test_default_ssd_gradients(planted):
    res = planted["ssd"].fit_multipls(planted["Yfull"], n_perm=19, n_splits=5)
    assert res.keys() == ["dim-1", "dim-2", "dim-3", "dim-4"]
    # With groups the default is the number of rotation target rows: g1 + y3 + y4.
    grouped = planted["ssd"].fit_multipls(
        planted["Yfull"], rotate="outcomes", outcome_groups={"g1": ["y1", "y2"]},
        n_perm=19, n_splits=5,
    )
    assert len(grouped.keys()) == 3


def test_outcomes_rotation_at_q_loads_each_dim_on_one_outcome(planted):
    V_rot = planted["ssd"].fit_multipls(
        planted["Yfull"], rotate="outcomes", n_perm=19, n_splits=5
    ).V_rot
    # A signed permutation of the identity: one ±1 per column, in distinct rows.
    np.testing.assert_allclose(np.abs(V_rot).max(axis=0), 1.0, atol=1e-10)
    np.testing.assert_allclose(np.abs(V_rot).sum(axis=0), 1.0, atol=1e-10)
    assert sorted(np.argmax(np.abs(V_rot), axis=0)) == [0, 1, 2, 3]


def test_guard_no_embeddings(planted):
    ssd = copy.copy(planted["ssd"])
    ssd.embeddings = None
    with pytest.raises(RuntimeError, match="no embeddings"):
        ssd.fit_multipls(planted["Yfull"], ssd_gradients=2)


def test_guard_k_used_below_k(planted):
    Y = planted["Y"]
    Y5 = np.column_stack([Y, Y[:, 0] + Y[:, 2]])
    with pytest.raises(ValueError, match="k_used"):
        _fit_with(planted, Y5, ssd_gradients=5)


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def test_outcome_scale_invariance(planted, fits):
    Y2 = planted["Y"].copy()
    Y2[:, 0] *= 1000
    res = _fit_with(planted, Y2, ssd_gradients=2, n_perm=99, n_splits=10)
    ref = fits[None]
    assert res.stats.cov_explained == pytest.approx(ref.stats.cov_explained, rel=1e-6)
    np.testing.assert_allclose(np.abs(res.U_rot), np.abs(ref.U_rot), rtol=1e-6)


def test_dataframe_integer_column_names(planted):
    import pandas as pd

    df = pd.DataFrame(planted["make"](planted["Y"]), columns=[0, 1, 2, 3])
    res = planted["ssd"].fit_multipls(
        df, ssd_gradients=2, rotate="outcomes", outcome_groups={"g": ["0", "1"]},
        n_perm=19, n_splits=5,
    )
    assert res.outcome_names == ["0", "1", "2", "3"]
    assert "0" in [r.outcome for r in res["dim-1"].outcomes]


def test_test_rerun_updates_pvalue(planted):
    res = planted["ssd"].fit_multipls(planted["Yfull"], ssd_gradients=2, n_perm=19, n_splits=5)
    res.test(n_splits=5, n_perm=49, seed=1)
    assert res.test.name == "split_exact"
    assert res.test.n_perm == 49
    assert res.stats.pvalue == res.test.pvalue


def test_words_rotation_on_a_partial_wordpack(wordpack_path, monkeypatch):
    import ssdiff.embeddings as em
    monkeypatch.setattr(em, "_RAM_TOP_N", 18)

    emb = Embeddings.load(wordpack_path)
    keys = list(emb.index_to_key)
    rng = np.random.default_rng(0)
    seeds = keys[15:18]
    docs = [
        [seeds[j % 3]] + [keys[i] for i in rng.choice(np.arange(18, 48), 3, replace=False)]
        for j in range(60)
    ]
    ssd = SSD(emb, docs, lexicon=set(seeds))
    res = ssd.fit_multipls(rng.normal(size=(60, 3)), ssd_gradients=2, rotate="words", n_perm=19, n_splits=5)
    assert res.U_rot.shape == (16, 2)
