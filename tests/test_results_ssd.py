"""ContinuousResult / PLSResult / PCAOLSResult — view contract and cache.

Heavy / integration coverage lives in tests/test_results.py (ported later).
This file focuses on the view surface and cache semantics in isolation,
with synthetic small inputs.
"""

import numpy as np
import pytest

from ssdiff.results.continuous_result import PCAOLSResult, PLSResult
from ssdiff.results.schema import (
    Cluster,
    Word,
)


def _make_continuous(n=50, d=8, seed=0):
    rng = np.random.default_rng(seed)
    beta = rng.standard_normal(d)
    x = rng.standard_normal((n, d))
    y = x @ beta + 0.1 * rng.standard_normal(n)
    return {
        "backend": "PLS",
        "x": x.astype(np.float64),
        "beta": beta.astype(np.float64),
        "keep_mask": np.ones(n, dtype=bool),
        "n_raw": n, "n_kept": n, "n_dropped": 0,
        "y": y.astype(np.float64),
        "_y_mean": np.array([y.mean()]),
        "_y_scale": np.array([y.std() + 1e-12]),
        "r2": 0.47, "r2_adj": 0.46, "pvalue": 1e-5,
        "embeddings": None, "lexicon": set(), "window": 3, "sif_a": 1e-3, "lang": "pl",
    }


def test_pls_stats_scalarview_exposes_fields():
    r = PLSResult(**_make_continuous())
    assert isinstance(r.stats, object)
    assert abs(r.stats.r2 - 0.47) < 1e-9
    assert r.stats.backend == "PLS"
    assert r.stats.n_kept == 50


def test_stats_to_dict_is_flat():
    r = PLSResult(**_make_continuous())
    d = r.stats.to_dict()
    assert "r2" in d and "n_kept" in d


def test_words_view_requires_embeddings():
    r = PLSResult(**_make_continuous())
    with pytest.raises(RuntimeError, match="attach"):
        _ = r.words


def test_clusters_param_cache_keeps_variants(monkeypatch):
    """D5: different params → different cache entries; neither overwrites the other."""
    r = PLSResult(**_make_continuous())
    r.attach(embeddings=object())

    calls = {"count": 0}

    def _fake_clusters(self, *, side, topn, k, k_min, k_max, random_state, min_cluster_size):
        calls["count"] += 1
        return (
            [Cluster(cluster_id=0, side=side, size=topn,
                     coherence=0.5, centroid_cos_beta=0.3, contrast=None)],
            [],
        )

    monkeypatch.setattr(type(r), "_compute_clusters_for_side", _fake_clusters)

    v100 = r.clusters.pos                    # defaults topn=100
    v50 = r.clusters.pos(topn=50)            # new entry
    v100_again = r.clusters.pos              # cache hit, NOT overwritten
    assert v100 is v100_again
    assert v100[0].size == 100
    assert v50[0].size == 50
    assert calls["count"] == 2


def test_clear_cache_drops_clusters():
    r = PLSResult(**_make_continuous())
    key = ("clusters", (("side", "pos"), ("topn", 100)))
    r._cache[key] = object()
    r.clear_cache("clusters")
    assert key not in r._cache


def test_docs_pos_neg_ordered_by_yhat():
    r = PLSResult(**_make_continuous())
    pos3 = r.docs.pos(3)
    yhats = [d.y_hat for d in pos3]
    assert yhats == sorted(yhats, reverse=True)
    neg3 = r.docs.neg(3)
    yhats = [d.y_hat for d in neg3]
    assert yhats == sorted(yhats)


def test_repr_one_liner():
    r = PLSResult(**_make_continuous())
    s = repr(r)
    assert "PLSResult" in s
    assert "r²" in s or "r2" in s


def test_result_repr_html_is_compact_html():
    r = PLSResult(**_make_continuous())
    html = r._repr_html_()
    assert "PLSResult" in html


def test_report_save_writes_markdown(tmp_path):
    r = PLSResult(**_make_continuous())
    p = tmp_path / "r.md"
    r.report().save(str(p))
    content = p.read_text(encoding="utf-8")
    assert "#" in content  # atx headers


def test_per_view_save_writes_files(tmp_path):
    r = PLSResult(**_make_continuous())
    r.stats.save(str(tmp_path / "stats.csv"))
    r.docs.save(str(tmp_path / "docs.csv"))
    assert (tmp_path / "stats.csv").exists()
    assert (tmp_path / "docs.csv").exists()


def test_pcaols_has_sweep_and_f_test():
    data = _make_continuous()
    data["backend"] = "PCA+OLS"
    data["sweep"] = [(k, 0.3 + 0.01 * k, 0.3, 0.01) for k in range(1, 6)]
    r = PCAOLSResult(**data)
    assert hasattr(r, "sweep")
    assert r.test.name == "f_test"
    assert r.test.pvalue == data["pvalue"]


def test_pls_has_test_view_not_sweep():
    data = _make_continuous()
    data["test_name"] = "split"
    data["test_info"] = {
        "pvalue": 0.001, "split_r2": 0.3,
        "n_splits": 50, "split_ratio": 0.5, "random_state": None,
    }
    r = PLSResult(**data)
    assert r.test.name == "split"
    assert r.test.pvalue == 0.001
    assert not hasattr(r, "sweep")


def test_pls_preserves_fit_info_and_raw_diagnostics():
    data = _make_continuous()
    data["fit_info"] = {
        "n_components": 3, "pca_k": 50, "p_method": "perm",
        "n_perm": 2000, "random_state": 42,
    }
    data["raw_diagnostics"] = {
        "cv_scores": {1: 0.12, 2: 0.34, 3: 0.47},
        "perm_null": np.zeros(2000),
        "cv_result": "CV_SENTINEL",
    }
    r = PLSResult(**data)
    assert r.fit_info.n_components == 3
    assert r.fit_info.p_method == "perm"
    assert r.fit_info.n_perm == 2000
    assert r.cv_scores == {1: 0.12, 2: 0.34, 3: 0.47}
    assert r.perm_null.shape == (2000,)
    assert r.cv_result == "CV_SENTINEL"


def test_pcaols_preserves_sweep_result_and_plot_sweep_available():
    data = _make_continuous()
    data["backend"] = "PCA+OLS"
    data["sweep"] = [(k, 0.3 + 0.01 * k, 0.3, 0.01) for k in range(1, 6)]
    data["fit_info"] = {"n_components": 64, "k_min": 20, "k_max": 120,
                         "k_step": 2, "best_k": 64}

    class _FakeSweep:
        best_k = 64
        df_joined = [{"PCA_K": k} for k in range(20, 121, 2)]
    data["raw_diagnostics"] = {"sweep_result": _FakeSweep()}
    r = PCAOLSResult(**data)
    assert r.fit_info.best_k == 64
    assert r.sweep_result.best_k == 64
    assert callable(r.plot_sweep)


def test_pcaols_plot_sweep_raises_without_sweep_result():
    data = _make_continuous()
    data["backend"] = "PCA+OLS"
    r = PCAOLSResult(**data)
    with pytest.raises(RuntimeError, match="No sweep data"):
        r.plot_sweep()


def test_attach_embeddings_lets_words_resolve(monkeypatch):
    r = PLSResult(**_make_continuous())

    class _FakeEmb:
        pass

    def _fake_words(self):
        return [Word(side="pos", rank=1, word="x", cos_beta=0.1, contrast=None)]

    monkeypatch.setattr(type(r), "_compute_words_rows", _fake_words)
    r.attach(embeddings=_FakeEmb())
    assert len(r.words) == 1
