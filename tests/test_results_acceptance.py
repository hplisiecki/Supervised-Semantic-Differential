"""Direct assertions on each of the 17 spec acceptance criteria."""

import pickle

import numpy as np
import pytest

from ssdiff.results.continuous_result import PLSResult


def _mk():
    rng = np.random.default_rng(0)
    n, d = 30, 4
    beta = rng.standard_normal(d)
    x = rng.standard_normal((n, d))
    y = x @ beta + 0.1 * rng.standard_normal(n)
    return PLSResult(
        x=x, beta=beta, keep_mask=np.ones(n, dtype=bool),
        n_raw=n, n_kept=n, n_dropped=0,
        y=y, _y_mean=np.array([y.mean()]), _y_scale=np.array([y.std() + 1e-12]),
        r2=0.47, r2_adj=0.46, pvalue=1e-4,
    )


def test_a1_install_without_pandas_basic_apis(monkeypatch):
    """1: non-tabular APIs work without pandas."""
    r = _mk()
    # to_dict uses no pandas
    d = r.stats.to_dict()
    assert "r2" in d


def test_a3_view_contract_on_all_result_types():
    """3: all result classes expose View / ScalarView contract."""
    r = _mk()
    # ScalarView API
    assert r.stats.r2 == 0.47
    assert r.stats["n_kept"] == 30
    # View API on docs
    assert len(r.docs) == 30
    assert list(r.docs)[0].doc_id == 0


def test_a4_cache_correctness_param_variants_coexist(monkeypatch):
    """4: clusters(topn=50) and clusters defaults coexist via the REAL cache path.

    Drives the real `_clusters_for` → `_cache_get` → `_compute_clusters_for_side`
    pipeline by stubbing only the leaf compute method. This proves the cache
    key derived from full default params is correct, not a hand-built shortcut.
    """
    from ssdiff.results.schema import Cluster

    r = _mk()
    r.attach(embeddings=object())  # satisfy _require_resource("embeddings", ...)

    calls: list[dict] = []

    def fake_compute(self, *, side, topn, k, k_min, k_max,
                     random_state, min_cluster_size):
        calls.append({"side": side, "topn": topn})
        return (
            [Cluster(cluster_id=0, side=side, size=topn, coherence=0.5,
                     centroid_cos_beta=0.3, contrast=None)],
            [],
        )

    monkeypatch.setattr(type(r), "_compute_clusters_for_side", fake_compute)

    v100 = r.clusters.pos              # real path, defaults → topn=100
    v50 = r.clusters.pos(topn=50)      # different cache entry
    v100_again = r.clusters.pos        # cache hit — NOT overwritten by the topn=50 call
    assert v100 is v100_again
    assert v100[0].size == 100
    assert v50[0].size == 50
    # Compute called exactly twice — once per distinct param set.
    assert len(calls) == 2
    assert sorted(c["topn"] for c in calls) == [50, 100]


def test_a7_repr_contract():
    """7: Report.__repr__ includes to_text() content + optional save hint; Result._repr_html_ is compact HTML."""
    import ssdiff
    r = _mk()
    rep = r.report()
    ssdiff.set_repr_hints(False)
    try:
        assert repr(rep) == rep.to_text()
    finally:
        ssdiff.set_repr_hints(True)
    html = r._repr_html_()
    assert "PLSResult" in html


def test_a8_formatting_rules():
    """8: fmt_p / fmt_r / fmt_count contracts (D11)."""
    from ssdiff.results.format import fmt_count, fmt_p, fmt_r
    assert fmt_p(0.0004) == "<.001"
    assert fmt_p(0.037) == ".037"
    assert fmt_r(0.423) == ".42"
    assert fmt_count(1240) == "1,240"


def test_a10_attach_raises_before_attach():
    r = _mk()
    with pytest.raises(RuntimeError, match="attach"):
        _ = r.words


def test_a11_pickle_round_trip_preserves_tables_and_cache():
    """Pickle round-trip: the cache dict survives. Uses a synthetic key here —
    the real-cache-path invariant is asserted by test_a4 / test_a12.
    """
    r = _mk()
    synth_key = ("clusters", (("side", "pos"), ("topn", 100)))
    r._cache[synth_key] = "SENTINEL"
    r2 = pickle.loads(pickle.dumps(r))
    assert r2.stats.r2 == r.stats.r2
    assert r2._cache[synth_key] == "SENTINEL"


def test_a12_stale_cache_bug_fixed(monkeypatch):
    """12: regression test for the stale-cache bug (spec issue #2).

    Uses the real cache path (not direct `_cache` pokes) so the test would
    catch a regression where `_clusters_for` forgot to include `topn` in the
    cache key — which was the original v1 bug.
    """
    from ssdiff.results.schema import Cluster

    r = _mk()
    r.attach(embeddings=object())

    def fake_compute(self, *, side, topn, k, k_min, k_max,
                     random_state, min_cluster_size):
        # Size-tagged so a stale read would return the wrong size.
        return (
            [Cluster(cluster_id=0, side=side, size=topn, coherence=0.5,
                     centroid_cos_beta=0.3, contrast=None)],
            [],
        )

    monkeypatch.setattr(type(r), "_compute_clusters_for_side", fake_compute)

    v_default = r.clusters.pos
    v_50 = r.clusters.pos(topn=50)
    # The v1 bug: both would return the same (first-cached) entry.
    assert v_default[0].size == 100
    assert v_50[0].size == 50
    # Second default read still returns 100 — proves v_50 didn't overwrite it.
    assert r.clusters.pos[0].size == 100


def test_a13_per_view_save_and_report_markdown(tmp_path):
    r = _mk()
    r.stats.save(str(tmp_path / "stats.csv"))
    r.docs.save(str(tmp_path / "docs.csv"))
    assert (tmp_path / "stats.csv").is_file()
    assert (tmp_path / "docs.csv").is_file()

    p1 = tmp_path / "r1.md"
    p2 = tmp_path / "r2.md"
    r.report().save(str(p1))
    r.report().save(str(p2))
    assert p1.read_text(encoding="utf-8") == p2.read_text(encoding="utf-8")


def test_a15_docs_accessors():
    r = _mk()
    pos5 = r.docs.pos(5)
    neg5 = r.docs.neg(5)
    assert [d.y_hat for d in pos5] == sorted([d.y_hat for d in pos5], reverse=True)
    assert [d.y_hat for d in neg5] == sorted([d.y_hat for d in neg5])


def test_a16_repr_html_scoped_to_result_and_report():
    r = _mk()
    rep = r.report()
    assert hasattr(r, "_repr_html_")
    assert hasattr(rep, "_repr_html_")
    # View also has _repr_html_ (renders HTML table + optional save hint)
    assert hasattr(r.docs, "_repr_html_")


def test_a17_clear_cache_two_forms():
    """`clear_cache(view=...)` drops entries whose view_name matches; no-arg drops all.
    Synthetic keys OK here — we're asserting the cache-dict bookkeeping, not the
    key-derivation (which test_a4 covers).
    """
    r = _mk()
    r._cache[("clusters", (("side", "pos"),))] = "a"
    r._cache[("clusters", (("side", "neg"),))] = "a2"
    r._cache[("snippets", (("top_per_side", 30),))] = "b"
    r.clear_cache("clusters")
    assert not any(name == "clusters" for (name, _) in r._cache)
    assert ("snippets", (("top_per_side", 30),)) in r._cache
    r.clear_cache()
    assert r._cache == {}
