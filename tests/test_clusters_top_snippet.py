"""Tests for top_words / top_snippet on cluster views.

All tests stay within the in-memory fixtures from conftest.py — no real
embeddings, no spaCy downloads. Run with `-m "not slow"`.
"""
from __future__ import annotations

import pytest

from ssdiff.results.schema import Cluster


def test_cluster_dataclass_has_new_fields():
    """Cluster dataclass exposes top_words and top_snippet, both defaulting to ''."""
    c = Cluster(
        cluster_id=0, side="pos", size=3,
        coherence=0.5, centroid_cos_beta=0.4,
    )
    assert c.top_words == ""
    assert c.top_snippet == ""
    # Override
    c2 = Cluster(
        cluster_id=1, side="neg", size=2,
        coherence=0.3, centroid_cos_beta=-0.2,
        top_words="alpha, beta", top_snippet="...some text...",
    )
    assert c2.top_words == "alpha, beta"
    assert c2.top_snippet == "...some text..."


def test_top_words_populated_at_construction(pls_result):
    """Every cluster row carries a non-empty top_words string of <=5 tokens."""
    for c in pls_result.clusters.pos:
        # Always populated when embeddings are attached
        assert isinstance(c.top_words, str)
        assert c.top_words != ""
        parts = c.top_words.split(", ")
        assert 1 <= len(parts) <= 5
        # Order must match the words view (which is cos_centroid desc)
        cluster_words = [
            w.word for w in pls_result.clusters.pos.words
            if w.cluster_id == c.cluster_id
        ][:5]
        assert parts == cluster_words


def test_clusters_view_columns_include_new_fields():
    """ClustersViewSided.columns and ClustersView.columns expose top_words/top_snippet."""
    from ssdiff.results.continuous_result import ClustersView, ClustersViewSided
    assert "top_words" in ClustersViewSided._columns
    assert "top_snippet" in ClustersViewSided._columns
    assert "top_words" in ClustersView._columns
    assert "top_snippet" in ClustersView._columns


def test_clusters_view_default_cols(pls_result):
    """Default to_dict on clusters.pos includes top_words; top_snippet is opt-in."""
    rows = pls_result.clusters.pos.to_dict()
    assert rows, "expected at least one cluster row"
    assert "top_words" in rows[0]
    # top_snippet is opt-in only — not in defaults
    assert "top_snippet" not in rows[0]
    # `side` is dropped from sided default
    assert "side" not in rows[0]
    # cols="all" surfaces top_snippet
    all_rows = pls_result.clusters.pos.to_dict(cols="all")
    assert "top_snippet" in all_rows[0]
    # flat ClustersView: top_words in defaults, side kept, top_snippet opt-in
    flat_rows = pls_result.clusters.to_dict()
    assert "side" in flat_rows[0]
    assert "top_words" in flat_rows[0]
    assert "top_snippet" not in flat_rows[0]


def test_top_snippet_empty_without_corpus(pls_result):
    """With parent.corpus = None, top_snippet stays '' even when explicitly requested."""
    saved = pls_result.corpus
    pls_result.corpus = None
    try:
        pls_result.clear_cache("clusters")
        # Opt in to top_snippet — fill should still no-op (no corpus)
        rows = pls_result.clusters.pos.to_dict(cols="all")
        for r in rows:
            assert r["top_snippet"] == ""
            # top_words is still populated — comes from embeddings only
            assert r["top_words"] != ""
    finally:
        pls_result.corpus = saved
        pls_result.clear_cache("clusters")


def test_top_snippet_filled_lazy(pls_result):
    """With corpus attached and cols='all', top_snippet matches max-cosine text_window per cluster."""
    pls_result.clear_cache("clusters")
    pls_result.clear_cache("cluster_snippets")
    try:
        # Opt-in path triggers fill
        rows = pls_result.clusters.pos.to_dict(cols="all")
        snippets_pos = list(pls_result.clusters.pos.snippets(k=None))
        for r in rows:
            cluster_snips = [s for s in snippets_pos if s.cluster_id == r["cluster_id"]]
            if not cluster_snips:
                assert r["top_snippet"] == ""
                continue
            best = max(cluster_snips, key=lambda s: s.cosine)
            assert r["top_snippet"] == best.text_window
    finally:
        # Drop the mutated cached view so later tests get a clean fixture
        pls_result.clear_cache("clusters")


def test_top_snippet_in_exports(pls_result):
    """to_dict default includes top_words; cols='all' surfaces top_snippet and triggers fill."""
    import csv
    import tempfile
    from pathlib import Path
    pls_result.clear_cache("clusters")
    # Default cols: top_words present, top_snippet absent (opt-in)
    rows = pls_result.clusters.pos.to_dict()
    assert "top_words" in rows[0]
    assert "top_snippet" not in rows[0]
    # cols="all" surfaces top_snippet (and triggers lazy fill)
    all_rows = pls_result.clusters.pos.to_dict(cols="all")
    assert "top_snippet" in all_rows[0]
    assert "top_words" in all_rows[0]
    # to_records — width matches default to_dict
    records = pls_result.clusters.pos.to_records()
    assert records and len(records[0]) == len(rows[0])
    # CSV save with cols="all" writes both new columns
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "clusters.csv"
        pls_result.clusters.pos.save(p, cols="all")
        with open(p, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            csv_rows = list(reader)
        assert csv_rows
        assert "top_words" in csv_rows[0]
        assert "top_snippet" in csv_rows[0]


def test_zoom_renders_top10_snippets(pls_result):
    """repr(zoomed view) contains the 'Top 5 cluster snippets:' header + a 4-column sub-table."""
    pls_result.clear_cache("clusters")
    pls_result.clear_cache("cluster_snippets")
    cids = [c.cluster_id for c in pls_result.clusters.pos]
    assert cids, "expected at least one cluster"
    # Check whether the fixture actually produces any snippets for any cluster.
    all_snips = list(pls_result.clusters.pos.snippets(k=None))
    zoomed = pls_result.clusters.pos(cids[0])
    text = repr(zoomed)
    assert "Top 5 cluster snippets" in text
    if not all_snips:
        # Tiny fixture produces no snippets — sub-table still appears but says so.
        assert "(no snippets matched this cluster)" in text
    else:
        # Header line contains the four columns only when rows are present
        assert "seed" in text and "cosine" in text and "doc_id" in text and "text_window" in text


def test_zoom_no_corpus_fallback(pls_result):
    """Zoom repr without a corpus prints '(attach corpus to populate)' rather than raising."""
    saved = pls_result.corpus
    pls_result.corpus = None
    try:
        pls_result.clear_cache("clusters")
        cids = [c.cluster_id for c in pls_result.clusters.pos]
        if not cids:
            pytest.skip("no clusters on this fixture")
        zoomed = pls_result.clusters.pos(cids[0])
        text = repr(zoomed)
        assert "(attach corpus to populate)" in text
    finally:
        pls_result.corpus = saved
        pls_result.clear_cache("clusters")


def test_zoom_top_snippets_kwarg(pls_result):
    """clusters.pos(cid, top_snippets=N) builds an N-row sub-table when ≥N snippets exist."""
    pls_result.clear_cache("clusters")
    pls_result.clear_cache("cluster_snippets")
    try:
        cids = [c.cluster_id for c in pls_result.clusters.pos]
        if not cids:
            pytest.skip("no clusters")
        cid = cids[0]
        # Count actual available snippets in this cluster
        all_snips = [
            s for s in pls_result.clusters.pos.snippets(k=None)
            if s.cluster_id == cid
        ]
        if not all_snips:
            pytest.skip("no snippets for this cluster on the fixture")
        n_request = min(5, len(all_snips))
        zoomed = pls_result.clusters.pos(cid, top_snippets=n_request)
        # Count rows via the helper that builds the sub-table — avoids parsing
        # repr text or being coupled to save-hint phrasing.
        sub = zoomed._top_snippets_subtable()
        assert sub is not None, "expected a sub-table on zoomed view"
        assert sub.startswith(f"Top {n_request} cluster snippets:")
        # Sanity: the header in the rendered repr also references the chosen N
        assert f"Top {n_request} cluster snippets" in repr(zoomed)
    finally:
        pls_result.clear_cache("clusters")
        pls_result.clear_cache("cluster_snippets")


def test_zoom_top_snippets_kwarg_requires_cluster_id(pls_result):
    """clusters.pos(top_snippets=5) without a positional cluster_id raises TypeError."""
    with pytest.raises(TypeError, match="top_snippets="):
        pls_result.clusters.pos(top_snippets=5)


def test_zoom_save_writes_one_row(pls_result, tmp_path):
    """save() on a zoomed view writes the 1-row summary, not the sub-table."""
    import csv
    pls_result.clear_cache("clusters")
    try:
        cids = [c.cluster_id for c in pls_result.clusters.pos]
        if not cids:
            pytest.skip("no clusters")
        zoomed = pls_result.clusters.pos(cids[0])
        # Default save: top_words present, top_snippet excluded (opt-in)
        p_default = tmp_path / "zoom_default.csv"
        zoomed.save(p_default)
        with open(p_default, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1
        assert "top_words" in rows[0]
        assert "top_snippet" not in rows[0]
        # cols="all" surfaces top_snippet (and triggers fill)
        p_all = tmp_path / "zoom_all.csv"
        zoomed.save(p_all, cols="all")
        with open(p_all, newline="", encoding="utf-8") as fh:
            rows_all = list(csv.DictReader(fh))
        assert len(rows_all) == 1
        assert "top_words" in rows_all[0] and "top_snippet" in rows_all[0]
    finally:
        pls_result.clear_cache("clusters")


def test_group_pair_clusters_inherit(group_result_2g):
    """PairResult.clusters carries top_words eagerly; top_snippet via cols='all'."""
    pair = list(group_result_2g.pairs)[0]
    pair_result = group_result_2g[(pair.g1, pair.g2)]
    if len(pair_result.clusters.pos) == 0:
        pytest.skip("synthetic fixture produced no positive-side clusters")
    # top_words populated at construction
    for c in pair_result.clusters.pos:
        assert c.top_words != "", "PairResult cluster should have top_words populated"
    # Default cols: top_words yes, top_snippet no
    rows = pair_result.clusters.pos.to_dict()
    assert "top_words" in rows[0]
    assert "top_snippet" not in rows[0]
    # Opt-in surfaces top_snippet
    rows_all = pair_result.clusters.pos.to_dict(cols="all")
    assert "top_snippet" in rows_all[0]


def test_clear_cache_clusters_refills(pls_result):
    """clear_cache('clusters') drops the view; next opt-in access re-fills identically."""
    pls_result.clear_cache("clusters")
    try:
        rows_before = [
            (r["cluster_id"], r["top_snippet"])
            for r in pls_result.clusters.pos.to_dict(cols="all")
        ]
        pls_result.clear_cache("clusters")
        rows_after = [
            (r["cluster_id"], r["top_snippet"])
            for r in pls_result.clusters.pos.to_dict(cols="all")
        ]
        assert rows_before == rows_after
    finally:
        pls_result.clear_cache("clusters")


def test_zoom_subtable_sees_beyond_default_k(pls_result):
    """Regression: ``_top_snippets_subtable`` must scan all snippets, not just
    the first ``k=30`` rows visible through the default-sized iterator.

    The cached ``_cluster_snippets_for`` returns a ``SnippetsViewSided`` whose
    public iterator respects its display ``k`` (default 30). Earlier code
    iterated that directly, silently dropping snippets that didn't land in
    the first 30. Here we hand-craft a view where the target cluster's only
    rows live past index 30, and verify the subtable still finds them.
    """
    from ssdiff.results.continuous_result import SnippetsViewSided
    from ssdiff.results.schema import Snippet

    pls_result.clear_cache("clusters")
    pls_result.clear_cache("cluster_snippets")
    cids = [c.cluster_id for c in pls_result.clusters.pos]
    if len(cids) < 2:
        pytest.skip("need ≥2 clusters to exercise the buried-cluster case")
    target_cid = cids[-1]

    def _mk(sid: int, cid: int, cos: float) -> Snippet:
        return Snippet(
            snippet_id=sid, side="pos", doc_id=sid, cosine=cos,
            seed="seed", start_token_idx=0, end_token_idx=1,
            start_sent_idx=0, end_sent_idx=1,
            text_window=f"window_{sid}_cid_{cid}",
            text_surface=f"surface_{sid}", text_lemmas=f"lemmas_{sid}",
            cluster_id=cid, contrast=None, post_id=None,
        )

    crowd_cid = cids[0]
    crowded = [_mk(i, crowd_cid, 0.9 - 0.001 * i) for i in range(40)]
    buried = [_mk(40 + i, target_cid, 0.5 - 0.001 * i) for i in range(5)]
    fake_view = SnippetsViewSided(side="pos", all_rows=crowded + buried)

    real = pls_result._cluster_snippets_for
    pls_result._cluster_snippets_for = lambda side, **kw: fake_view
    try:
        zoomed = pls_result.clusters.pos(target_cid)
        sub = zoomed._top_snippets_subtable()
        assert sub is not None
        assert "(no snippets matched this cluster)" not in sub, (
            "expected to find buried-cluster snippets past the default k=30 cap"
        )
        assert "window_40" in sub or "window_41" in sub
    finally:
        pls_result._cluster_snippets_for = real
        pls_result.clear_cache("clusters")
        pls_result.clear_cache("cluster_snippets")


def test_combined_clusters_view_renders_table(pls_result):
    """``pls.clusters`` (no .pos/.neg) renders the combined table with both sides."""
    pls_result.clear_cache("clusters")
    view = pls_result.clusters
    text = repr(view)
    assert "ClustersView" in text
    assert " pos " in text or "pos +" in text
    # Combined table must include the `side` column header so rows disambiguate.
    assert "side" in text
    # Both side labels appear somewhere in body rows.
    n_pos = len(pls_result.clusters.pos)
    n_neg = len(pls_result.clusters.neg)
    if n_pos:
        assert "pos" in text
    if n_neg:
        assert "neg" in text
    # top_words column dropped from the table header; top_words listed in
    # a "Words:" block keyed by side:cluster_id.
    sample = next(iter(view), None)
    if sample is not None and sample.top_words:
        assert "Words:" in text
        # The label uses `side:cluster_id`, e.g. "[pos:0]"
        assert f"[{sample.side}:{sample.cluster_id}]" in text


def test_sided_clusters_view_words_block(pls_result):
    """``pls.clusters.pos`` lifts top_words out of the table into a Words: block."""
    pls_result.clear_cache("clusters")
    view = pls_result.clusters.pos
    if len(view) == 0:
        pytest.skip("no positive-side clusters on this fixture")
    text = repr(view)
    sample = view[0]
    if sample.top_words:
        assert "Words:" in text
        assert f"[{sample.cluster_id}]" in text
        # Words string itself appears under the block, not as a wide column.
        assert sample.top_words in text
    # Explicit cols=... still routes through the classic table path.
    text_with_cols = view.to_text(cols=("cluster_id", "top_words"))
    assert "Words:" not in text_with_cols
    assert "top_words" in text_with_cols


def test_top_snippet_clipped_in_repr_only(pls_result):
    """top_snippet (when opted in via cols=) is clipped in to_text; full text in to_dict."""
    import dataclasses
    pls_result.clear_cache("clusters")
    try:
        view = pls_result.clusters.pos
        long_text = "x" * 200
        # Inject directly into cached view's rows — finally drops the cache.
        view._rows = (
            [dataclasses.replace(view._rows[0], top_snippet=long_text)]
            + list(view._rows[1:])
        )
        view._top_snippet_filled = True  # avoid re-fill overwriting our injection
        cols = ("cluster_id", "top_words", "top_snippet")
        text = view.to_text(cols=cols)
        # Clipped in to_text rendering
        assert long_text not in text
        # Full text preserved in to_dict (opt-in cols)
        assert any(r.get("top_snippet") == long_text for r in view.to_dict(cols=cols))
        # top_words is NOT clipped — confirm a long top_words survives
        long_words = "y" * 200
        view._rows[0] = dataclasses.replace(view._rows[0], top_words=long_words)
        assert long_words in view.to_text(cols=cols)
    finally:
        pls_result.clear_cache("clusters")
