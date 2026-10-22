"""Tests for top_words / top_snippet on cluster views.

All tests stay within the in-memory fixtures from conftest.py — no real
embeddings, no spaCy downloads.
"""
from __future__ import annotations

import csv
import dataclasses

import pytest


@pytest.fixture
def fresh_clusters(pls_result):
    """``pls_result`` with its cluster caches empty before and after the test."""
    def clear():
        pls_result.clear_cache("clusters")
        pls_result.clear_cache("cluster_snippets")
    clear()
    yield pls_result
    clear()


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


def test_clusters_view_default_cols(fresh_clusters):
    """Default to_dict on clusters.pos includes top_words; top_snippet is opt-in."""
    rows = fresh_clusters.clusters.pos.to_dict()
    assert rows, "expected at least one cluster row"
    assert "top_words" in rows[0]
    # top_snippet is opt-in only — not in defaults
    assert "top_snippet" not in rows[0]
    # `side` is dropped from sided default
    assert "side" not in rows[0]
    # cols="all" surfaces top_snippet
    all_rows = fresh_clusters.clusters.pos.to_dict(cols="all")
    assert "top_snippet" in all_rows[0]
    assert "top_words" in all_rows[0]
    # flat ClustersView: top_words in defaults, side kept, top_snippet opt-in
    flat_rows = fresh_clusters.clusters.to_dict()
    assert "side" in flat_rows[0]
    assert "top_words" in flat_rows[0]
    assert "top_snippet" not in flat_rows[0]
    assert "top_snippet" in fresh_clusters.clusters.to_dict(cols="all")[0]


def test_top_snippet_empty_without_text(fresh_clusters, monkeypatch):
    """With parent.text = None, top_snippet stays '' even when explicitly requested."""
    monkeypatch.setattr(fresh_clusters, "text", None)
    rows = fresh_clusters.clusters.pos.to_dict(cols="all")
    for r in rows:
        assert r["top_snippet"] == ""
        # top_words is still populated — comes from embeddings only
        assert r["top_words"] != ""


def test_top_snippet_filled_lazy(fresh_clusters):
    """With the text attached and cols='all', top_snippet matches max-cosine text_window per cluster."""
    rows = fresh_clusters.clusters.pos.to_dict(cols="all")
    snippets_pos = list(fresh_clusters.clusters.pos.snippets(k=None))
    for r in rows:
        cluster_snips = [s for s in snippets_pos if s.cluster_id == r["cluster_id"]]
        assert cluster_snips
        best = max(cluster_snips, key=lambda s: s.cosine)
        assert r["top_snippet"] == best.text_window


def test_top_snippet_in_exports(fresh_clusters, tmp_path):
    """to_records keeps the default width; a cols='all' CSV carries top_words and top_snippet."""
    records = fresh_clusters.clusters.pos.to_records()
    assert records and len(records[0]) == len(fresh_clusters.clusters.pos.to_dict()[0])
    p = tmp_path / "clusters.csv"
    fresh_clusters.clusters.pos.save(p, cols="all")
    with open(p, newline="", encoding="utf-8") as fh:
        csv_rows = list(csv.DictReader(fh))
    assert csv_rows
    assert "top_words" in csv_rows[0]
    assert "top_snippet" in csv_rows[0]


def test_zoom_renders_the_snippet_subtable(fresh_clusters):
    """repr(zoomed view) contains the 'Top 5 cluster snippets:' header + a 4-column sub-table."""
    cid = next(iter(fresh_clusters.clusters.pos)).cluster_id
    best = max(
        (s for s in fresh_clusters.clusters.pos.snippets(k=None) if s.cluster_id == cid),
        key=lambda s: s.cosine,
    )
    text = repr(fresh_clusters.clusters.pos(cid))
    assert "Top 5 cluster snippets" in text
    for column in ("seed", "cosine", "doc_id", "text_window"):
        assert column in text
    # the repr clips long windows, so only the start is compared
    assert best.text_window[:30] in text


def test_zoom_no_text_fallback(fresh_clusters, monkeypatch):
    """Zoom repr without a text prints '(attach texts to populate)' rather than raising."""
    monkeypatch.setattr(fresh_clusters, "text", None)
    cids = [c.cluster_id for c in fresh_clusters.clusters.pos]
    assert cids
    text = repr(fresh_clusters.clusters.pos(cids[0]))
    assert "(attach texts to populate)" in text


def test_zoom_top_snippets_kwarg(fresh_clusters):
    """clusters.pos(cid, top_snippets=N) sizes the sub-table to N, not the default 5."""
    cid = next(iter(fresh_clusters.clusters.pos)).cluster_id
    text = repr(fresh_clusters.clusters.pos(cid, top_snippets=3))
    assert "Top 3 cluster snippets" in text
    assert "Top 5 cluster snippets" not in text


def test_zoom_top_snippets_kwarg_requires_cluster_id(pls_result):
    """clusters.pos(top_snippets=5) without a positional cluster_id raises TypeError."""
    with pytest.raises(TypeError, match="top_snippets="):
        pls_result.clusters.pos(top_snippets=5)


def test_zoom_save_writes_one_row(fresh_clusters, tmp_path):
    """save() on a zoomed view writes the 1-row summary, not the sub-table."""
    cids = [c.cluster_id for c in fresh_clusters.clusters.pos]
    assert cids
    zoomed = fresh_clusters.clusters.pos(cids[0])
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


def test_group_pair_clusters_inherit(group_result_2g):
    """PairResult.clusters carries top_words eagerly; top_snippet via cols='all'."""
    pair = list(group_result_2g.pairs)[0]
    pair_result = group_result_2g[(pair.g1, pair.g2)]
    assert len(pair_result.clusters.pos) > 0
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


def test_clear_cache_clusters_refills(fresh_clusters):
    """clear_cache('clusters') drops the view; next opt-in access re-fills identically."""
    before = fresh_clusters.clusters.pos
    rows_before = [(r["cluster_id"], r["top_snippet"]) for r in before.to_dict(cols="all")]
    fresh_clusters.clear_cache("clusters")
    after = fresh_clusters.clusters.pos
    assert after is not before
    assert [(r["cluster_id"], r["top_snippet"]) for r in after.to_dict(cols="all")] == rows_before


def test_zoom_subtable_sees_beyond_default_k(fresh_clusters, monkeypatch):
    """Regression: ``_top_snippets_subtable`` must scan all snippets, not just
    the first ``k=30`` rows visible through the default-sized iterator.

    The cached ``_cluster_snippets_for`` returns a ``SnippetsViewSided`` whose
    public iterator respects its display ``k`` (default 30). Iterating
    that directly would silently drop snippets past the first 30, because
    the iterator respects the display ``k``. Here we hand-craft a view where the target cluster's only
    rows live past index 30, and verify the subtable still finds them.
    """
    from ssdiff.results.continuous_result import SnippetsViewSided
    from ssdiff.results.schema import Snippet

    cids = [c.cluster_id for c in fresh_clusters.clusters.pos]
    assert len(cids) >= 2, "need ≥2 clusters to exercise the buried-cluster case"
    target_cid = cids[-1]

    def _mk(sid: int, cid: int, cos: float) -> Snippet:
        return Snippet(
            snippet_id=sid, side="pos", doc_id=sid, cosine=cos,
            seed="seed", start_token_idx=0, end_token_idx=1,
            start_sent_idx=0, end_sent_idx=1,
            text_window=f"window_{sid}_cid_{cid}",
            text_surface=f"surface_{sid}", text_lemmas=f"lemmas_{sid}",
            cluster_id=cid, contrast=None,
        )

    crowd_cid = cids[0]
    crowded = [_mk(i, crowd_cid, 0.9 - 0.001 * i) for i in range(40)]
    buried = [_mk(40 + i, target_cid, 0.5 - 0.001 * i) for i in range(5)]
    fake_view = SnippetsViewSided(side="pos", all_rows=crowded + buried)

    monkeypatch.setattr(
        type(fresh_clusters), "_cluster_snippets_for",
        lambda self, side, **kw: fake_view,
    )
    zoomed = fresh_clusters.clusters.pos(target_cid)
    sub = zoomed._top_snippets_subtable()
    assert sub is not None
    assert "(no snippets matched this cluster)" not in sub, (
        "expected to find buried-cluster snippets past the default k=30 cap"
    )
    assert "window_40" in sub or "window_41" in sub


def test_combined_clusters_view_renders_table(fresh_clusters):
    """``pls.clusters`` (no .pos/.neg) renders the combined table with both sides."""
    view = fresh_clusters.clusters
    text = repr(view)
    assert "ClustersView" in text
    assert " pos " in text or "pos +" in text
    # Combined table must include the `side` column header so rows disambiguate.
    assert "side" in text
    assert "pos" in text
    assert "neg" in text
    # top_words column dropped from the table header; top_words listed in
    # a "Words:" block keyed by side:cluster_id.
    sample = next(iter(view))
    assert "Words:" in text
    # The label uses `side:cluster_id`, e.g. "[pos:0]"
    assert f"[{sample.side}:{sample.cluster_id}]" in text


def test_sided_clusters_view_words_block(fresh_clusters):
    """``pls.clusters.pos`` lifts top_words out of the table into a Words: block."""
    view = fresh_clusters.clusters.pos
    text = repr(view)
    sample = view[0]
    assert "Words:" in text
    assert f"[{sample.cluster_id}]" in text
    # Words string itself appears under the block, not as a wide column.
    assert sample.top_words in text
    # Explicit cols=... still routes through the classic table path.
    text_with_cols = view.to_text(cols=("cluster_id", "top_words"))
    assert "Words:" not in text_with_cols
    assert "top_words" in text_with_cols


def test_top_snippet_clipped_in_repr_only(fresh_clusters):
    """top_snippet (when opted in via cols=) is clipped in to_text; full text in to_dict."""
    view = fresh_clusters.clusters.pos
    long_text = "x" * 200
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
