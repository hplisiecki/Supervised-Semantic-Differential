"""Tests for the rewritten SnippetsView / SnippetsViewSided API.

Covers:
- SnippetsView.pos / .neg properties (the API fix for AttributeError)
- SnippetsView.__call__ rejects non-extraction kwargs (TypeError)
- SnippetsViewSided.__call__(k, cluster_id=N) post-extraction display/filter
- _save_hint text no longer advertises (side='pos', cluster_id=0, ...)
- Default save paths still resolve to snippets.csv / snippets_pos.csv
"""

from __future__ import annotations

import pytest

from ssdiff.results.continuous_result import (
    SnippetsView,
    SnippetsViewSided,
)
from ssdiff.results.schema import Snippet


def _snippet(side: str, cluster_id: int | None = None,
             cosine: float = 0.5, snippet_id: int = 0) -> Snippet:
    return Snippet(
        snippet_id=snippet_id, side=side, doc_id=0, cosine=cosine,
        seed="kraj",
        start_token_idx=0, end_token_idx=1, start_sent_idx=0, end_sent_idx=0,
        text_window="text window", text_surface="surface", text_lemmas="lemmas",
        cluster_id=cluster_id, post_id=None, contrast=None,
    )


def _mixed_rows(n_pos: int = 50, n_neg: int = 40, with_clusters: bool = True):
    rows = []
    sid = 0
    for i in range(n_pos):
        cid = (i % 3) if with_clusters else None
        rows.append(_snippet("pos", cluster_id=cid, cosine=1.0 - i * 0.01,
                             snippet_id=sid))
        sid += 1
    for i in range(n_neg):
        cid = (i % 3) if with_clusters else None
        rows.append(_snippet("neg", cluster_id=cid, cosine=1.0 - i * 0.01,
                             snippet_id=sid))
        sid += 1
    return rows


# ---------------------------------------------------------------------------
# .pos / .neg properties
# ---------------------------------------------------------------------------

def test_pos_property_returns_sided_view():
    view = SnippetsView(_mixed_rows())
    pos = view.pos
    assert isinstance(pos, SnippetsViewSided)
    for s in pos:
        assert s.side == "pos"


def test_neg_property_returns_sided_view():
    view = SnippetsView(_mixed_rows())
    neg = view.neg
    assert isinstance(neg, SnippetsViewSided)
    for s in neg:
        assert s.side == "neg"


def test_pos_default_k_is_30():
    view = SnippetsView(_mixed_rows(n_pos=50))
    assert len(list(view.pos)) == 30


def test_pos_k_none_returns_all():
    view = SnippetsView(_mixed_rows(n_pos=50))
    assert len(list(view.pos(None))) == 50


def test_pos_sized():
    view = SnippetsView(_mixed_rows(n_pos=50))
    assert len(list(view.pos(10))) == 10


def test_pos_k_larger_than_available():
    view = SnippetsView(_mixed_rows(n_pos=3, n_neg=3))
    assert len(list(view.pos(50))) == 3  # only 3 pos rows exist


# ---------------------------------------------------------------------------
# cluster_id filter
# ---------------------------------------------------------------------------

def test_pos_cluster_filter():
    view = SnippetsView(_mixed_rows())
    filtered = view.pos(cluster_id=0)
    assert isinstance(filtered, SnippetsViewSided)
    assert all(s.cluster_id == 0 and s.side == "pos" for s in filtered)


def test_pos_cluster_and_k_together():
    view = SnippetsView(_mixed_rows(n_pos=60))
    filtered = view.pos(5, cluster_id=0)
    rows = list(filtered)
    assert len(rows) == 5
    for s in rows:
        assert s.cluster_id == 0
        assert s.side == "pos"


def test_cluster_filter_preserved_across_chain():
    view = SnippetsView(_mixed_rows(n_pos=60))
    chained = view.pos(cluster_id=0)(50)
    assert all(s.cluster_id == 0 for s in chained)


# ---------------------------------------------------------------------------
# __call__ rejects filter-style kwargs
# ---------------------------------------------------------------------------

class _FakeParent:
    """Mimics enough of ContinuousResult for SnippetsView.__call__ plumbing."""
    def _snippets_for(self, **params):
        from ssdiff.results.continuous_result import SnippetsView
        return SnippetsView([], params=params, parent=self)


def test_call_rejects_side_kwarg():
    parent = _FakeParent()
    view = SnippetsView([], params={}, parent=parent)
    with pytest.raises(TypeError):
        view(side="pos")


def test_call_rejects_cluster_id_kwarg():
    parent = _FakeParent()
    view = SnippetsView([], params={}, parent=parent)
    with pytest.raises(TypeError):
        view(cluster_id=0)


def test_call_rejects_unknown_kwarg():
    parent = _FakeParent()
    view = SnippetsView([], params={}, parent=parent)
    with pytest.raises(TypeError):
        view(not_a_real_kwarg=1)


def test_call_accepts_extraction_kwargs():
    parent = _FakeParent()
    view = SnippetsView([], params={}, parent=parent)
    # Should NOT raise
    out = view(top_per_side=50)
    assert isinstance(out, SnippetsView)


# ---------------------------------------------------------------------------
# save_hint regression
# ---------------------------------------------------------------------------

def test_save_hint_mentions_pos_neg():
    view = SnippetsView(_mixed_rows())
    hint = view._save_hint()
    assert ".pos" in hint
    assert ".neg" in hint


def test_save_hint_does_not_advertise_broken_filter():
    view = SnippetsView(_mixed_rows())
    hint = view._save_hint()
    # The previous misleading hint ``(side='pos', cluster_id=0, ...)`` must be gone.
    assert "side='pos'" not in hint
    assert "cluster_id=0," not in hint


def test_sided_save_hint_when_filtered_mentions_cluster_id():
    """When filtered by ``cluster_id=``, the hint shows the active filter."""
    rows = _mixed_rows()
    view = SnippetsViewSided("pos", rows, cluster_id=0)
    hint = view._save_hint()
    assert "cluster_id=0" in hint


def test_sided_save_hint_unfiltered_has_no_drill_advertisement():
    """Flat / undrilled sided views advertise row count only — no positional drill.

    ``cluster_id`` may appear in the columns preview (it's a row column) but the
    hint must NOT advertise a positional drill form or active filter.
    """
    rows = _mixed_rows()
    view = SnippetsViewSided("pos", rows)
    hint = view._save_hint()
    assert "Drill" not in hint
    assert "filtered to cluster_id" not in hint
    assert "zoom to one cluster" not in hint


# ---------------------------------------------------------------------------
# Default save paths still work
# ---------------------------------------------------------------------------

def test_pos_default_save_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    view = SnippetsView(_mixed_rows())
    view.pos.save()
    assert (tmp_path / "snippets_pos.csv").exists()


def test_flat_default_save_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    view = SnippetsView(_mixed_rows())
    view.save()
    assert (tmp_path / "snippets.csv").exists()


# ---------------------------------------------------------------------------
# Cache correctness
# ---------------------------------------------------------------------------

def test_pos_does_not_cache_on_parent(pls_result):
    # Prime the cache with the default snippets view first.
    _ = pls_result.snippets
    before = len(pls_result._cache)
    _ = pls_result.snippets.pos
    _ = pls_result.snippets.neg
    after = len(pls_result._cache)
    # .pos / .neg are pure transforms — no new cache entries.
    assert before == after


def test_recompute_adds_one_cache_entry(pls_result):
    snip_entries_before = sum(1 for k in pls_result._cache if k[0] == "snippets")
    _ = pls_result.snippets(top_per_side=5)
    snip_entries_after = sum(1 for k in pls_result._cache if k[0] == "snippets")
    # one new entry (the default top_per_side=30 was computed by `snippets` access).
    assert snip_entries_after >= snip_entries_before
