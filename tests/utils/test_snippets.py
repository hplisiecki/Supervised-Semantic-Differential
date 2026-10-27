"""Tests for ssdiff/utils/snippets.py

Tests cover:
- _make_snippet_anchor: multi-sentence window
- _top_per_group: top-k per group from sorted rows
- snippets_along_beta: returns pos/neg dicts, cosine signs, sorted descending
- cluster_snippets_by_centroids: 2-cluster case returns 2-keyed structure
"""

from __future__ import annotations

import pytest

from ssdiff.utils.snippets import (
    _DocLike,
    _make_snippet_anchor,
    _top_per_group,
    cluster_snippets_by_centroids,
    snippets_along_beta,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doclike(
    sents_surface: list[str],
    doc_lemmas: list[str],
    token_to_sent: list[int],
    doc_id: int = 0,
) -> _DocLike:
    return _DocLike(
        doc_id=doc_id,
        sents_surface=sents_surface,
        doc_lemmas=doc_lemmas,
        token_to_sent=token_to_sent,
    )


class _FakeSSD:
    """Minimal duck-type satisfying snippets_along_beta and cluster_snippets_by_centroids."""

    def __init__(self, embeddings, gradient, lexicon=None):
        self.embeddings = embeddings
        self.gradient = gradient
        self.beta = gradient  # fallback attr
        self.lexicon = lexicon or []


# ---------------------------------------------------------------------------
# _make_snippet_anchor: multi-sentence window
# ---------------------------------------------------------------------------

class TestMakeSnippetAnchorMultiSentence:
    """multi-sentence window spans across sentence boundaries."""

    def test_window_reaches_previous_sentence(self):
        """When start_tok falls in a previous sentence, prev sentence is prepended."""
        # Doc: sent0 = tokens 0-1, sent1 = tokens 2-3
        D = _make_doclike(
            sents_surface=["Narod jest wielki.", "Kraj jest silny."],
            doc_lemmas=["narod", "jest", "kraj", "silny"],
            token_to_sent=[0, 0, 1, 1],
        )
        # anchor at tok 2 (sent 1), window start at tok 0 (sent 0)
        snippet, s_start, s_end = _make_snippet_anchor(D, i=2, start_tok=0, end_tok=3)
        # Should include previous sentence
        assert "Narod jest wielki." in snippet
        assert "Kraj jest silny." in snippet
        assert s_start == 0
        assert s_end == 1

    def test_window_reaches_next_sentence(self):
        """When end_tok falls in a next sentence, next sentence is appended."""
        D = _make_doclike(
            sents_surface=["Narod jest wielki.", "Kraj jest silny."],
            doc_lemmas=["narod", "jest", "kraj", "silny"],
            token_to_sent=[0, 0, 1, 1],
        )
        # anchor at tok 1 (sent 0), window end at tok 3 (sent 1)
        snippet, s_start, s_end = _make_snippet_anchor(D, i=1, start_tok=0, end_tok=3)
        # Because start_sent == anchor sent (both 0), the code checks next sentence
        assert "Narod jest wielki." in snippet
        assert "Kraj jest silny." in snippet
        assert s_start == 0
        assert s_end == 1

    def test_anchor_in_middle_no_cross_boundary_stays_single(self):
        """If window stays within anchor sentence, returns just that sentence."""
        D = _make_doclike(
            sents_surface=["Narod jest wielki.", "Kraj jest silny."],
            doc_lemmas=["narod", "jest", "kraj", "silny"],
            token_to_sent=[0, 0, 1, 1],
        )
        # anchor at tok 0 (sent 0), window within [0,1] — same sentence
        snippet, s_start, s_end = _make_snippet_anchor(D, i=0, start_tok=0, end_tok=1)
        assert snippet == "Narod jest wielki."
        assert s_start == 0
        assert s_end == 0


# ---------------------------------------------------------------------------
# _top_per_group
# ---------------------------------------------------------------------------

class TestTopPerGroup:
    """_top_per_group returns top-k per group key."""

    def _make_rows(self, items):
        """items: list of (group, cosine) tuples."""
        return [{"centroid_label": g, "cosine": c} for g, c in items]

    def test_top2_per_group_caps_at_available(self):
        """k=2 returns up to 2 per group, even if fewer exist."""
        rows = self._make_rows([
            ("X", 0.9), ("X", 0.7),
            ("Y", 0.6),
        ])
        result = _top_per_group(rows, "centroid_label", 2)
        x_rows = [r for r in result if r["centroid_label"] == "X"]
        y_rows = [r for r in result if r["centroid_label"] == "Y"]
        assert len(x_rows) == 2
        assert len(y_rows) == 1

    def test_preserves_order(self):
        """Input order within group is preserved (assumes pre-sorted by cosine desc)."""
        rows = self._make_rows([
            ("A", 0.95), ("A", 0.80), ("A", 0.50),
        ])
        result = _top_per_group(rows, "centroid_label", 2)
        cosines = [r["cosine"] for r in result]
        assert cosines == [0.95, 0.80]


# ---------------------------------------------------------------------------
# snippets_along_beta
# ---------------------------------------------------------------------------

class TestSnippetsAlongBeta:
    """snippets_along_beta returns k_per_side snippets, sorted."""

    def test_doc_id_points_back_to_the_document(self, sample_preprocessed_docs, tiny_kv):
        """doc_id indexes the document the snippet came from."""
        ssd = _FakeSSD(tiny_kv, tiny_kv.get_vector("kraj", norm=True), lexicon=["kraj", "narod", "panstwo"])
        rows = snippets_along_beta(
            pre_docs=sample_preprocessed_docs, ssd=ssd, token_window=2, top_per_side=200
        )
        assert rows["pos"] and rows["neg"]
        for row in rows["pos"] + rows["neg"]:
            doc = sample_preprocessed_docs[row["doc_id"]]
            assert row["essay_text_lemmas"] == " ".join(doc.doc_lemmas)
            assert row["essay_text_surface"] == " ".join(doc.sents_surface)

    @pytest.mark.parametrize("side", ["pos", "neg"])
    def test_snippets_sorted_by_cosine_descending(self, sample_preprocessed_docs, tiny_kv, side):
        gradient = tiny_kv.get_vector("kraj", norm=True)
        ssd = _FakeSSD(tiny_kv, gradient, lexicon=["kraj", "narod", "panstwo"])
        result = snippets_along_beta(
            pre_docs=sample_preprocessed_docs,
            ssd=ssd,
            token_window=2,
            top_per_side=20,
        )
        rows = result[side]
        assert len(rows) > 1
        cosines = [r["cosine"] for r in rows]
        assert cosines == sorted(cosines, reverse=True)

    def test_one_job_equals_thread_pool(self, sample_preprocessed_docs, tiny_kv):
        """n_jobs=1 runs without threads (the Pyodide path) and finds the same snippets."""
        gradient = tiny_kv.get_vector("kraj", norm=True)
        ssd = _FakeSSD(tiny_kv, gradient, lexicon=["kraj", "narod", "panstwo"])
        kwargs = dict(pre_docs=sample_preprocessed_docs, ssd=ssd, token_window=2, top_per_side=200)
        serial = snippets_along_beta(n_jobs=1, **kwargs)
        pooled = snippets_along_beta(n_jobs=-1, **kwargs)

        def rows(side):
            # Tied cosines may come out of the pool in either order.
            return sorted(repr(sorted(r.items())) for r in side)

        assert serial["pos"] and rows(serial["pos"]) == rows(pooled["pos"])
        assert rows(serial["neg"]) == rows(pooled["neg"])

    def test_pos_neg_cosines_are_negatives_of_each_other(self, sample_preprocessed_docs, tiny_kv):
        """for same occurrence, pos_cosine == -neg_cosine."""
        gradient = tiny_kv.get_vector("kraj", norm=True)
        ssd = _FakeSSD(tiny_kv, gradient, lexicon=["kraj"])
        result = snippets_along_beta(
            pre_docs=sample_preprocessed_docs,
            ssd=ssd,
            token_window=2,
            top_per_side=200,
        )
        pos = result["pos"]
        neg = result["neg"]
        # Both sides should have same number of snippets (no threshold)
        assert len(pos) == len(neg)
        # Sort both by (doc_id, seed) to align matching rows
        pos_sorted = sorted(pos, key=lambda r: (r["doc_id"], r["seed"]))
        neg_sorted = sorted(neg, key=lambda r: (r["doc_id"], r["seed"]))
        for p, n in zip(pos_sorted, neg_sorted):
            assert abs(p["cosine"] + n["cosine"]) < 1e-9

    def test_top_per_side_limits_output(self, sample_preprocessed_docs, tiny_kv):
        """top_per_side caps the number of returned snippets per side."""
        gradient = tiny_kv.get_vector("kraj", norm=True)
        ssd = _FakeSSD(tiny_kv, gradient, lexicon=["kraj", "narod", "panstwo"])
        kwargs = dict(pre_docs=sample_preprocessed_docs, ssd=ssd, token_window=2)
        result = snippets_along_beta(top_per_side=1, **kwargs)
        full = snippets_along_beta(top_per_side=200, **kwargs)
        for side in ("pos", "neg"):
            assert len(result[side]) == 1
            assert result[side][0]["cosine"] == max(r["cosine"] for r in full[side])

    def test_snippet_dict_has_required_keys(self, sample_preprocessed_docs, tiny_kv):
        """Each snippet dict contains the documented keys."""
        gradient = tiny_kv.get_vector("kraj", norm=True)
        ssd = _FakeSSD(tiny_kv, gradient, lexicon=["kraj", "narod", "panstwo"])
        result = snippets_along_beta(
            pre_docs=sample_preprocessed_docs,
            ssd=ssd,
            token_window=2,
            top_per_side=10,
        )
        required_keys = {
            "side", "doc_id", "cosine", "seed",
            "start_token_idx", "end_token_idx", "start_sent_idx", "end_sent_idx",
            "snippet_anchor", "essay_text_surface", "essay_text_lemmas",
        }
        assert result["pos"] and result["neg"]
        for snippet in result["pos"] + result["neg"]:
            assert set(snippet) == required_keys


# ---------------------------------------------------------------------------
# cluster_snippets_by_centroids
# ---------------------------------------------------------------------------

class TestClusterSnippetsByCentroids:
    """2-cluster case returns dict with 2 keys."""

    def test_cluster_snippet_has_centroid_label(self, sample_preprocessed_docs, tiny_kv):
        """Each cluster snippet has a centroid_label key."""
        gradient = tiny_kv.get_vector("kraj", norm=True)
        ssd = _FakeSSD(tiny_kv, gradient, lexicon=["kraj", "narod", "panstwo"])

        pos_clusters = [{"words": [{"word": "kraj"}]}]
        neg_clusters = [{"words": [{"word": "panstwo"}]}]

        result = cluster_snippets_by_centroids(
            pre_docs=sample_preprocessed_docs,
            ssd=ssd,
            pos_clusters=pos_clusters,
            neg_clusters=neg_clusters,
            token_window=2,
            seeds=["kraj", "narod", "panstwo"],
            top_per_cluster=50,
        )
        assert result["pos"] and result["neg"]
        assert {s["centroid_label"] for s in result["pos"]} == {"pos_cluster_1"}
        assert {s["centroid_label"] for s in result["neg"]} == {"neg_cluster_1"}
