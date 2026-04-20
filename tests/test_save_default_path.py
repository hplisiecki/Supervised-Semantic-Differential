"""Tests for View.save() default path and Path-object acceptance.

Spec §4.3: every view has ``save(path=None, ...)`` where omitting ``path``
writes ``<cwd>/<view_name>.csv`` (or ``<cwd>/<side-stem>.csv`` for sided views).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ssdiff.results.continuous_result import (
    ClustersViewSided,
    SnippetsView,
    SnippetsViewSided,
    WordsView,
    WordsViewSided,
)
from ssdiff.results.schema import Cluster, Snippet, Word


# ---------------------------------------------------------------------------
# Minimal synthetic row helpers (no embeddings needed)
# ---------------------------------------------------------------------------

def _word(side: str, rank: int = 1, word: str = "test") -> Word:
    return Word(side=side, rank=rank, word=word, cos_beta=0.5, contrast=None)


def _snippet(side: str) -> Snippet:
    return Snippet(
        snippet_id=0, side=side, doc_id=0, cosine=0.5, seed="kraj",
        start_token_idx=0, end_token_idx=1, start_sent_idx=0, end_sent_idx=0,
        text_window="test window", text_surface="surface", text_lemmas="lemmas",
        cluster_id=None, post_id=None, contrast=None,
    )


def _cluster(side: str) -> Cluster:
    return Cluster(
        cluster_id=0, side=side, size=3, coherence=0.8,
        centroid_cos_beta=0.6, contrast=None,
    )


# ---------------------------------------------------------------------------
# WordsView — default path
# ---------------------------------------------------------------------------

def test_words_view_default_path(tmp_path, monkeypatch):
    """WordsView.save() with no argument writes ./words.csv."""
    monkeypatch.chdir(tmp_path)
    rows = [_word("pos"), _word("neg")]
    view = WordsView(rows)
    view.save()
    out = tmp_path / "words.csv"
    assert out.exists(), f"Expected {out} to exist"
    content = out.read_text()
    assert "side" in content
    assert "word" in content


# ---------------------------------------------------------------------------
# WordsViewSided — default paths per side
# ---------------------------------------------------------------------------

def test_words_view_sided_pos_default_path(tmp_path, monkeypatch):
    """WordsViewSided (pos) defaults to ./words_pos.csv."""
    monkeypatch.chdir(tmp_path)
    all_rows = [_word("pos", rank=1), _word("pos", rank=2), _word("neg", rank=1)]
    view = WordsViewSided("pos", all_rows)
    view.save()
    out = tmp_path / "words_pos.csv"
    assert out.exists(), f"Expected {out} to exist"


def test_words_view_sided_neg_default_path(tmp_path, monkeypatch):
    """WordsViewSided (neg) defaults to ./words_neg.csv."""
    monkeypatch.chdir(tmp_path)
    all_rows = [_word("pos", rank=1), _word("neg", rank=1)]
    view = WordsViewSided("neg", all_rows)
    view.save()
    out = tmp_path / "words_neg.csv"
    assert out.exists(), f"Expected {out} to exist"


# ---------------------------------------------------------------------------
# SnippetsView — default path
# ---------------------------------------------------------------------------

def test_snippets_view_default_path(tmp_path, monkeypatch):
    """SnippetsView.save() with no argument writes ./snippets.csv."""
    monkeypatch.chdir(tmp_path)
    view = SnippetsView([_snippet("pos"), _snippet("neg")])
    view.save()
    out = tmp_path / "snippets.csv"
    assert out.exists(), f"Expected {out} to exist"
    content = out.read_text()
    assert "side" in content
    assert "cosine" in content


# ---------------------------------------------------------------------------
# SnippetsViewSided — default paths per side
# ---------------------------------------------------------------------------

def test_snippets_view_sided_pos_default_path(tmp_path, monkeypatch):
    """SnippetsViewSided (pos) defaults to ./snippets_pos.csv."""
    monkeypatch.chdir(tmp_path)
    all_rows = [_snippet("pos"), _snippet("neg")]
    view = SnippetsViewSided("pos", all_rows)
    view.save()
    out = tmp_path / "snippets_pos.csv"
    assert out.exists(), f"Expected {out} to exist"


def test_snippets_view_sided_neg_default_path(tmp_path, monkeypatch):
    """SnippetsViewSided (neg) defaults to ./snippets_neg.csv."""
    monkeypatch.chdir(tmp_path)
    all_rows = [_snippet("pos"), _snippet("neg")]
    view = SnippetsViewSided("neg", all_rows)
    view.save()
    out = tmp_path / "snippets_neg.csv"
    assert out.exists(), f"Expected {out} to exist"


# ---------------------------------------------------------------------------
# ClustersViewSided — default paths per side
# ---------------------------------------------------------------------------

def test_clusters_view_sided_pos_default_path(tmp_path, monkeypatch):
    """ClustersViewSided (pos) defaults to ./clusters_pos.csv."""
    monkeypatch.chdir(tmp_path)
    view = ClustersViewSided(
        parent=None, side="pos",
        rows=[_cluster("pos")], words_rows=[], snippets_rows=None, params={},
    )
    view.save()
    out = tmp_path / "clusters_pos.csv"
    assert out.exists(), f"Expected {out} to exist"


def test_clusters_view_sided_neg_default_path(tmp_path, monkeypatch):
    """ClustersViewSided (neg) defaults to ./clusters_neg.csv."""
    monkeypatch.chdir(tmp_path)
    view = ClustersViewSided(
        parent=None, side="neg",
        rows=[_cluster("neg")], words_rows=[], snippets_rows=None, params={},
    )
    view.save()
    out = tmp_path / "clusters_neg.csv"
    assert out.exists(), f"Expected {out} to exist"


# ---------------------------------------------------------------------------
# Extension dispatch still works when path is explicit
# ---------------------------------------------------------------------------

def test_explicit_path_str_works(tmp_path):
    """Passing path='foo.xlsx' still invokes extension dispatch."""
    pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")
    out = tmp_path / "foo.xlsx"
    view = WordsView([_word("pos"), _word("neg")])
    view.save(str(out))
    assert out.exists()


def test_explicit_path_object_works(tmp_path):
    """Passing a Path object as path= is accepted."""
    out = tmp_path / "bar.json"
    view = WordsView([_word("pos"), _word("neg")])
    view.save(out)
    assert out.exists()
    import json
    data = json.loads(out.read_text())
    assert isinstance(data, list)


# ---------------------------------------------------------------------------
# pls_result fixture — real views check their filename stem
# ---------------------------------------------------------------------------

def test_pls_words_default_stem(pls_result, tmp_path, monkeypatch):
    """pls_result.words.save() with no path writes ./words.csv in cwd."""
    monkeypatch.chdir(tmp_path)
    pls_result.words.save()
    assert (tmp_path / "words.csv").exists()


def test_pls_words_pos_default_stem(pls_result, tmp_path, monkeypatch):
    """pls_result.words.pos.save() writes ./words_pos.csv."""
    monkeypatch.chdir(tmp_path)
    pls_result.words.pos.save()
    assert (tmp_path / "words_pos.csv").exists()


def test_pls_words_neg_default_stem(pls_result, tmp_path, monkeypatch):
    """pls_result.words.neg.save() writes ./words_neg.csv."""
    monkeypatch.chdir(tmp_path)
    pls_result.words.neg.save()
    assert (tmp_path / "words_neg.csv").exists()


# ---------------------------------------------------------------------------
# Multi-pair GroupResult: default save() fans out to ./<view_name>/ folder
# ---------------------------------------------------------------------------

def test_multi_pair_words_default_save_creates_folder(group_result_3g, tmp_path, monkeypatch):
    """For N>=2 csv default save, gr.words.save() writes ./words/g_i_vs_g_j.csv."""
    monkeypatch.chdir(tmp_path)
    with pytest.warns(UserWarning, match="fans out"):
        group_result_3g.words.save()
    subfolder = tmp_path / "words"
    assert subfolder.is_dir()
    csv_files = sorted(subfolder.glob("*.csv"))
    assert len(csv_files) == 3  # 3 pairs for G=3
