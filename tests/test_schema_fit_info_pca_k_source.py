"""Tests for FitInfo.pca_k_source field."""

from __future__ import annotations

import numpy as np
import pytest

from ssdiff.corpus import Corpus
from ssdiff.embeddings import Embeddings
from ssdiff.results.schema import FitInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kv(words, dim=8, seed=42):
    rng = np.random.default_rng(seed)
    mat = rng.normal(size=(len(words), dim)).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    return Embeddings(words, mat)


VOCAB = [
    "kraj", "narod", "panstwo",
    "piekny", "silny", "zly", "dobry",
    "wielki", "maly", "stary", "nowy",
    "dom", "szkola", "praca", "miasto",
    "rzeka", "gora", "las",
    "ABC123", "Warszawa",
]

LEXICON = {"kraj", "narod", "panstwo"}


def _make_docs(n, seed=0):
    seeds = ["kraj", "narod", "panstwo"]
    ctx = ["piekny", "silny", "zly", "dobry", "wielki", "maly",
           "stary", "nowy", "dom", "szkola", "praca", "miasto"]
    rng = np.random.default_rng(seed)
    docs = []
    for i in range(n):
        seed_tok = seeds[i % len(seeds)]
        context = list(rng.choice(ctx, size=3, replace=False))
        docs.append([seed_tok] + context)
    return docs


@pytest.fixture(scope="module")
def ssd_50():
    """SSD instance with 50 docs and continuous y."""
    from ssdiff.ssd import SSD
    rng = np.random.default_rng(1)
    docs = _make_docs(50, seed=1)
    y = rng.normal(size=50)
    emb = _make_kv(VOCAB, dim=8, seed=42)
    corpus = Corpus(docs, pretokenized=True, lang="pl")
    return SSD(emb, corpus, y, LEXICON)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_fit_info_field_exists_with_default_none():
    """FitInfo.pca_k_source exists and defaults to None."""
    fi = FitInfo()
    assert hasattr(fi, "pca_k_source")
    assert fi.pca_k_source is None


def test_fit_info_accepts_sweep_value():
    fi = FitInfo(pca_k_source="sweep")
    assert fi.pca_k_source == "sweep"


def test_fit_info_accepts_fixed_value():
    fi = FitInfo(pca_k_source="fixed")
    assert fi.pca_k_source == "fixed"


def test_pca_ols_sweep_sets_pca_k_source(ssd_50):
    """fit_ols() with no fixed_k (sweep path) → pca_k_source == 'sweep'."""
    result = ssd_50.fit_ols(fixed_k=None, k_min=2, k_max=6, k_step=1)
    assert result.fit_info._info.pca_k_source == "sweep"


def test_pca_ols_fixed_k_sets_pca_k_source(ssd_50):
    """fit_ols() with fixed_k=K → pca_k_source == 'fixed'."""
    result = ssd_50.fit_ols(fixed_k=3)
    assert result.fit_info._info.pca_k_source == "fixed"


def test_pls_pca_k_source_is_none(ssd_50):
    """fit_pls() does not set pca_k_source; it remains None."""
    result = ssd_50.fit_pls(n_components=1, p_method=None)
    assert result.fit_info._info.pca_k_source is None
