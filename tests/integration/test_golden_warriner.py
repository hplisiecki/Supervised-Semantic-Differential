"""Golden regression test: SSD.fit_pls() against frozen Warriner-valence results.

Runs entirely from small committed fixtures (no large embeddings, no spaCy, no
network), so it lives in the normal CI suite — unlike the original heavy
local-only version it replaces. The fixtures are a thin slice of real English
GloVe (PCA'd to a handful of dims), keyed on Warriner-rated words, with valence
as the outcome in lexical-norm mode (``use_full_doc=True``). See
``tests/fixtures/golden_warriner/generate_fixtures.py`` to regenerate them.

Assertions use tolerances rather than bit parity: integer counts are exact
(pure bookkeeping), R² is checked to a relative tolerance, and top words are
compared by set overlap — all robust across the BLAS / OS / Python-version CI
matrix.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from ssdiff import SSD, Corpus, Embeddings

FIX_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures", "golden_warriner")
EMB_PATH = os.path.join(FIX_DIR, "embedding.bin")
CORPUS_PATH = os.path.join(FIX_DIR, "corpus.json")
GOLDEN_PATH = os.path.join(FIX_DIR, "golden.json")

WORD_OVERLAP_MIN = 0.8  # fraction of top words that must match (rank-order is fragile)


@pytest.fixture(scope="module")
def golden_run():
    """Reproduce the frozen PLS fit from committed fixtures."""
    for p in (EMB_PATH, CORPUS_PATH, GOLDEN_PATH):
        if not os.path.exists(p):
            pytest.skip(
                f"golden-warriner fixture missing: {os.path.basename(p)} "
                "(run tests/fixtures/golden_warriner/generate_fixtures.py)"
            )

    with open(CORPUS_PATH, encoding="utf-8") as f:
        cdata = json.load(f)
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        golden = json.load(f)

    emb = Embeddings.load(EMB_PATH)
    corpus = Corpus(cdata["docs"], pretokenized=True, lang="en")
    y = np.asarray(cdata["y"], dtype=float)
    ssd = SSD(emb, corpus, y, use_full_doc=True)
    result = ssd.fit_pls(
        k=golden["k"], n_splits=cdata["n_splits"], random_state=cdata["random_state"],
    )
    return result, golden


def test_counts_exact(golden_run):
    """Document counts are pure bookkeeping — must match exactly."""
    result, golden = golden_run
    assert result.stats.n_raw == golden["n_raw"]
    assert result.stats.n_kept == golden["n_kept"]
    assert result.stats.n_dropped == golden["n_dropped"]
    assert result.fit_info.n_components == golden["n_components"]


def test_r2_within_tolerance(golden_run):
    """R² is deterministic given pinned k; allow BLAS-level cross-platform drift."""
    result, golden = golden_run
    assert result.stats.r2 == pytest.approx(golden["r2"], rel=1e-4, abs=1e-4)


@pytest.mark.parametrize("side", ["pos", "neg"])
def test_top_words_overlap(golden_run, side):
    """Top words match by set overlap — robust to rank reordering near ties."""
    result, golden = golden_run
    expected = golden[f"{side}_words"]
    n_words = len(expected)
    actual = [w.word for w in getattr(result.words, side)(n_words)]
    overlap = len(set(actual) & set(expected))
    threshold = math.ceil(WORD_OVERLAP_MIN * n_words)
    assert overlap >= threshold, (
        f"{side} words overlap {overlap}/{n_words} < {threshold}\n"
        f"  actual:   {actual}\n"
        f"  expected: {expected}"
    )
