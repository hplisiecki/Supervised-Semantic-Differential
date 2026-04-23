"""Tests for canonical group labels in GroupResult.__init__."""

from __future__ import annotations

import re

import numpy as np
import pytest

from ssdiff.corpus import Corpus
from ssdiff.embeddings import Embeddings


# ---------------------------------------------------------------------------
# Helpers (mirrors conftest helpers for self-contained clarity)
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
    """n docs each containing at least one lexicon seed."""
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


# ---------------------------------------------------------------------------
# Fixture: 3-group result with non-alphabetic group names
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def group_result_3_cities():
    """GroupResult fitted with groups ['Warsaw', 'Kraków', 'Berlin'] (20 each)."""
    from ssdiff.ssd import SSD

    n_per_group = 20
    total = n_per_group * 3
    docs = _make_docs(total, seed=7)
    groups = np.array(["Warsaw"] * n_per_group + ["Kraków"] * n_per_group + ["Berlin"] * n_per_group,
                      dtype=object)

    emb = _make_kv(VOCAB, dim=8, seed=42)
    corpus = Corpus(docs, pretokenized=True, lang="pl")
    ssd = SSD(emb, corpus, groups, LEXICON)
    return ssd.fit_groups(n_perm=50, random_state=42)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_group_labels_mapping(group_result_3_cities):
    """Canonical labels map g1→Berlin, g2→Kraków, g3→Warsaw (sorted by str)."""
    gr = group_result_3_cities
    assert gr.group_labels == {"g1": "Berlin", "g2": "Kraków", "g3": "Warsaw"}


def test_groups_array_values_are_canonical(group_result_3_cities):
    """Every value in gr.groups must be a canonical label key."""
    gr = group_result_3_cities
    allowed = set(gr.group_labels.keys())
    for val in gr.groups:
        assert val in allowed, f"Unexpected group value: {val!r}"


def test_pair_contrast_is_canonical(group_result_3_cities):
    """Every Pair.contrast matches the pattern gN_gM."""
    gr = group_result_3_cities
    pattern = re.compile(r"^g\d+_g\d+$")
    for p in gr.pairs:
        assert pattern.match(p.contrast), f"Bad contrast: {p.contrast!r}"


def test_pair_g1_g2_are_canonical_keys(group_result_3_cities):
    """Every Pair.g1 and Pair.g2 must be a key in gr.group_labels."""
    gr = group_result_3_cities
    allowed = set(gr.group_labels.keys())
    for p in gr.pairs:
        assert p.g1 in allowed, f"Pair.g1={p.g1!r} not in group_labels"
        assert p.g2 in allowed, f"Pair.g2={p.g2!r} not in group_labels"


def test_pair_canonical_order(group_result_3_cities):
    """Pair.g1 must lex-sort before Pair.g2 (canonical order)."""
    gr = group_result_3_cities
    for p in gr.pairs:
        assert p.g1 <= p.g2, (
            f"Pair not in canonical order: g1={p.g1!r}, g2={p.g2!r}"
        )


def test_pair_contrast_matches_g1_g2(group_result_3_cities):
    """Pair.contrast must equal f'{g1}_{g2}'."""
    gr = group_result_3_cities
    for p in gr.pairs:
        expected = f"{p.g1}_{p.g2}"
        assert p.contrast == expected, (
            f"Contrast mismatch: {p.contrast!r} != {expected!r}"
        )


def test_canonical_pair_order_with_many_groups():
    """With G>=10, canonical pair order must be numeric, not lexicographic.

    Under lex sort 'g10' < 'g2', so (g2, g10) would wrongly become
    (g10, g2).  Under numeric sort it stays (g2, g10).
    """
    from ssdiff.results.group_result import GroupResult
    from ssdiff.results.schema import Pair

    G = 11
    n_per_group = 2
    labels = [f"orig_{i:02d}" for i in range(G)]   # "orig00" .. "orig10"
    groups = np.array([l for l in labels for _ in range(n_per_group)], dtype=object)
    x = np.random.default_rng(0).standard_normal((len(groups), 4))

    pairs = [
        Pair(
            contrast=f"{a}_{b}",
            g1=a, g2=b,
            T=0.0, p_raw=1.0, p_corrected=1.0, cohens_d=0.0,
            n_g1=n_per_group, n_g2=n_per_group, contrast_norm=0.0,
        )
        for i, a in enumerate(labels) for b in labels[i + 1:]
    ]

    gr = GroupResult(
        G=G, n_kept=len(groups), n_perm=0, correction="none",
        random_state=0, omnibus_T=0.0, omnibus_p=1.0,
        pairs=pairs,
        x=x, groups=groups,
    )

    # g2 vs g10: numeric index 2 < 10, so g2 must come first.
    targets = [p for p in gr.pairs if {p.g1, p.g2} == {"g2", "g10"}]
    assert len(targets) == 1, "Expected exactly one pair involving g2 and g10"
    p = targets[0]
    assert p.g1 == "g2" and p.g2 == "g10", (
        f"Expected g2 before g10 by numeric order, got ({p.g1}, {p.g2})"
    )
