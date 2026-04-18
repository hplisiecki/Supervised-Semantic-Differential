"""GroupResult + PairView contract."""

import numpy as np
import pytest

from ssdiff.results.group_result import GroupResult, PairView
from ssdiff.results.schema import Pair


def _fake_group():
    # Unequal group sizes so n_g1/n_g2 swap is observable on reverse access.
    pairs = [
        Pair(contrast="A_vs_B", g1="A", g2="B",
             T=1.0, p_raw=0.01, p_corrected=0.02, cohens_d=0.5,
             n_g1=30, n_g2=70, contrast_norm=0.7),
        Pair(contrast="A_vs_C", g1="A", g2="C",
             T=2.0, p_raw=0.001, p_corrected=0.003, cohens_d=0.8,
             n_g1=30, n_g2=80, contrast_norm=0.9),
    ]
    return GroupResult(
        G=3, n_kept=150, n_perm=5000, correction="holm", random_state=42,
        omnibus_T=1.83, omnibus_p=0.001,
        pairs=pairs,
        words_rows=[], cluster_rows=[], cluster_words_rows=[], snippets_rows=[],
        embeddings=None, corpus=None,
    )


def test_omnibus_on_test_view():
    gr = _fake_group()
    assert abs(gr.test.omnibus_T - 1.83) < 1e-9
    assert gr.test.omnibus_p == 0.001
    assert gr.test.pvalue == 0.001


def test_pairs_view_iterates_pair_rows():
    gr = _fake_group()
    assert len(gr.pairs) == 2
    contrasts = {p.contrast for p in gr.pairs}
    assert contrasts == {"A_vs_B", "A_vs_C"}


def test_pair_access_by_tuple_returns_pairview():
    gr = _fake_group()
    pv = gr.pairs["A", "B"]
    assert isinstance(pv, PairView)
    assert pv.contrast == "A_vs_B"


def test_pair_reverse_access_flips_sign():
    """Q-C: symmetric keying with sign-flip on reverse order."""
    gr = _fake_group()
    ab = gr.pairs["A", "B"]
    ba = gr.pairs["B", "A"]
    assert ab.contrast == "A_vs_B"
    assert ba.contrast == "B_vs_A"
    # Sign-flipped fields: T, cohens_d, contrast_norm.
    assert abs(ab.pair.cohens_d - 0.5) < 1e-9
    assert abs(ba.pair.cohens_d + 0.5) < 1e-9
    assert abs(ab.pair.T - 1.0) < 1e-9
    assert abs(ba.pair.T + 1.0) < 1e-9
    assert abs(ba.pair.contrast_norm + 0.7) < 1e-9


def test_pair_reverse_access_swaps_g1_g2_and_n():
    """n_g1 and n_g2 swap on reverse access (Q-C semantics)."""
    gr = _fake_group()
    ab = gr.pairs["A", "B"]
    ba = gr.pairs["B", "A"]
    assert (ab.pair.g1, ab.pair.g2) == ("A", "B")
    assert (ba.pair.g1, ba.pair.g2) == ("B", "A")
    assert (ab.pair.n_g1, ab.pair.n_g2) == (30, 70)
    assert (ba.pair.n_g1, ba.pair.n_g2) == (70, 30)


def test_pair_reverse_access_preserves_pvalues():
    """Two-sided p-values are symmetric — must NOT flip on reverse access."""
    gr = _fake_group()
    ab = gr.pairs["A", "B"]
    ba = gr.pairs["B", "A"]
    assert ab.pair.p_raw == ba.pair.p_raw
    assert ab.pair.p_corrected == ba.pair.p_corrected


def test_pair_view_missing_pair_raises():
    gr = _fake_group()
    with pytest.raises(KeyError, match="no pair"):
        _ = gr.pairs["X", "Y"]


def test_group_stats_scalarview():
    gr = _fake_group()
    assert gr.stats.G == 3
    assert gr.stats.n_kept == 150
    assert gr.stats.n_perm == 5000
    assert gr.stats.correction == "holm"
    assert gr.stats.random_state == 42


def test_pairview_stats_scalarview():
    gr = _fake_group()
    ab = gr.pairs["A", "B"]
    ba = gr.pairs["B", "A"]

    # Forward direction
    assert ab.stats.T == 1.0
    assert abs(ab.stats.cohens_d - 0.5) < 1e-9
    assert ab.stats.p_raw == 0.01
    assert ab.stats.n_g1 == 30

    # Reverse direction — T and cohens_d flipped, p_raw preserved, n_g1/n_g2 swapped
    assert ba.stats.T == -1.0
    assert abs(ba.stats.cohens_d + 0.5) < 1e-9
    assert ba.stats.p_raw == 0.01        # not flipped
    assert ba.stats.n_g1 == 70           # swapped
