"""GroupResult + pairs/top-level view contract (post-pivot)."""

import pytest

from ssdiff.results.group_result import GroupResult
from ssdiff.results.schema import Pair


def _fake_group():
    # Unequal group sizes so they can't be confused.
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
        # groups=None → canonicalization skipped; pairs keep "A"/"B"/"C" labels.
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


def test_pair_access_by_tuple_returns_pair_dataclass():
    gr = _fake_group()
    p = gr.pairs[("A", "B")]
    assert isinstance(p, Pair)
    assert p.contrast == "A_vs_B"


def test_pair_reverse_access_raises_keyerror():
    """Reverse-order lookup raises KeyError — no sign-flip semantics."""
    gr = _fake_group()
    with pytest.raises(KeyError):
        _ = gr.pairs[("B", "A")]


def test_pair_view_missing_pair_raises():
    gr = _fake_group()
    with pytest.raises(KeyError, match="no pair"):
        _ = gr.pairs[("X", "Y")]


def test_group_stats_scalarview():
    gr = _fake_group()
    assert gr.stats.G == 3
    assert gr.stats.n_kept == 150
    assert gr.stats.n_perm == 5000
    assert gr.stats.correction == "holm"
    assert gr.stats.random_state == 42
