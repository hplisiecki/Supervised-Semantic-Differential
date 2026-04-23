"""Item-access tests for GroupResult (post-refactor).

Replaces the old `gr(...)` filter tests. The new API is strict:
- ``gr[('g1', 'g2')]`` → PairResult
- reverse order / unknown pair / non-tuple keys all raise KeyError
- no list / string / positional / raw-label / pairs= forms
"""
import numpy as np
import pytest


def test_item_access_returns_pair_result(group_result_3g):
    from ssdiff.results.group_result import PairResult
    gr = group_result_3g
    pr = gr[("g1", "g2")]
    assert isinstance(pr, PairResult)
    assert (pr.g1, pr.g2) == ("g1", "g2")


def test_item_access_x_contains_correct_rows(group_result_3g):
    gr = group_result_3g
    pr = gr[("g1", "g2")]
    # pr.x is a boolean-masked slice of gr._x — cannot share memory (masked copies),
    # but must contain exactly the rows belonging to g1 or g2.
    mask = (gr._groups == "g1") | (gr._groups == "g2")
    expected = gr._x[mask]
    np.testing.assert_array_equal(pr.x, expected)


def test_item_access_reverse_order_raises(group_result_3g):
    gr = group_result_3g
    with pytest.raises(KeyError, match="canonical order"):
        gr[("g2", "g1")]


def test_item_access_unknown_pair_raises(group_result_3g):
    gr = group_result_3g
    with pytest.raises(KeyError, match="unknown pair"):
        gr[("g1", "g99")]


def test_item_access_string_raises(group_result_3g):
    gr = group_result_3g
    with pytest.raises(KeyError, match="tuple of two strings"):
        gr["g1_g2"]


def test_item_access_list_raises(group_result_3g):
    gr = group_result_3g
    with pytest.raises(KeyError):
        gr[[("g1", "g2"), ("g1", "g3")]]


def test_item_access_positional_pair_raises(group_result_3g):
    """The old `gr('g1', 'g2')` form is gone — gr is not callable."""
    gr = group_result_3g
    with pytest.raises(TypeError):
        gr("g1", "g2")


def test_pair_result_views_work(group_result_3g):
    from ssdiff.results.continuous_result import WordsView
    gr = group_result_3g
    pr = gr[("g1", "g2")]
    # Leaf exposes single-gradient WordsView directly (not a shim).
    if gr.embeddings is not None:
        assert isinstance(pr.words, WordsView)


def test_shim_access_keyed(group_result_3g):
    """Sanity-check the power-user shortcut: gr.words[pair]."""
    from ssdiff.results.continuous_result import WordsView
    gr = group_result_3g
    if gr.embeddings is not None:
        assert isinstance(gr.words[("g1", "g2")], WordsView)


# -- float-label canonicalization regression (unchanged from old test) -------

def test_fit_groups_float_labels_produces_distinct_gradients(
    tiny_kv, large_docs_3x20, lexicon,
):
    """Regression for dtype-mismatch bug in _canonicalize.

    Float user labels must canonicalize to ``g1, g2, …`` cleanly, producing
    finite distinct gradients across all pairs.
    """
    from ssdiff.corpus import Corpus
    from ssdiff.ssd import SSD

    y = np.array([1.0] * 20 + [2.0] * 20 + [3.0] * 20)
    corpus = Corpus(large_docs_3x20, pretokenized=True, lang="pl")
    gr = SSD(tiny_kv, corpus, y, lexicon).fit_groups(n_perm=50, random_state=42)

    canonical_keys = [(p.g1, p.g2) for p in gr.pairs]
    assert all(a.startswith("g") and b.startswith("g")
               for a, b in canonical_keys), canonical_keys

    gradients = gr.gradient  # always dict now
    assert isinstance(gradients, dict)
    grad_list = list(gradients.values())
    assert all(np.isfinite(g).all() for g in grad_list)
    for i in range(len(grad_list)):
        for j in range(i + 1, len(grad_list)):
            assert not np.allclose(grad_list[i], grad_list[j])
