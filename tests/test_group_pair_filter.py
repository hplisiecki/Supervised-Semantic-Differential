"""Pair-filter accessor tests for GroupResult.__call__."""

import numpy as np
import pytest


def test_filter_returns_shallow_copy(group_result_3g):
    gr = group_result_3g
    filtered = gr([("g1", "g2")])
    assert filtered is not gr
    assert gr.embeddings is filtered.embeddings
    assert gr.corpus is filtered.corpus
    assert gr.x is filtered.x
    assert gr.groups is filtered.groups


def test_filter_restricts_pairs(group_result_3g):
    gr = group_result_3g
    filtered = gr([("g1", "g2")])
    assert [(p.g1, p.g2) for p in filtered.pairs] == [("g1", "g2")]


def test_filter_downstream_views_inherit(group_result_3g):
    gr = group_result_3g
    filtered = gr([("g1", "g2")])
    from ssdiff.results.continuous_result import WordsView
    from ssdiff.results.paired_view import WordsViewPaired
    # Single-pair filtered result should give a WordsView (single-pair dispatch)
    # while the original multi-pair gives a WordsViewPaired
    assert isinstance(filtered.words, WordsView)
    assert isinstance(gr.words, WordsViewPaired)


def test_filter_accepts_pairs_keyword(group_result_3g):
    gr = group_result_3g
    a = gr([("g1", "g2")])
    b = gr(pairs=[("g1", "g2")])
    assert [(p.g1, p.g2) for p in a.pairs] == [(p.g1, p.g2) for p in b.pairs]


def test_filter_normalizes_reverse_order(group_result_3g):
    gr = group_result_3g
    filtered = gr([("g2", "g1")])
    assert [(p.g1, p.g2) for p in filtered.pairs] == [("g1", "g2")]


def test_filter_unknown_pair_raises(group_result_3g):
    gr = group_result_3g
    with pytest.raises(KeyError):
        gr([("g1", "g99")])


def test_filter_empty_list_raises(group_result_3g):
    gr = group_result_3g
    with pytest.raises(ValueError):
        gr([])


def test_single_pair_noop(group_result_3g):
    gr = group_result_3g
    all_keys = [(p.g1, p.g2) for p in gr.pairs]
    filtered = gr(all_keys)
    assert [(p.g1, p.g2) for p in filtered.pairs] == all_keys


# -- ergonomic __call__ forms --------------------------------------------------

def test_filter_contrast_string(group_result_3g):
    gr = group_result_3g
    filtered = gr("g1_g2")
    assert [(p.g1, p.g2) for p in filtered.pairs] == [("g1", "g2")]


def test_filter_positional_pair(group_result_3g):
    gr = group_result_3g
    filtered = gr("g1", "g2")
    assert [(p.g1, p.g2) for p in filtered.pairs] == [("g1", "g2")]


def test_filter_single_tuple(group_result_3g):
    gr = group_result_3g
    filtered = gr(("g1", "g2"))
    assert [(p.g1, p.g2) for p in filtered.pairs] == [("g1", "g2")]


def test_filter_single_list_as_pair(group_result_3g):
    gr = group_result_3g
    filtered = gr(["g1", "g2"])
    assert [(p.g1, p.g2) for p in filtered.pairs] == [("g1", "g2")]


def test_filter_list_of_contrast_strings(group_result_3g):
    gr = group_result_3g
    filtered = gr(["g1_g2", "g1_g3"])
    assert [(p.g1, p.g2) for p in filtered.pairs] == [("g1", "g2"), ("g1", "g3")]


def test_filter_multiple_positional_contrasts(group_result_3g):
    gr = group_result_3g
    filtered = gr("g1_g2", "g1_g3")
    assert [(p.g1, p.g2) for p in filtered.pairs] == [("g1", "g2"), ("g1", "g3")]


def test_filter_raw_label_resolution(group_result_3g):
    gr = group_result_3g
    raw_by_canon = gr.group_labels
    raw1, raw2 = raw_by_canon["g1"], raw_by_canon["g2"]
    filtered = gr(raw1, raw2)
    assert [(p.g1, p.g2) for p in filtered.pairs] == [("g1", "g2")]


def test_filter_no_underscore_string_raises(group_result_3g):
    gr = group_result_3g
    with pytest.raises(ValueError):
        gr("g1g2")


def test_filter_no_args_raises(group_result_3g):
    gr = group_result_3g
    with pytest.raises(TypeError):
        gr()


def test_filter_mix_args_and_kwarg_raises(group_result_3g):
    gr = group_result_3g
    with pytest.raises(TypeError):
        gr("g1_g2", pairs=[("g1", "g2")])


# -- float-label canonicalization regression ---------------------------------

def test_fit_groups_float_labels_produces_distinct_gradients(
    tiny_kv, large_docs_3x20, lexicon,
):
    """Regression for dtype-mismatch bug in _canonicalize.

    Before fix: float user labels produced uncanonicalized Pair.g1/g2 strings
    that didn't match the relabeled groups array, yielding nan gradients and
    identical top words for every pair.
    """
    from ssdiff.corpus import Corpus
    from ssdiff.ssd import SSD

    y = np.array([1.0] * 20 + [2.0] * 20 + [3.0] * 20)
    corpus = Corpus(large_docs_3x20, pretokenized=True, lang="pl")
    gr = SSD(tiny_kv, corpus, y, lexicon).fit_groups(n_perm=50, random_state=42)

    canonical_keys = [(p.g1, p.g2) for p in gr.pairs]
    assert all(a.startswith("g") and b.startswith("g")
               for a, b in canonical_keys), canonical_keys

    gradients = gr.gradient
    grad_list = list(gradients.values())
    assert all(np.isfinite(g).all() for g in grad_list), \
        "gradients must be finite — nan means the groups mask was empty"
    for i in range(len(grad_list)):
        for j in range(i + 1, len(grad_list)):
            assert not np.allclose(grad_list[i], grad_list[j]), \
                "distinct pairs must produce distinct gradients"
