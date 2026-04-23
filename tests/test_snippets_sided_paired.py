"""Tests for sided snippet access on paired GroupResult.

Post-refactor: gr.snippets is a _ShimView (dict of per-pair SnippetsView).
The old gr.snippets.pos (SnippetsViewSidedPaired) is a removed feature.
Canonical path: gr[pair].snippets.pos (SnippetsViewSided).

# removed: test_snippets_pos_is_sided_paired — gr.snippets.pos no longer exists;
#           gr.snippets is a _ShimView with no .pos attribute.
# removed: test_snippets_neg_is_sided_paired — same reason.
# removed: test_snippets_pos_per_pair_index — was testing SnippetsViewSidedPaired[(pair)]
#           which is a removed class-level feature.
# removed: test_snippets_pos_save_fanout — sided fanout from top-level is removed;
#           use gr[pair].snippets.pos.save() per pair.
"""
import pytest
from ssdiff.results.continuous_result import SnippetsViewSided


def test_pair_snippets_pos_is_sided(group_result_3g):
    """Canonical path: gr[pair].snippets.pos → SnippetsViewSided."""
    gr = group_result_3g
    first_pair = next(iter(gr.pairs))
    pv = gr[(first_pair.g1, first_pair.g2)].snippets
    assert isinstance(pv.pos, SnippetsViewSided)
    assert all(s.side == "pos" for s in pv.pos)


def test_pair_snippets_neg_is_sided(group_result_3g):
    """Canonical path: gr[pair].snippets.neg → SnippetsViewSided."""
    gr = group_result_3g
    first_pair = next(iter(gr.pairs))
    pv = gr[(first_pair.g1, first_pair.g2)].snippets
    assert isinstance(pv.neg, SnippetsViewSided)
    assert all(s.side == "neg" for s in pv.neg)
