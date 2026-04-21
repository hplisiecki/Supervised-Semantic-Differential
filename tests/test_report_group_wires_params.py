"""GroupResult.report() wires top_words / clusters / snippets_per_cluster."""

from __future__ import annotations

import pytest


def test_report_no_kwargs_has_omnibus_and_labels(group_result_2g):
    text = group_result_2g.report(top_words=None).to_text()
    assert "Omnibus" in text
    assert "Group labels" in text  # new section
    assert "Pairwise contrasts" in text


def test_report_bare_call_renders_default_sections(group_result_2g):
    """Spec §5.2: bare report() uses the default top_words=5 and renders the
    standard omnibus + group-labels + pairwise-contrasts + per-pair top-words."""
    text = group_result_2g.report().to_text()
    assert "Omnibus" in text
    assert "Group labels" in text
    assert "Pairwise contrasts" in text
    # No clusters or snippets by default
    assert "— pos" not in text
    assert "Snippets —" not in text


def test_report_omnibus_has_random_state(group_result_2g):
    text = group_result_2g.report(top_words=None).to_text()
    assert "random_state" in text


def test_report_group_labels_section_lists_canonical_to_original(group_result_2g):
    gr = group_result_2g
    text = gr.report(top_words=None).to_text()
    # Each canonical label key should appear
    for canonical in gr.group_labels:
        assert canonical in text


def test_report_top_words_adds_pair_sections(group_result_2g):
    text = group_result_2g.report(top_words=5).to_text()
    # The pair title appears as a section heading
    gr = group_result_2g
    pair = next(iter(gr.pairs))
    pair_title = f"{pair.g1} vs {pair.g2}"
    assert pair_title in text


def test_report_clusters_adds_pos_neg_sections(group_result_2g):
    text = group_result_2g.report(top_words=None, clusters=30).to_text()
    assert "pos" in text
    assert "neg" in text


def test_report_top_words_none_skips_word_sections(group_result_2g):
    gr = group_result_2g
    text = gr.report(top_words=None).to_text()
    # Without top_words, no pair vs. pair word table sections
    # The section headers like "g1 vs g2" should NOT be present
    # (only pairwise contrasts table will have pair info, but as a table body)
    # Check that clusters section isn't there either
    assert "— pos" not in text
    assert "— neg" not in text


def test_report_clusters_none_skips_cluster_sections(group_result_2g):
    text = group_result_2g.report(top_words=5, clusters=None).to_text()
    # With clusters=None, no pos/neg cluster section titles
    assert "— pos" not in text
    assert "— neg" not in text


def test_report_3g_top_words_adds_multiple_pair_sections(group_result_3g):
    gr = group_result_3g
    text = gr.report(top_words=5).to_text()
    # All three pair titles should appear
    for p in gr.pairs:
        pair_title = f"{p.g1} vs {p.g2}"
        assert pair_title in text


def test_report_3g_clusters_adds_pos_neg_per_pair(group_result_3g):
    gr = group_result_3g
    text = gr.report(top_words=None, clusters=30).to_text()
    # 3 pairs × 2 sides = 6 cluster sections
    assert text.count("— pos") == 3
    assert text.count("— neg") == 3


def test_report_snippets_adds_per_pair_section(group_result_2g):
    text = group_result_2g.report(top_words=None, snippets_per_cluster=2).to_text()
    gr = group_result_2g
    pair = next(iter(gr.pairs))
    pair_title = f"{pair.g1} vs {pair.g2}"
    assert f"Snippets — {pair_title}" in text


def test_report_snippets_none_skips_snippet_sections(group_result_2g):
    text = group_result_2g.report(top_words=None, snippets_per_cluster=None).to_text()
    assert "Snippets —" not in text


def test_report_3g_snippets_adds_section_per_pair(group_result_3g):
    text = group_result_3g.report(top_words=None, snippets_per_cluster=2).to_text()
    assert text.count("Snippets —") == 3
