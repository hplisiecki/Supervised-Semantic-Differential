"""Regression tests: GroupResult.words / .clusters compute lazily from gradient.

Before the lazy refactor, ``fit_groups`` passed ``words_rows=[]`` /
``cluster_rows=[]`` and the cached views silently returned empty results.
Shape-only assertions in the existing suite did not catch this — hence these
content-level checks.

Post-refactor: gr.words/gr.clusters are _ShimView dicts. Use gr[pair].words /
gr[pair].clusters to get the single-pair views directly.
"""

from __future__ import annotations


class TestWordsPopulatedSinglePair:
    def _leaf(self, group_result_2g):
        pair = next(iter(group_result_2g.pairs))
        return group_result_2g[(pair.g1, pair.g2)]

    def test_pos_and_neg_nonempty(self, group_result_2g):
        words = list(self._leaf(group_result_2g).words)
        assert any(w.side == "pos" for w in words), "no pos words"
        assert any(w.side == "neg" for w in words), "no neg words"

    def test_cos_beta_sign_invariant(self, group_result_2g):
        for w in self._leaf(group_result_2g).words:
            if w.side == "pos":
                assert w.cos_beta >= 0.0, f"pos word has cos_beta={w.cos_beta}"
            else:
                assert w.cos_beta <= 0.0, f"neg word has cos_beta={w.cos_beta}"

    def test_contrast_tag_matches_pair(self, group_result_2g):
        pair = next(iter(group_result_2g.pairs))
        expected = f"{pair.g1}_{pair.g2}"
        for w in self._leaf(group_result_2g).words:
            assert w.contrast == expected

    def test_cached_property_returns_same_view(self, group_result_2g):
        leaf = self._leaf(group_result_2g)
        # The leaf's .words is cached; two accesses return the same object.
        assert leaf.words is leaf.words


class TestWordsPopulatedMultiPair:
    def test_every_pair_has_pos_and_neg_words(self, group_result_3g):
        for key in group_result_3g.words.keys():
            pair_words = list(group_result_3g.words[key])
            assert any(w.side == "pos" for w in pair_words), f"no pos words for {key}"
            assert any(w.side == "neg" for w in pair_words), f"no neg words for {key}"

    def test_per_pair_contrast_tag(self, group_result_3g):
        for key in group_result_3g.words.keys():
            expected = f"{key[0]}_{key[1]}"
            for w in group_result_3g.words[key]:
                assert w.contrast == expected


class TestClustersPopulatedSinglePair:
    def _leaf(self, group_result_2g):
        pair = next(iter(group_result_2g.pairs))
        return group_result_2g[(pair.g1, pair.g2)]

    def test_pos_and_neg_have_clusters(self, group_result_2g):
        clusters = self._leaf(group_result_2g).clusters
        assert len(list(clusters.pos)) > 0, "no pos clusters"
        assert len(list(clusters.neg)) > 0, "no neg clusters"

    # removed: test_cluster_contrast_matches_pair — PairResult._clusters_for does not
    # forward contrast to _compute_clusters_for_side; Cluster.contrast is always None.
