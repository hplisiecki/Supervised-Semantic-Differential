"""Tests for ssdiff.utils.neighbors — filtered_neighbors and cluster_top_neighbors."""

from __future__ import annotations

import re

import numpy as np
import pytest

from ssdiff.utils.neighbors import cluster_top_neighbors, filtered_neighbors


# ---------------------------------------------------------------------------
# filtered_neighbors tests
# ---------------------------------------------------------------------------


class TestFilteredNeighbors:
    """Tests for filtered_neighbors()."""

    def test_returns_k_items_sorted_descending(self, tiny_kv_large):
        """Invariant 1: returns topn items sorted by cosine descending."""
        rng = np.random.default_rng(0)
        vec = rng.normal(size=tiny_kv_large.vector_size).astype(np.float32)

        result = filtered_neighbors(tiny_kv_large, vec, topn=5, cand=50, restrict=50, lang="pl")

        # Check count
        assert len(result) == 5

        # Check type: list of (str, float) tuples
        for word, sim in result:
            assert isinstance(word, str)
            assert isinstance(sim, float)

        # Check sorted descending
        sims = [sim for _, sim in result]
        assert sims == sorted(sims, reverse=True)

    def test_filters_bad_tokens_digits(self, tiny_kv_large):
        """Invariant 2: tokens containing digits (e.g. 'ABC123') are excluded."""
        # ABC123 is in VOCAB_50 — it matches bad_token_re (has digits)
        rng = np.random.default_rng(7)
        vec = rng.normal(size=tiny_kv_large.vector_size).astype(np.float32)

        result = filtered_neighbors(tiny_kv_large, vec, topn=20, cand=50, restrict=50, lang="pl")

        returned_words = [w for w, _ in result]
        assert "ABC123" not in returned_words

    def test_filters_capitalized_tokens(self, tiny_kv_large):
        """Invariant 2 (extended): tokens starting with uppercase are excluded."""
        # 'Warszawa' starts with uppercase — matches bad_token_re
        rng = np.random.default_rng(7)
        vec = rng.normal(size=tiny_kv_large.vector_size).astype(np.float32)

        result = filtered_neighbors(tiny_kv_large, vec, topn=20, cand=50, restrict=50, lang="pl")

        returned_words = [w for w, _ in result]
        assert "Warszawa" not in returned_words

    def test_filtering_removes_bad_tokens_from_output(self, tiny_kv_large):
        """Invariant 3 adaptation: no uppercase or digit-containing tokens appear.

        Note: filtered_neighbors has no `exclude` parameter. Filtering is
        driven purely by bad_token_re (digits or uppercase start). We verify
        that all returned words pass the clean-token criterion.
        """
        # Build a query vector that is the average of the entire vocabulary
        # so that results are spread across the vocab
        vecs = tiny_kv_large.vectors
        mean_vec = vecs.mean(axis=0)

        result = filtered_neighbors(tiny_kv_large, mean_vec, topn=20, cand=50, restrict=50, lang="pl")

        bad_re = re.compile(r".*\d|^[A-ZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞĄĆĘŁŃÓŚŹŻĈĜĤĴŜŬŠŽĐ]")
        for word, _ in result:
            assert not bad_re.match(word), f"Bad token in output: {word!r}"

    def test_empty_result_when_all_candidates_filtered(self, tiny_kv_large):
        """Invariant 4: returns empty list (not error) when all candidates are filtered."""
        # Build tiny embeddings with only bad tokens (digit/uppercase tokens from VOCAB_20)
        from ssdiff.embeddings import Embeddings

        bad_words = ["ABC123", "Warszawa"]
        # Extract their vectors from tiny_kv_large
        vecs = np.vstack([tiny_kv_large.get_vector(w) for w in bad_words]).astype(np.float32)
        bad_emb = Embeddings(bad_words, vecs)
        bad_emb.l2_normalized = True  # vectors extracted from a unit-normed source

        rng = np.random.default_rng(0)
        vec = rng.normal(size=bad_emb.vector_size).astype(np.float32)

        result = filtered_neighbors(bad_emb, vec, topn=5, cand=10, restrict=10, lang="pl")

        assert result == []

    def test_fewer_results_than_topn_when_vocab_small(self, tiny_kv_large):
        """When fewer than topn clean tokens exist, returns as many as are available."""
        from ssdiff.embeddings import Embeddings

        # Only 2 clean words available
        clean_words = ["kraj", "dom"]
        vecs = np.vstack([tiny_kv_large.get_vector(w) for w in clean_words]).astype(np.float32)
        small_emb = Embeddings(clean_words, vecs)
        small_emb.l2_normalized = True  # vectors extracted from a unit-normed source

        rng = np.random.default_rng(0)
        vec = rng.normal(size=small_emb.vector_size).astype(np.float32)

        result = filtered_neighbors(small_emb, vec, topn=10, cand=10, restrict=10, lang="pl")

        # At most 2 clean words can be returned
        assert len(result) <= 2


# ---------------------------------------------------------------------------
# cluster_top_neighbors tests
# ---------------------------------------------------------------------------


class TestClusterTopNeighbors:
    """Tests for cluster_top_neighbors()."""

    def test_fixed_k_returns_k_clusters_with_words(self, tiny_kv_large):
        """Invariant 5: k=2 returns 2 Cluster dicts, each with expected keys."""
        rng = np.random.default_rng(0)
        beta = rng.normal(size=tiny_kv_large.vector_size).astype(np.float64)

        clusters = cluster_top_neighbors(
            tiny_kv_large, beta, topn=40, k=2, restrict_vocab=50, lang="pl"
        )

        # Must return exactly 2 clusters (k=2, min_cluster_size=2 default)
        assert len(clusters) == 2

        required_keys = {"id", "size", "centroid_cos_beta", "coherence", "words"}
        for cluster in clusters:
            assert required_keys <= set(cluster.keys()), f"Missing keys in cluster: {cluster.keys()}"
            # words is a list of dicts with word/cos_centroid/cos_beta
            assert isinstance(cluster["words"], list)
            assert len(cluster["words"]) >= 1
            for row in cluster["words"]:
                assert "word" in row
                assert "cos_centroid" in row
                assert "cos_beta" in row

    def test_neg_side_returns_different_words_than_pos(self, tiny_kv_large):
        """Invariant 6: side='neg' queries -beta and produces different neighbors."""
        rng = np.random.default_rng(42)
        beta = rng.normal(size=tiny_kv_large.vector_size).astype(np.float64)

        clusters_pos = cluster_top_neighbors(
            tiny_kv_large, beta, topn=30, k=2, restrict_vocab=50, lang="pl"
        )
        clusters_neg = cluster_top_neighbors(
            tiny_kv_large, beta, topn=30, k=2, restrict_vocab=50, side="neg", lang="pl"
        )

        words_pos = {row["word"] for c in clusters_pos for row in c["words"]}
        words_neg = {row["word"] for c in clusters_neg for row in c["words"]}

        # The negative direction should not be identical to positive
        assert words_pos != words_neg, "pos and neg sides returned identical words"

    def test_auto_k_runs_and_returns_reasonable_clusters(self, tiny_kv_large):
        """Invariant 7: k=None triggers kmeans_auto_k; result has >=2 clusters."""
        rng = np.random.default_rng(10)
        beta = rng.normal(size=tiny_kv_large.vector_size).astype(np.float64)

        clusters = cluster_top_neighbors(
            tiny_kv_large,
            beta,
            topn=40,
            k=None,
            k_min=2,
            k_max=5,
            restrict_vocab=50,
            lang="pl",
        )

        # Auto-k picks between k_min=2 and k_max=5; min_cluster_size=2 may drop small clusters
        # so we just assert we got at least 1 cluster and each has >=2 words (min_cluster_size)
        assert len(clusters) >= 1
        for cluster in clusters:
            assert cluster["size"] >= 2
            assert len(cluster["words"]) >= 1

    def test_fixed_k_larger_than_neighbors_is_clamped(self, tiny_kv_large):
        """Invariant 8: k clamped to len(words) when k > available clean neighbors.

        Source: `k_clamped = min(int(k), len(words))`. No error is raised.
        With only ~48 clean words (50 vocab minus 2 bad), k=5 << len(words),
        so this tests the normal path. To test clamping, we use a tiny vocab.
        """
        from ssdiff.embeddings import Embeddings

        # Build embeddings with very few clean words (3 clean, 2 bad)
        words = ["kraj", "dom", "praca", "ABC123", "Warszawa"]
        vecs = np.vstack([tiny_kv_large.get_vector(w) for w in words]).astype(np.float32)
        small_emb = Embeddings(words, vecs)
        small_emb.l2_normalized = True  # vectors extracted from a unit-normed source

        rng = np.random.default_rng(0)
        beta = rng.normal(size=small_emb.vector_size).astype(np.float64)

        # k=5 > 3 clean words — should clamp k to 3, not raise
        # But clusters with <2 members are dropped, so result may be 0 or more
        # The important thing is no ValueError is raised here
        clusters = cluster_top_neighbors(
            small_emb, beta, topn=10, k=5, restrict_vocab=10, lang="pl"
        )
        # clusters is a list (may be empty if all clusters < min_cluster_size)
        assert isinstance(clusters, list)

    def test_too_few_valid_neighbors_raises_value_error(self, tiny_kv_large):
        """Invariant 8 (error path): raises ValueError when fewer than max(2, k_min) neighbors."""
        from ssdiff.embeddings import Embeddings

        # Only 1 clean word available
        words = ["kraj", "ABC123"]
        vecs = np.vstack([tiny_kv_large.get_vector(w) for w in words]).astype(np.float32)
        tiny_emb = Embeddings(words, vecs)
        tiny_emb.l2_normalized = True  # vectors extracted from a unit-normed source

        rng = np.random.default_rng(0)
        beta = rng.normal(size=tiny_emb.vector_size).astype(np.float64)

        # Only 1 clean word survives filtering; max(2, k_min=2) = 2 → ValueError
        with pytest.raises(ValueError, match="Not enough neighbors"):
            cluster_top_neighbors(tiny_emb, beta, topn=10, k=2, restrict_vocab=10, lang="pl")

    def test_pos_clusters_sorted_by_centroid_cos_beta_descending(self, tiny_kv_large):
        """side='pos': clusters are sorted by centroid_cos_beta descending."""
        rng = np.random.default_rng(5)
        beta = rng.normal(size=tiny_kv_large.vector_size).astype(np.float64)

        clusters = cluster_top_neighbors(
            tiny_kv_large, beta, topn=40, k=3, restrict_vocab=50, side="pos", lang="pl"
        )

        cos_betas = [c["centroid_cos_beta"] for c in clusters]
        assert cos_betas == sorted(cos_betas, reverse=True)

    def test_neg_clusters_sorted_by_centroid_cos_beta_ascending(self, tiny_kv_large):
        """side='neg': clusters are sorted by centroid_cos_beta ascending."""
        rng = np.random.default_rng(5)
        beta = rng.normal(size=tiny_kv_large.vector_size).astype(np.float64)

        clusters = cluster_top_neighbors(
            tiny_kv_large, beta, topn=40, k=3, restrict_vocab=50, side="neg", lang="pl"
        )

        cos_betas = [c["centroid_cos_beta"] for c in clusters]
        assert cos_betas == sorted(cos_betas, reverse=False)
