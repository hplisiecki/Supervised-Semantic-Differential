"""Tests for ssdiff.utils.neighbors — filtered_neighbors and cluster_top_neighbors."""

from __future__ import annotations

import re

import numpy as np
import pytest

from ssdiff.embeddings import Embeddings
from ssdiff.utils.neighbors import cluster_top_neighbors, filtered_neighbors, skip_neighbor
from tests.conftest import make_kv

# ---------------------------------------------------------------------------
# filtered_neighbors tests
# ---------------------------------------------------------------------------


class TestFilteredNeighbors:
    """Tests for filtered_neighbors()."""

    def test_returns_k_items_sorted_descending(self, tiny_kv_large):
        """returns topn items sorted by cosine descending."""
        rng = np.random.default_rng(0)
        vec = rng.normal(size=tiny_kv_large.vector_size).astype(np.float32)

        result = filtered_neighbors(tiny_kv_large, vec, topn=5, cand=50, restrict=50)

        # Check count
        assert len(result) == 5

        # Check type: list of (str, float) tuples
        for word, sim in result:
            assert isinstance(word, str)
            assert isinstance(sim, float)

        # Check sorted descending
        sims = [sim for _, sim in result]
        assert sims == sorted(sims, reverse=True)

    def test_empty_result_when_all_candidates_filtered(self, tiny_kv_large):
        """returns empty list (not error) when all candidates are filtered."""
        # Only bad tokens (digit / uppercase start)
        bad_emb = make_kv(["ABC123", "Warszawa"], dim=10)

        rng = np.random.default_rng(0)
        vec = rng.normal(size=bad_emb.vector_size).astype(np.float32)

        result = filtered_neighbors(bad_emb, vec, topn=5, cand=10, restrict=10)

        assert result == []

    def test_fewer_results_than_topn_when_vocab_small(self, tiny_kv_large):
        """When fewer than topn clean tokens exist, returns as many as are available."""
        # Only 2 clean words available
        small_emb = make_kv(["kraj", "dom"], dim=10)

        rng = np.random.default_rng(0)
        vec = rng.normal(size=small_emb.vector_size).astype(np.float32)

        result = filtered_neighbors(small_emb, vec, topn=10, cand=10, restrict=10)

        assert sorted(w for w, _ in result) == ["dom", "kraj"]


# ---------------------------------------------------------------------------
# cluster_top_neighbors tests
# ---------------------------------------------------------------------------


class TestClusterTopNeighbors:
    """Tests for cluster_top_neighbors()."""

    def test_fixed_k_returns_k_clusters_with_words(self, tiny_kv_large):
        """k=2 returns 2 Cluster dicts, each with expected keys."""
        rng = np.random.default_rng(0)
        beta = rng.normal(size=tiny_kv_large.vector_size).astype(np.float64)

        clusters = cluster_top_neighbors(
            tiny_kv_large, beta, topn=40, k=2, restrict_vocab=50
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

    def test_neg_side_is_pos_side_of_negated_beta(self, tiny_kv_large):
        """side='neg' queries -beta: same clusters as side='pos' on -beta."""
        rng = np.random.default_rng(42)
        beta = rng.normal(size=tiny_kv_large.vector_size).astype(np.float64)

        clusters_neg = cluster_top_neighbors(
            tiny_kv_large, beta, topn=30, k=2, restrict_vocab=50, side="neg"
        )
        clusters_pos_of_minus = cluster_top_neighbors(
            tiny_kv_large, -beta, topn=30, k=2, restrict_vocab=50
        )

        def words(clusters):
            return sorted(sorted(r["word"] for r in c["words"]) for c in clusters)

        assert words(clusters_neg) == words(clusters_pos_of_minus)

    def test_auto_k_runs_and_returns_reasonable_clusters(self, tiny_kv_large):
        """k=None triggers kmeans_auto_k; result has >=2 clusters."""
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
        )

        # Auto-k picks between k_min=2 and k_max=5; a fixed k=1 would return one cluster
        assert 2 <= len(clusters) <= 5
        for cluster in clusters:
            assert cluster["size"] >= 2
            assert len(cluster["words"]) >= 1

    def test_fixed_k_larger_than_neighbors_is_clamped(self):
        """k is clamped to the number of clean neighbors; no error is raised."""
        # 3 clean words, 2 bad
        small_emb = make_kv(["kraj", "dom", "praca", "ABC123", "Warszawa"], dim=10)

        rng = np.random.default_rng(0)
        beta = rng.normal(size=small_emb.vector_size).astype(np.float64)

        clusters = cluster_top_neighbors(
            small_emb, beta, topn=10, k=5, restrict_vocab=10, min_cluster_size=1
        )
        assert len(clusters) == 3

    def test_too_few_valid_neighbors_raises_value_error(self):
        """raises ValueError when fewer than max(2, k_min) neighbors."""
        # Only 1 clean word available
        tiny_emb = make_kv(["kraj", "ABC123"], dim=10)

        rng = np.random.default_rng(0)
        beta = rng.normal(size=tiny_emb.vector_size).astype(np.float64)

        # Only 1 clean word survives filtering; max(2, k_min=2) = 2 → ValueError
        with pytest.raises(ValueError, match="Not enough neighbors"):
            cluster_top_neighbors(tiny_emb, beta, topn=10, k=2, restrict_vocab=10)

    @pytest.mark.parametrize("side, reverse", [("pos", True), ("neg", False)])
    def test_clusters_sorted_by_centroid_cos_beta(self, tiny_kv_large, side, reverse):
        """pos: centroid_cos_beta descending; neg: ascending."""
        rng = np.random.default_rng(5)
        beta = rng.normal(size=tiny_kv_large.vector_size).astype(np.float64)

        clusters = cluster_top_neighbors(
            tiny_kv_large, beta, topn=40, k=3, restrict_vocab=50, side=side
        )

        cos_betas = [c["centroid_cos_beta"] for c in clusters]
        assert cos_betas == sorted(cos_betas, reverse=reverse)


def test_filtered_neighbors_prefers_the_embedding_filter():
    emb = make_kv(["Dog", "dog", "x1", "cat", "bird", "fish"], dim=8)
    vec = emb["Dog"]
    words = [w for w, _ in filtered_neighbors(emb, vec, topn=6, cand=6, restrict=6)]
    assert "Dog" not in words and "x1" not in words
    emb.neighbor_filter = re.compile(r".*\d")
    words = [w for w, _ in filtered_neighbors(emb, vec, topn=6, cand=6, restrict=6)]
    assert "Dog" in words and "x1" not in words


@pytest.mark.parametrize("word, skipped", [
    ("zly", False), ("ABC123", True), ("abc1", True), ("Warszawa", True),
    ("Ωmega", True), ("Ёлка", True), ("łąka", False), ("Łódź", True),
])
def test_default_neighbor_filter(tiny_kv, word, skipped):
    assert skip_neighbor(tiny_kv, word) is skipped


def test_wordpack_filter_wins(wordpack_path):
    emb = Embeddings.load(wordpack_path)
    assert emb.neighbor_filter is not None
    assert skip_neighbor(emb, "Dog") is False  # pack filter ".*\d" keeps it; the default rule would skip it
    assert skip_neighbor(emb, "dog1") is True
