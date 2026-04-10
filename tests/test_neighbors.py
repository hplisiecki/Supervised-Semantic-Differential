"""Tests for ssdiff.utils.neighbors — neighbor search and clustering."""

import numpy as np
import pytest

from ssdiff.utils.neighbors import cluster_top_neighbors, filtered_neighbors


class TestFilteredNeighbors:
    def test_basic(self, tiny_kv):
        vec = tiny_kv["kraj"]
        nbrs = filtered_neighbors(tiny_kv, vec, topn=5)
        assert len(nbrs) <= 5
        # "ABC123" and "Warszawa" should be filtered out
        words = [w for w, _ in nbrs]
        assert "ABC123" not in words
        assert "Warszawa" not in words
        # Neighbors should be sorted by descending similarity
        sims = [s for _, s in nbrs]
        assert sims == sorted(sims, reverse=True)
        # Similarities should be in valid cosine range (small tolerance for float precision)
        assert all(-1 - 1e-6 <= s <= 1 + 1e-6 for _, s in nbrs)

    def test_returns_tuples(self, tiny_kv):
        vec = tiny_kv["piekny"]
        nbrs = filtered_neighbors(tiny_kv, vec, topn=3)
        for item in nbrs:
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], float)


class TestClusterTopNeighbors:
    def test_basic(self, tiny_kv_large):
        rng = np.random.default_rng(42)
        beta = rng.normal(size=10).astype(np.float64)
        clusters = cluster_top_neighbors(
            tiny_kv_large, beta, topn=20, k=2, side="pos",
        )
        assert isinstance(clusters, list)
        assert len(clusters) > 0
        for c in clusters:
            assert "id" in c
            assert "size" in c
            assert "coherence" in c
            assert "words" in c
            assert isinstance(c["words"], list)
            assert c["size"] > 0
            assert 0 <= c["coherence"] <= 1
            assert len(c["words"]) > 0
            # Each word entry should have actual values
            for w in c["words"]:
                assert "word" in w
                assert "cos_centroid" in w
                assert -1 <= w["cos_centroid"] <= 1

    def test_not_enough_neighbors(self, tiny_kv):
        beta = np.ones(8, dtype=np.float64)
        # tiny_kv only has ~18 filterable words; with very restrictive params:
        with pytest.raises(ValueError, match="Not enough"):
            cluster_top_neighbors(tiny_kv, beta, topn=2, k_min=5, side="pos")
