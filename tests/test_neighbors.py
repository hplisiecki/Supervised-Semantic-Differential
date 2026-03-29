"""Tests for ssdlite.utils.neighbors — neighbor search and clustering."""

import numpy as np
import pytest

from ssdlite.utils.neighbors import filtered_neighbors, cluster_top_neighbors


class TestFilteredNeighbors:
    def test_basic(self, tiny_kv):
        vec = tiny_kv["kraj"]
        nbrs = filtered_neighbors(tiny_kv, vec, topn=5)
        assert len(nbrs) <= 5
        # "ABC123" and "Warszawa" should be filtered out
        words = [w for w, _ in nbrs]
        assert "ABC123" not in words
        assert "Warszawa" not in words

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

    def test_not_enough_neighbors(self, tiny_kv):
        beta = np.ones(8, dtype=np.float64)
        # tiny_kv only has ~18 filterable words; with very restrictive params:
        with pytest.raises(ValueError, match="Not enough"):
            cluster_top_neighbors(tiny_kv, beta, topn=2, k_min=5, side="pos")
