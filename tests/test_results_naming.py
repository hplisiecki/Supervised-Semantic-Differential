"""Tests for new attributes introduced by the result-naming alignment spec."""

import numpy as np
import pytest


class TestAlignmentScoresContinuous:
    def test_alignment_scores_shape(self, pls_result):
        assert pls_result.alignment_scores.shape == (pls_result.stats.n_kept,)

    def test_alignment_scores_in_unit_range(self, pls_result):
        s = pls_result.alignment_scores
        assert np.all(s >= -1.0 - 1e-10)
        assert np.all(s <= 1.0 + 1e-10)

    def test_alignment_scores_equals_xunit_dot_gradient(self, pls_result):
        x = pls_result.x
        x_norms = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        expected = ((x / x_norms) @ pls_result.gradient).ravel()
        np.testing.assert_allclose(
            pls_result.alignment_scores, expected, rtol=1e-10, atol=1e-10,
        )

    def test_alignment_scores_matches_per_doc(self, pls_result):
        per_doc = np.array([d.alignment_score for d in pls_result.docs])
        np.testing.assert_allclose(
            pls_result.alignment_scores, per_doc, rtol=1e-10, atol=1e-10,
        )

    def test_alignment_scores_cached(self, pls_result):
        a = pls_result.alignment_scores
        b = pls_result.alignment_scores
        # Same object returned on second access → property is cached.
        assert a is b


class TestPLSComponentSurface:
    def test_n_components_is_int(self, pls_result):
        assert isinstance(pls_result.n_components, int)
        assert pls_result.n_components >= 1

    def test_n_components_matches_fit_info(self, pls_result):
        assert pls_result.n_components == pls_result.fit_info.n_components

    def test_component_scores_shape(self, pls_result):
        T = pls_result.component_scores
        assert T.ndim == 2
        assert T.shape[0] == pls_result.stats.n_kept
        assert T.shape[1] == pls_result.n_components

    def test_component_weights_shape(self, pls_result):
        W = pls_result.component_weights
        assert W.ndim == 2
        assert W.shape[0] == pls_result.x.shape[1]  # D
        assert W.shape[1] == pls_result.n_components

    def test_component_weights_are_unit_normed(self, pls_result):
        W = pls_result.component_weights
        col_norms = np.linalg.norm(W, axis=0)
        np.testing.assert_allclose(col_norms, 1.0, rtol=1e-6, atol=1e-6)


class TestPCAOLSSurface:
    def test_pca_k_is_int(self, pcaols_result):
        assert isinstance(pcaols_result.pca_k, int)
        assert pcaols_result.pca_k >= 1

    def test_pca_k_matches_fit_info(self, pcaols_result):
        assert pcaols_result.pca_k == pcaols_result.fit_info.n_components

    def test_pca_components_shape(self, pcaols_result):
        V = pcaols_result.pca_components
        assert V.ndim == 2
        assert V.shape == (pcaols_result.pca_k, pcaols_result.x.shape[1])

    def test_pca_weights_shape(self, pcaols_result):
        w = pcaols_result.pca_weights
        assert w.ndim == 1
        assert w.shape == (pcaols_result.pca_k,)

    def test_pca_components_orthonormal(self, pcaols_result):
        V = pcaols_result.pca_components
        gram = V @ V.T
        np.testing.assert_allclose(
            gram, np.eye(pcaols_result.pca_k), rtol=1e-6, atol=1e-6,
        )


class TestGroupAlignmentScores:
    """Top-level ``gr.alignment_scores`` accessor — always a dict now."""

    def test_alignment_scores_shape_single_pair(self, group_result_2g):
        gr = group_result_2g
        scores = gr.alignment_scores
        # Single pair → dict with one entry
        assert isinstance(scores, dict)
        assert len(scores) == 1
        arr = next(iter(scores.values()))
        assert arr.shape == (gr.n_kept,)

    def test_alignment_scores_values_single_pair(self, group_result_2g):
        gr = group_result_2g
        scores = gr.alignment_scores
        # Equals x @ (c_g1 - c_g2) / ||c_g1 - c_g2||
        pair = next(iter(gr.pairs))
        arr = scores[(pair.g1, pair.g2)]
        x = gr.x
        groups = gr.groups
        c1 = x[groups == pair.g1].mean(axis=0)
        c2 = x[groups == pair.g2].mean(axis=0)
        contrast = c1 - c2
        grad = contrast / np.linalg.norm(contrast)
        expected = (x @ grad).ravel()
        np.testing.assert_allclose(arr, expected, rtol=1e-10, atol=1e-10)

    def test_alignment_scores_requires_x(self, group_result_2g):
        """When _x/_groups are None, building PairResult raises RuntimeError."""
        from ssdiff.results.group_result import PairResult
        gr = group_result_2g
        pair = next(iter(gr.pairs))
        # Temporarily null out the container arrays.
        saved_x, saved_groups = gr._x, gr._groups
        try:
            gr._x = None
            gr._groups = None
            with pytest.raises(RuntimeError):
                PairResult(container=gr, g1=pair.g1, g2=pair.g2)
        finally:
            gr._x = saved_x
            gr._groups = saved_groups

    def test_alignment_scores_three_groups_is_dict(self, group_result_3g):
        """3-group result → dict keyed by canonical pair tuples."""
        gr = group_result_3g
        scores = gr.alignment_scores
        assert isinstance(scores, dict)
        assert len(scores) == 3
        for key, arr in scores.items():
            assert isinstance(key, tuple) and len(key) == 2
            # Shape is per-pair n (n_g1 + n_g2), not total n_kept.
            leaf = gr[key]
            assert arr.shape == (len(leaf.x),)


class TestGroupBetaGradient:
    """Top-level ``gr.beta``, ``gr.gradient``, ``gr.beta_norm`` accessors — always dicts."""

    def test_beta_equals_centroid_difference_single_pair(self, group_result_2g):
        gr = group_result_2g
        pair = next(iter(gr.pairs))
        c1 = gr.x[gr.groups == pair.g1].mean(axis=0)
        c2 = gr.x[gr.groups == pair.g2].mean(axis=0)
        beta = gr.beta[(pair.g1, pair.g2)]
        np.testing.assert_allclose(beta, c1 - c2, rtol=1e-10, atol=1e-10)

    def test_gradient_is_unit_length_single_pair(self, group_result_2g):
        gr = group_result_2g
        pair = next(iter(gr.pairs))
        grad = gr.gradient[(pair.g1, pair.g2)]
        np.testing.assert_allclose(np.linalg.norm(grad), 1.0, rtol=1e-10)

    def test_gradient_equals_beta_over_norm_single_pair(self, group_result_2g):
        gr = group_result_2g
        pair = next(iter(gr.pairs))
        beta = gr.beta[(pair.g1, pair.g2)]
        grad = gr.gradient[(pair.g1, pair.g2)]
        np.testing.assert_allclose(
            grad, beta / np.linalg.norm(beta),
            rtol=1e-10, atol=1e-10,
        )

    def test_beta_norm_matches_norm_of_beta_single_pair(self, group_result_2g):
        gr = group_result_2g
        pair = next(iter(gr.pairs))
        beta = gr.beta[(pair.g1, pair.g2)]
        bn = gr.beta_norm[(pair.g1, pair.g2)]
        assert bn == pytest.approx(float(np.linalg.norm(beta)))

    def test_alignment_scores_consumes_gradient_single_pair(self, group_result_2g):
        """alignment_scores must equal x @ gradient (refactor symmetry check)."""
        gr = group_result_2g
        pair = next(iter(gr.pairs))
        grad = gr.gradient[(pair.g1, pair.g2)]
        scores = gr.alignment_scores[(pair.g1, pair.g2)]
        # The leaf's x is the slice belonging to both groups.
        leaf = gr[(pair.g1, pair.g2)]
        expected = (leaf.x @ grad).ravel()
        np.testing.assert_allclose(scores, expected, rtol=1e-10, atol=1e-10)

    def test_beta_dict_three_groups(self, group_result_3g):
        """3-group result → gr.beta is dict[tuple, ndarray]."""
        gr = group_result_3g
        assert isinstance(gr.beta, dict)
        assert len(gr.beta) == 3
        for key, b in gr.beta.items():
            assert isinstance(key, tuple) and len(key) == 2
            assert b.shape == (gr.x.shape[1],)

    def test_beta_norm_dict_three_groups(self, group_result_3g):
        gr = group_result_3g
        assert isinstance(gr.beta_norm, dict)
        for key, bn in gr.beta_norm.items():
            assert isinstance(bn, float)
            assert bn >= 0.0

    def test_gradient_dict_three_groups(self, group_result_3g):
        gr = group_result_3g
        assert isinstance(gr.gradient, dict)
        for key, grad in gr.gradient.items():
            np.testing.assert_allclose(np.linalg.norm(grad), 1.0, rtol=1e-10)
