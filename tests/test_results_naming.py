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


class TestPairViewAlignmentScores:
    def test_alignment_scores_shape(self, group_result_2g):
        gr = group_result_2g
        pair = next(iter(gr.pairs))
        pv = gr.pairs[pair.g1, pair.g2]
        scores = pv.alignment_scores
        assert scores.shape == (gr.n_kept,)

    def test_alignment_scores_values(self, group_result_2g):
        gr = group_result_2g
        pair = next(iter(gr.pairs))
        pv = gr.pairs[pair.g1, pair.g2]
        scores = pv.alignment_scores
        # Equals x @ (c_g1 - c_g2) / ||c_g1 - c_g2||
        x = gr.x
        groups = gr.groups
        c1 = x[groups == pair.g1].mean(axis=0)
        c2 = x[groups == pair.g2].mean(axis=0)
        contrast = c1 - c2
        grad = contrast / np.linalg.norm(contrast)
        expected = (x @ grad).ravel()
        np.testing.assert_allclose(scores, expected, rtol=1e-10, atol=1e-10)

    def test_alignment_scores_sign_flips_on_reverse(self, group_result_2g):
        gr = group_result_2g
        pair = next(iter(gr.pairs))
        fwd = gr.pairs[pair.g1, pair.g2].alignment_scores
        rev = gr.pairs[pair.g2, pair.g1].alignment_scores
        np.testing.assert_allclose(rev, -fwd, rtol=1e-10, atol=1e-10)

    def test_alignment_scores_requires_x(self, group_result_2g):
        """When x is None (e.g. after detach), accessing alignment_scores raises."""
        gr = group_result_2g
        saved_x, saved_groups = gr.x, gr.groups
        try:
            gr.x = None
            gr.groups = None
            pair = next(iter(gr.pairs))
            # Note: .pairs was built BEFORE we nulled x; the PairView still
            # holds a parent pointer to gr, so reading the now-None x raises.
            pv = gr.pairs[pair.g1, pair.g2]
            # Also clear any cached canonical (beta/gradient) so the raise path
            # is exercised, not a stale cache populated by an earlier test.
            pv._canonical_cache = None
            with pytest.raises(RuntimeError, match="requires the original x"):
                _ = pv.alignment_scores
        finally:
            gr.x = saved_x
            gr.groups = saved_groups

    def test_alignment_scores_three_groups(self, group_result_3g):
        """Smoke test on a 3-group result — multiple pairs, each independently."""
        gr = group_result_3g
        for pair in gr.pairs:
            pv = gr.pairs[pair.g1, pair.g2]
            scores = pv.alignment_scores
            assert scores.shape == (gr.n_kept,)


class TestPairViewBetaGradient:
    def test_beta_equals_centroid_difference(self, group_result_2g):
        gr = group_result_2g
        pair = next(iter(gr.pairs))
        pv = gr.pairs[pair.g1, pair.g2]
        c1 = gr.x[gr.groups == pair.g1].mean(axis=0)
        c2 = gr.x[gr.groups == pair.g2].mean(axis=0)
        np.testing.assert_allclose(pv.beta, c1 - c2, rtol=1e-10, atol=1e-10)

    def test_gradient_is_unit_length(self, group_result_2g):
        gr = group_result_2g
        pair = next(iter(gr.pairs))
        pv = gr.pairs[pair.g1, pair.g2]
        np.testing.assert_allclose(np.linalg.norm(pv.gradient), 1.0, rtol=1e-10)

    def test_gradient_equals_beta_over_norm(self, group_result_2g):
        gr = group_result_2g
        pair = next(iter(gr.pairs))
        pv = gr.pairs[pair.g1, pair.g2]
        np.testing.assert_allclose(
            pv.gradient, pv.beta / np.linalg.norm(pv.beta),
            rtol=1e-10, atol=1e-10,
        )

    def test_beta_norm_matches_norm_of_beta(self, group_result_2g):
        gr = group_result_2g
        pair = next(iter(gr.pairs))
        pv = gr.pairs[pair.g1, pair.g2]
        assert pv.beta_norm == pytest.approx(float(np.linalg.norm(pv.beta)))

    def test_beta_and_gradient_sign_flip_on_reverse(self, group_result_2g):
        gr = group_result_2g
        pair = next(iter(gr.pairs))
        fwd = gr.pairs[pair.g1, pair.g2]
        rev = gr.pairs[pair.g2, pair.g1]
        np.testing.assert_allclose(rev.beta, -fwd.beta, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(rev.gradient, -fwd.gradient, rtol=1e-10, atol=1e-10)

    def test_beta_norm_invariant_under_reverse(self, group_result_2g):
        """beta_norm is a magnitude — reversing the pair must not flip its sign."""
        gr = group_result_2g
        pair = next(iter(gr.pairs))
        fwd = gr.pairs[pair.g1, pair.g2]
        rev = gr.pairs[pair.g2, pair.g1]
        assert fwd.beta_norm == pytest.approx(rev.beta_norm)

    def test_alignment_scores_consumes_gradient(self, group_result_2g):
        """alignment_scores must equal x @ gradient (refactor symmetry check)."""
        gr = group_result_2g
        pair = next(iter(gr.pairs))
        pv = gr.pairs[pair.g1, pair.g2]
        expected = (gr.x @ pv.gradient).ravel()
        np.testing.assert_allclose(pv.alignment_scores, expected, rtol=1e-10, atol=1e-10)
