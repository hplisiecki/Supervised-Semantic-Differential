"""Integration tests for SSD.fit_multipls (end-to-end, uses conftest fixtures)."""

import numpy as np
import pytest


# All fixtures (`ssd_instance`, `tiny_kv`, etc.) come from tests/conftest.py.


class TestFitMultiPLSBasic:
    def test_returns_multipls_result(self, ssd_instance):
        from ssdiff.results import MultiPLSResult
        res = ssd_instance.fit_multipls(n_components=2, rotate="raw", p_method=None)
        assert isinstance(res, MultiPLSResult)

    def test_leaf_keys_shape(self, ssd_instance):
        res = ssd_instance.fit_multipls(n_components=2, rotate="raw", p_method=None)
        assert list(res._leaves.keys()) == ["dim-1", "dim-2", "combined"]

    def test_combined_leaf_beta_matches_fit_pls(self, ssd_instance):
        """`combined` leaf's beta must equal fit_pls(n_components=k).beta up to orientation.

        ``fit_pls`` applies an outcome-alignment sign flip (``_orient_beta``);
        ``mpls_fit`` returns the unrotated PLS coef without it. Compare via
        absolute value of cosine similarity to absorb any ±1 difference.
        """
        fit_pls = ssd_instance.fit_pls(n_components=2, p_method=None)
        res = ssd_instance.fit_multipls(n_components=2, rotate="varimax", p_method=None)

        # ``fit_pls.beta`` is post-scale-division + post-sign-flip.
        # ``res.beta_combined`` is pre-scale-division (i.e., in standardised
        # PCV space); align scales before comparing.
        col_std = ssd_instance.x.std(axis=0, ddof=0)
        scale = np.where(col_std > 1e-12, col_std, 1.0)
        beta_fit_pls_std = fit_pls.beta * scale
        beta_mpls_std = res.beta_combined

        cos = float(
            beta_fit_pls_std @ beta_mpls_std
            / (np.linalg.norm(beta_fit_pls_std) * np.linalg.norm(beta_mpls_std))
        )
        assert abs(cos) > 0.999

    def test_requires_embeddings(self, ssd_instance):
        """Full-vocabulary rotation target needs embeddings — no attached → raise."""
        saved = ssd_instance.embeddings
        try:
            ssd_instance.embeddings = None
            with pytest.raises(RuntimeError, match="embeddings"):
                ssd_instance.fit_multipls(n_components=2, rotate="varimax", p_method=None)
        finally:
            ssd_instance.embeddings = saved

    def test_rejects_categorical(self, tiny_kv, sample_docs, sample_groups, lexicon):
        from ssdiff.corpus import Corpus
        from ssdiff.ssd import SSD
        corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
        ssd = SSD(tiny_kv, corpus, sample_groups, lexicon)
        with pytest.raises(ValueError, match="numeric"):
            ssd.fit_multipls(n_components=2, rotate="varimax")

    def test_nipals_collapse_raises(self, tiny_kv, sample_docs, sample_y, lexicon):
        """Requesting more components than NIPALS can produce must raise."""
        from ssdiff.corpus import Corpus
        from ssdiff.ssd import SSD
        corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
        ssd = SSD(tiny_kv, corpus, sample_y, lexicon)
        # sample_y has 8 docs → n - 1 = 7. Request k = 50.
        with pytest.raises(ValueError, match="n_components"):
            ssd.fit_multipls(n_components=50, rotate="raw", p_method=None)

    def test_p_method_split_runs(self, ssd_instance):
        """Smoke test: split test executes on the container and populates pvalue."""
        res = ssd_instance.fit_multipls(
            n_components=2, rotate="raw", p_method="split",
            n_splits=5, random_state=42,
        )
        p = next(iter(res.stats))["pvalue"]
        assert 0 <= p <= 1

    def test_sign_flip_stable(self, ssd_instance):
        """Repeated fits produce the same dim orientations (signs don't flip at random)."""
        r1 = ssd_instance.fit_multipls(n_components=2, rotate="raw", p_method=None)
        r2 = ssd_instance.fit_multipls(n_components=2, rotate="raw", p_method=None)
        for key in ("dim-1", "dim-2", "combined"):
            np.testing.assert_allclose(r1[key].beta, r2[key].beta, rtol=1e-10)

    def test_rotate_raw_equivalence_to_fit_pls_per_component(self, ssd_instance):
        """With rotate='raw' and k=2, dim leaves' gradients should be
        ± the oriented per-component gradients of fit_pls(n_components=2).
        """
        from ssdiff.utils.math import unit_vector

        fit_pls = ssd_instance.fit_pls(n_components=2, p_method=None)
        res = ssd_instance.fit_multipls(n_components=2, rotate="raw", p_method=None)

        W_ref = np.asarray(fit_pls.component_weights)
        W_raw = res.W  # (D, 2), same standardised space

        def col_set(M):
            return [unit_vector(M[:, i]) for i in range(M.shape[1])]

        # Cosine-similarity match (allowing ±sign and any column permutation).
        ref = col_set(W_ref)
        got = col_set(W_raw)
        for g in got:
            cosines = [abs(float(g @ r)) for r in ref]
            assert max(cosines) > 0.999
