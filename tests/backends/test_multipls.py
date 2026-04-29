"""Unit tests for ssdiff.backends.pls.mpls_fit — plskit-delegating orchestrator."""

import numpy as np
import pytest


class TestMplsFit:
    @staticmethod
    def _make_toy(rng, n=200, D=50, k_signal=3):
        """Return pre-standardised (Xs, ys) plus a pre-standardised vocab Es."""
        from ssdiff.utils.math import standardize
        X = rng.normal(size=(n, D))
        beta = np.zeros(D)
        beta[:k_signal] = [1.0, 0.6, -0.4]
        y = X @ beta + 0.1 * rng.normal(size=n)
        Xs, x_mean, x_scale = standardize(X)
        ys2, _, _ = standardize(y.reshape(-1, 1))
        ys = ys2.ravel()
        # Vocab in the same column space as Xs: generate a fake
        # "embedding" matrix and put it through the same shift/scale.
        V = 400
        E = rng.normal(size=(V, D))
        Es = (E - x_mean) / np.where(x_scale > 1e-12, x_scale, 1.0)
        return Xs, ys, Es

    def test_raw_leaves_weights_unchanged(self):
        """rotate='raw' leaves the W-subspace unchanged (R is a signed permutation)."""
        from ssdiff.backends.pls import mpls_fit

        rng = np.random.default_rng(10)
        Xs, ys, Es = self._make_toy(rng)

        out = mpls_fit(Xs, ys, n_components=3, rotate="raw", E_target=Es)
        W, W_rot = out["W"], out["W_rot"]
        # Subspace equality: project W_rot onto W's column space → residual ≈ 0.
        P_W = W @ np.linalg.pinv(W)
        residual = np.linalg.norm(W_rot - P_W @ W_rot)
        assert residual < 1e-8
        # "raw" rotation R must be a signed permutation.
        R = out["rotation_meta"]["R"]
        abs_R = np.abs(R)
        np.testing.assert_allclose(abs_R.sum(axis=0), np.ones(3), atol=1e-6)
        np.testing.assert_allclose(abs_R.sum(axis=1), np.ones(3), atol=1e-6)

    def test_combined_beta_matches_pls1(self):
        """'combined' β equals the unrotated pls1_fit coef to machine precision."""
        import plskit
        from ssdiff.backends.pls import mpls_fit

        rng = np.random.default_rng(11)
        Xs, ys, Es = self._make_toy(rng)
        coef_ref = plskit.pls1_fit(Xs, ys, k=3, pre_standardized_X=True).coef

        out = mpls_fit(Xs, ys, n_components=3, rotate="varimax", E_target=Es)
        np.testing.assert_allclose(
            out["beta_combined"], coef_ref, rtol=1e-10, atol=1e-12,
        )

    def test_dim_ordering_by_abs_corr(self):
        """After ordering, |corr(dim_score_i, y)| is monotonically non-increasing."""
        from ssdiff.backends.pls import mpls_fit

        rng = np.random.default_rng(12)
        Xs, ys, Es = self._make_toy(rng)

        out = mpls_fit(Xs, ys, n_components=3, rotate="varimax", E_target=Es)
        T_rot = out["T_rot"]
        y_c = ys - ys.mean()
        corrs = []
        for i in range(T_rot.shape[1]):
            ti = T_rot[:, i] - T_rot[:, i].mean()
            denom = np.linalg.norm(ti) * np.linalg.norm(y_c)
            corrs.append(abs(float(ti @ y_c) / denom) if denom > 0 else 0.0)
        for a, b in zip(corrs, corrs[1:]):
            assert a + 1e-9 >= b

    def test_sign_flip_makes_corr_positive(self):
        """All dim scores have non-negative correlation with y after sign flip."""
        from ssdiff.backends.pls import mpls_fit

        rng = np.random.default_rng(13)
        Xs, ys, Es = self._make_toy(rng)

        out = mpls_fit(Xs, ys, n_components=3, rotate="varimax", E_target=Es)
        T_rot = out["T_rot"]
        y_c = ys - ys.mean()
        for i in range(T_rot.shape[1]):
            ti = T_rot[:, i] - T_rot[:, i].mean()
            denom = np.linalg.norm(ti) * np.linalg.norm(y_c)
            corr = float(ti @ y_c) / denom if denom > 0 else 0.0
            assert corr >= -1e-9

    def test_nipals_truncation_raises(self):
        """Requesting more components than NIPALS returns → raise (no silent truncation)."""
        from ssdiff.backends.pls import mpls_fit

        rng = np.random.default_rng(14)
        # n=10, D=5 → pls1_fit caps at min(n-1, D) = 5; request k=50.
        Xs = rng.normal(size=(10, 5))
        ys = rng.normal(size=10)
        Es = rng.normal(size=(20, 5))
        with pytest.raises(ValueError, match="n_components"):
            mpls_fit(Xs, ys, n_components=50, rotate="raw", E_target=Es)

    def test_rotation_meta_keys_varimax(self):
        """rotation_meta carries the required keys for varimax rotation."""
        from ssdiff.backends.pls import mpls_fit

        rng = np.random.default_rng(15)
        Xs, ys, Es = self._make_toy(rng)

        out = mpls_fit(Xs, ys, n_components=3, rotate="varimax", E_target=Es)
        meta = out["rotation_meta"]
        required = {
            "rotate", "R", "kaiser_normalized", "sweeps", "V_converged",
        }
        assert required.issubset(meta)
        assert meta["rotate"] == "varimax"

    def test_mpls_fit_varimax_R_matches_direct_plskit_rotate(self):
        """Lock in: SSD's varimax path goes through plskit.rotate,
        not a forked algorithm."""
        import numpy as np
        import plskit
        from ssdiff.backends.pls import mpls_fit

        rng = np.random.default_rng(0)
        n, d, V, k = 50, 10, 100, 3
        X = rng.normal(size=(n, d))
        Xs = (X - X.mean(0)) / X.std(0)
        y = (rng.normal(size=n))
        ys = (y - y.mean()) / y.std()
        E_target = rng.normal(size=(V, d))

        out = mpls_fit(Xs, ys, n_components=k, rotate="varimax", E_target=E_target)

        # Direct plskit.rotate against the same L the orchestrator computes.
        model = plskit.pls1_fit(Xs, ys, k=k, pre_standardized_X=True)
        L = E_target @ model.W
        direct = plskit.rotate(model.W, method="varimax", L=L)

        # The mpls_fit output applies an extra ordering+sign-flip on top of
        # plskit.rotate's R; reverse-engineer that to compare the underlying
        # rotation. order and signs are recorded in rotation_meta.
        order = out["order"]
        signs = out["signs"]
        R_unordered_unsigned = out["rotation_meta"]["R"] / signs[np.newaxis, :]
        # The columns of R_unordered_unsigned were re-ordered by `order`;
        # restore original column order.
        inv = np.argsort(order)
        R_orig = R_unordered_unsigned[:, inv]
        np.testing.assert_allclose(R_orig, direct.spec.R, atol=1e-12)
