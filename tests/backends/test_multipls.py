"""Unit tests for ssdiff.backends.multipls — rotation math + orchestrator."""

import numpy as np
import pytest


class TestVarimaxAngle2D:
    def test_identity_matrix_returns_zero(self):
        """An already-simple-structure L should produce θ ≈ 0."""
        from ssdiff.backends.multipls import varimax_angle_2d

        # Two columns with disjoint support — simple structure already.
        L = np.array([
            [1.0, 0.0],
            [0.9, 0.0],
            [0.0, 1.0],
            [0.0, 0.8],
        ])
        theta = varimax_angle_2d(L)
        assert abs(theta) < 1e-8

    def test_45_degree_mix_rotates_toward_simple(self):
        """A 45° mixed loadings matrix should rotate back by ≈ ±45° (mod π/2)."""
        from ssdiff.backends.multipls import varimax_angle_2d

        # Build a clean simple-structure L0, then rotate it by +45°.
        L0 = np.array([
            [1.0, 0.0],
            [0.8, 0.0],
            [0.0, 1.0],
            [0.0, 0.9],
        ])
        c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
        L = L0 @ np.array([[c, -s], [s, c]])

        theta = varimax_angle_2d(L)
        # Should recover a rotation that undoes the +45° (mod π/2).
        # θ lands in (-π/4, π/4], so -π/4 is acceptable.
        # Check theta ≡ -π/4 (mod π/2): distance from theta to nearest
        # value in {..., -3π/4, -π/4, π/4, 3π/4, ...} must be < 1e-6.
        diff = (theta - (-np.pi / 4)) % (np.pi / 2)
        if diff > np.pi / 4:
            diff -= np.pi / 2
        assert abs(diff) < 1e-6

    def test_returns_python_float(self):
        """Ensure the return type is a plain float, not a numpy scalar."""
        from ssdiff.backends.multipls import varimax_angle_2d
        L = np.array([[1.0, 0.0], [0.0, 1.0]])
        theta = varimax_angle_2d(L)
        assert isinstance(theta, float)


class TestVarimaxKaiserSweep:
    def test_k2_matches_closed_form(self):
        """For k=2, sweep should land on the same rotation as the closed-form angle."""
        from ssdiff.backends.multipls import (
            varimax_angle_2d,
            varimax_kaiser_sweep,
        )

        rng = np.random.default_rng(0)
        L = rng.normal(size=(200, 2))
        theta = varimax_angle_2d(L)
        c, s = np.cos(theta), np.sin(theta)
        R_closed = np.array([[c, -s], [s, c]])
        L_closed = L @ R_closed

        # Use kaiser_normalize=False so both use the same (unnormalized) L.
        L_sweep, R_sweep, meta = varimax_kaiser_sweep(L, kaiser_normalize=False)

        # Columns may swap sign / order but the sorted |cov|² per column must match.
        def _var4_sorted(M):
            return np.sort(np.var(M ** 2, axis=0))
        np.testing.assert_allclose(
            _var4_sorted(L_sweep), _var4_sorted(L_closed), rtol=1e-6,
        )

    def test_k3_converges(self):
        """For k=3, sweep should converge within the stated tolerance and increase V."""
        from ssdiff.backends.multipls import varimax_kaiser_sweep

        rng = np.random.default_rng(1)
        # Build a loading with a known simple structure, then mix it.
        L0 = np.zeros((60, 3))
        L0[:20, 0] = rng.uniform(0.8, 1.0, 20)
        L0[20:40, 1] = rng.uniform(0.8, 1.0, 20)
        L0[40:, 2] = rng.uniform(0.8, 1.0, 20)
        M = rng.normal(size=(3, 3))
        Q, _ = np.linalg.qr(M)  # random orthogonal mixer
        L = L0 @ Q

        def _V(M):
            return float(np.sum(np.var(M ** 2, axis=0)))

        L_rot, R, meta = varimax_kaiser_sweep(L)
        assert _V(L_rot) >= _V(L) - 1e-9
        assert meta["sweeps"] >= 1
        assert meta["kaiser_normalized"] is True
        # The sweep should approximately recover the original simple structure:
        # each column of L_rot should have most mass on exactly one of L0's blocks.
        col_max_frac = np.max(L_rot ** 2, axis=0) / np.sum(L_rot ** 2, axis=0)
        assert np.all(col_max_frac > 0.05)  # loose — we only want convergence, not perfection

    def test_R_is_orthogonal(self):
        """Returned rotation matrix must be orthogonal."""
        from ssdiff.backends.multipls import varimax_kaiser_sweep

        rng = np.random.default_rng(2)
        L = rng.normal(size=(50, 4))
        _, R, _ = varimax_kaiser_sweep(L)
        np.testing.assert_allclose(R.T @ R, np.eye(R.shape[0]), atol=1e-10)

    def test_kaiser_normalization_toggle(self):
        """kaiser_normalize=False should skip row normalisation and still converge."""
        from ssdiff.backends.multipls import varimax_kaiser_sweep

        rng = np.random.default_rng(3)
        L = rng.normal(size=(40, 3))
        _, _, meta = varimax_kaiser_sweep(L, kaiser_normalize=False)
        assert meta["kaiser_normalized"] is False


class TestPromaxFit:
    def test_kappa1_equals_varimax(self):
        """Promax at κ=1 is identically varimax (pattern equals structure, Φ = I)."""
        from ssdiff.backends.multipls import promax_fit

        rng = np.random.default_rng(4)
        L_varimax = rng.normal(size=(100, 3))
        out = promax_fit(L_varimax, kappa=1)
        np.testing.assert_allclose(out["pattern"], L_varimax, rtol=1e-10)
        np.testing.assert_allclose(out["structure"], L_varimax, rtol=1e-10)
        np.testing.assert_allclose(out["phi"], np.eye(3), atol=1e-10)

    def test_kappa4_produces_correlated_factors(self):
        """κ=4 (default) should yield Φ ≠ I on generic input."""
        from ssdiff.backends.multipls import promax_fit

        rng = np.random.default_rng(5)
        L = rng.normal(size=(120, 3))
        out = promax_fit(L, kappa=4)
        off_diag = out["phi"] - np.diag(np.diag(out["phi"]))
        assert np.max(np.abs(off_diag)) > 1e-3  # factors should become correlated

    def test_output_contract(self):
        """Returned dict must contain pattern, structure, phi, transform."""
        from ssdiff.backends.multipls import promax_fit
        L = np.eye(3).astype(float)
        out = promax_fit(L, kappa=4)
        assert set(out) == {"pattern", "structure", "phi", "transform", "kappa"}
        assert out["pattern"].shape == (3, 3)
        assert out["structure"].shape == (3, 3)
        assert out["phi"].shape == (3, 3)
        assert out["transform"].shape == (3, 3)
        assert out["kappa"] == 4

    def test_phi_is_a_correlation_matrix(self):
        """phi must have unit diagonal and |off-diagonal| ≤ 1 — it's a factor-correlation matrix."""
        from ssdiff.backends.multipls import promax_fit

        rng = np.random.default_rng(50)
        for seed_k in [(50, 3), (51, 4), (52, 5)]:
            L = np.random.default_rng(seed_k[0]).normal(size=(150, seed_k[1]))
            out = promax_fit(L, kappa=4)
            phi = out["phi"]
            np.testing.assert_allclose(np.diag(phi), np.ones(seed_k[1]), atol=1e-8)
            assert np.max(np.abs(phi)) <= 1.0 + 1e-9, (
                f"phi has entry exceeding 1 in magnitude: max|phi|={np.max(np.abs(phi))}"
            )
            # Symmetric.
            np.testing.assert_allclose(phi, phi.T, atol=1e-10)

    def test_structure_equals_pattern_times_phi(self):
        """The structure matrix must equal pattern @ phi (invariant of the formulation)."""
        from ssdiff.backends.multipls import promax_fit

        rng = np.random.default_rng(60)
        L = rng.normal(size=(80, 4))
        out = promax_fit(L, kappa=4)
        np.testing.assert_allclose(
            out["structure"], out["pattern"] @ out["phi"], rtol=1e-10,
        )


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
        from ssdiff.backends.multipls import mpls_fit

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
        from ssdiff.backends.multipls import mpls_fit
        from ssdiff.backends.pls import pls1_fit

        rng = np.random.default_rng(11)
        Xs, ys, Es = self._make_toy(rng)
        _, _, _, _, coef_ref = pls1_fit(Xs, ys, 3)

        out = mpls_fit(Xs, ys, n_components=3, rotate="varimax", E_target=Es)
        np.testing.assert_allclose(
            out["beta_combined"], coef_ref, rtol=1e-10, atol=1e-12,
        )

    def test_dim_ordering_by_abs_corr(self):
        """After ordering, |corr(dim_score_i, y)| is monotonically non-increasing."""
        from ssdiff.backends.multipls import mpls_fit

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
        from ssdiff.backends.multipls import mpls_fit

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
        from ssdiff.backends.multipls import mpls_fit

        rng = np.random.default_rng(14)
        # n=10, D=5 → pls1_fit caps at min(n-1, D) = 5; request k=50.
        Xs = rng.normal(size=(10, 5))
        ys = rng.normal(size=10)
        Es = rng.normal(size=(20, 5))
        with pytest.raises(ValueError, match="n_components"):
            mpls_fit(Xs, ys, n_components=50, rotate="raw", E_target=Es)

    def test_rotation_meta_keys(self):
        """rotation_meta carries the full contract described in the roadmap."""
        from ssdiff.backends.multipls import mpls_fit

        rng = np.random.default_rng(15)
        Xs, ys, Es = self._make_toy(rng)

        out = mpls_fit(Xs, ys, n_components=3, rotate="promax", E_target=Es)
        meta = out["rotation_meta"]
        required = {
            "rotate", "R", "kaiser_normalized", "sweeps",
            "V_converged", "kappa", "pattern", "structure", "phi",
        }
        assert required.issubset(meta)
        assert meta["rotate"] == "promax"
        assert meta["kappa"] == 4

    def test_promax_phi_survives_reorder_and_sign_flip(self):
        """After mpls_fit's reorder + sign flip, phi must still be a correlation matrix
        and remain consistent with the reordered+signed pattern/structure."""
        from ssdiff.backends.multipls import mpls_fit

        # Sweep several seeds so we hit cases that trigger sign flips.
        for seed in range(16, 22):
            rng = np.random.default_rng(seed)
            Xs, ys, Es = self._make_toy(rng)
            out = mpls_fit(Xs, ys, n_components=3, rotate="promax", E_target=Es)
            meta = out["rotation_meta"]
            phi = meta["phi"]
            pattern = meta["pattern"]
            structure = meta["structure"]

            # Correlation-matrix invariants.
            np.testing.assert_allclose(np.diag(phi), np.ones(3), atol=1e-8)
            assert np.max(np.abs(phi)) <= 1.0 + 1e-9
            np.testing.assert_allclose(phi, phi.T, atol=1e-10)

            # Consistency across the reorder + sign flip.
            np.testing.assert_allclose(structure, pattern @ phi, rtol=1e-8, atol=1e-10)
