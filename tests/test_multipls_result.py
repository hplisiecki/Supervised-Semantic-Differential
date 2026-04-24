"""Unit tests for _PLSComponentResult and MultiPLSResult class mechanics.

No embeddings or corpus — these tests exercise only the result-object
plumbing. Integration tests that go through SSD.fit_multipls live in
tests/test_fit_multipls.py.
"""

import numpy as np
import pytest


def _fake_container(*, W_rot, beta_combined, x, rotation_meta=None):
    """Minimal stand-in for MultiPLSResult — just enough for a leaf to read."""
    class _C:
        pass
    c = _C()
    c._W_rot = W_rot
    c._beta_combined = beta_combined
    c._x = x
    c.embeddings = None
    c.corpus = None
    c.lexicon = set()
    c.window = 3
    c.sif_a = 1e-3
    c.lang = "pl"
    c._rotation_meta = rotation_meta or {"rotate": "varimax", "pattern": None}
    return c


class TestPLSComponentResultLeaf:
    def test_dim_leaf_beta_is_W_rot_column(self):
        from ssdiff.results.multi_pls_result import _PLSComponentResult

        rng = np.random.default_rng(0)
        x = rng.normal(size=(30, 10))
        W_rot = rng.normal(size=(10, 3))
        coef = rng.normal(size=10)
        cont = _fake_container(W_rot=W_rot, beta_combined=coef, x=x)

        leaf = _PLSComponentResult(container=cont, key="dim-1", dim_index=0)
        np.testing.assert_allclose(leaf.beta, W_rot[:, 0])

    def test_dim_leaf_uses_pattern_matrix_in_promax(self):
        from ssdiff.results.multi_pls_result import _PLSComponentResult

        rng = np.random.default_rng(1)
        x = rng.normal(size=(30, 10))
        W_rot = rng.normal(size=(10, 3))
        pattern = rng.normal(size=(500, 3))  # full-vocab pattern matrix
        coef = rng.normal(size=10)
        cont = _fake_container(
            W_rot=W_rot, beta_combined=coef, x=x,
            rotation_meta={"rotate": "promax", "pattern": pattern},
        )

        leaf = _PLSComponentResult(container=cont, key="dim-2", dim_index=1)
        # beta for dim leaves under promax must still be W_rot column
        # (used for projections into embedding space); .words consults the
        # pattern matrix via _compute_words_rows override, not beta.
        np.testing.assert_allclose(leaf.beta, W_rot[:, 1])

    def test_promax_words_ranks_by_pattern_column(self):
        """Regression guard: promax .words must sort vocab by the pattern
        column (length V), NOT feed it to similar_by_vector (which expects
        length D and would crash on shape mismatch)."""
        from types import SimpleNamespace

        from ssdiff.results.multi_pls_result import _PLSComponentResult

        rng = np.random.default_rng(20)
        V, D, k = 50, 10, 2
        # bad_token_re filters tokens containing digits or leading caps, so use
        # alphabetic lowercase stems (unique per index via repeated letter).
        keys = ["a" * (i + 1) for i in range(V)]
        pattern = rng.normal(size=(V, k))
        pattern[7, 0] = 5.0    # top positive  → keys[7] = "aaaaaaaa"
        pattern[19, 0] = -5.0  # top negative  → keys[19] = "a" * 20

        embed_stub = SimpleNamespace(
            index_to_key=keys, vectors=np.zeros((V, D), dtype=np.float32),
        )
        cont = _fake_container(
            W_rot=rng.normal(size=(D, k)),
            beta_combined=rng.normal(size=D),
            x=rng.normal(size=(30, D)),
            rotation_meta={"rotate": "promax", "pattern": pattern},
        )
        cont.embeddings = embed_stub

        leaf = _PLSComponentResult(container=cont, key="dim-1", dim_index=0)
        words = list(leaf.words)
        top_pos_key = keys[7]
        top_neg_key = keys[19]
        assert any(w.word == top_pos_key and w.side == "pos" for w in words), (
            f"expected {top_pos_key!r} (highest pattern[:, 0]) in pos side"
        )
        assert any(w.word == top_neg_key and w.side == "neg" for w in words), (
            f"expected {top_neg_key!r} (lowest pattern[:, 0]) in neg side"
        )
        # Pattern-column scores (not cosines) should flow through.
        top_pos = next(w for w in words if w.side == "pos" and w.rank == 1)
        assert top_pos.word == top_pos_key
        assert top_pos.cos_beta == pytest.approx(5.0)

    def test_combined_leaf_beta_is_beta_combined(self):
        from ssdiff.results.multi_pls_result import _PLSComponentResult

        rng = np.random.default_rng(2)
        x = rng.normal(size=(30, 10))
        W_rot = rng.normal(size=(10, 3))
        coef = rng.normal(size=10)
        cont = _fake_container(W_rot=W_rot, beta_combined=coef, x=x)

        leaf = _PLSComponentResult(container=cont, key="combined", dim_index=None)
        np.testing.assert_allclose(leaf.beta, coef)

    def test_no_test_attribute(self):
        from ssdiff.results.multi_pls_result import _PLSComponentResult

        rng = np.random.default_rng(3)
        cont = _fake_container(
            W_rot=rng.normal(size=(5, 2)),
            beta_combined=rng.normal(size=5),
            x=rng.normal(size=(8, 5)),
        )
        leaf = _PLSComponentResult(container=cont, key="dim-1", dim_index=0)
        # Mirrors PairResult — tests live on the container only.
        assert not hasattr(leaf, "test")

    def test_pickling_drops_container(self):
        import pickle
        from ssdiff.results.multi_pls_result import _PLSComponentResult

        rng = np.random.default_rng(4)
        cont = _fake_container(
            W_rot=rng.normal(size=(5, 2)),
            beta_combined=rng.normal(size=5),
            x=rng.normal(size=(8, 5)),
        )
        leaf = _PLSComponentResult(container=cont, key="dim-1", dim_index=0)
        state = leaf.__getstate__()
        assert state["_container"] is None
        # beta + key + dim_index are preserved.
        np.testing.assert_allclose(state["beta"], leaf.beta)
        assert state["_key"] == "dim-1"
        assert state["_dim_index"] == 0


class TestMultiPLSResultContainer:
    @staticmethod
    def _build(k=3, n=50, D=10, with_combined=True):
        """Build a MultiPLSResult directly from synthetic arrays (no fit)."""
        from ssdiff.results.multi_pls_result import MultiPLSResult
        rng = np.random.default_rng(42)
        return MultiPLSResult(
            x=rng.normal(size=(n, D)),
            y=rng.normal(size=n),
            W=rng.normal(size=(D, k)),
            P=rng.normal(size=(D, k)),
            Q=rng.normal(size=k),
            W_rot=rng.normal(size=(D, k)),
            T_rot=rng.normal(size=(n, k)),
            beta_combined=rng.normal(size=D),
            n_components=k,
            pca_k=None,
            rotation_meta={
                "rotate": "varimax", "R": np.eye(k),
                "kaiser_normalized": True, "sweeps": 3,
                "V_converged": 0.5, "kappa": None,
                "pattern": None, "structure": None, "phi": None,
            },
            r2=0.42,
            test_name=None,
            test_info=None,
            embeddings=None, corpus=None, lexicon=None,
            window=3, sif_a=1e-3, lang="pl",
        ), k

    def test_leaf_keys(self):
        res, k = self._build()
        expected = [f"dim-{i+1}" for i in range(k)] + ["combined"]
        assert list(res._leaves.keys()) == expected

    def test_getitem_returns_component_leaf(self):
        from ssdiff.results.multi_pls_result import _PLSComponentResult
        res, _ = self._build()
        for key in ("dim-1", "dim-2", "dim-3", "combined"):
            assert isinstance(res[key], _PLSComponentResult)

    def test_key_to_str_filename_safe(self):
        res, _ = self._build()
        assert res._key_to_str("dim-1") == "dim_1"
        assert res._key_to_str("combined") == "combined"

    def test_key_repr_human_readable(self):
        res, _ = self._build()
        assert res._key_repr("dim-1") == "Dim 1"
        assert res._key_repr("combined") == "Combined"

    def test_pls_info_view_columns(self):
        res, k = self._build()
        row = next(iter(res.pls_info))
        assert row["n_components"] == k
        assert row["rotate"] == "varimax"
        assert row["pvalue_source"] in ("split", "perm", "split_cal", None)

    def test_setstate_rewires_container(self):
        import pickle
        res, _ = self._build()
        # Force a leaf to cache its beta before pickling.
        _ = res["dim-1"].beta
        pkl = pickle.dumps(res)
        res2 = pickle.loads(pkl)
        for leaf in res2._leaves.values():
            assert leaf._container is res2

    def test_raw_diagnostics_access(self):
        """Unrotated W / P / Q / beta_combined are reachable on the container."""
        res, k = self._build()
        assert res.W.shape == (10, k)
        assert res.W_rot.shape == (10, k)
        assert res.beta_combined.shape == (10,)

    def test_attach_fans_out_to_leaves(self):
        """res.attach(embeddings=X) must update every leaf, not just the container."""
        res, _ = self._build()
        marker = object()
        res.attach(embeddings=marker, corpus=marker)
        assert res.embeddings is marker
        assert res.corpus is marker
        for leaf in res._leaves.values():
            assert leaf.embeddings is marker
            assert leaf.corpus is marker


class TestPackageRegistration:
    def test_top_level_import(self):
        from ssdiff.results import MultiPLSResult, PLSComponentResult  # noqa: F401

    def test_in_all(self):
        from ssdiff import results
        assert "MultiPLSResult" in results.__all__
        assert "PLSComponentResult" in results.__all__
