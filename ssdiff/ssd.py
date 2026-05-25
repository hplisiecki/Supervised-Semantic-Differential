"""SSD: Supervised Semantic Differential — continuous outcome analysis."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from ssdiff.corpus import Corpus
from ssdiff.embeddings import Embeddings
from ssdiff.utils.math import _categorical_mask, f_sf, pca_fit_transform, standardize
from ssdiff.utils.vectors import build_and_normalize_doc_vectors


class SSD:
    """Supervised Semantic Differential — continuous outcome.

    Attributes
    ----------
    x : ndarray of shape (n, D)
        Matrix of per-document vectors. In the standard SSD pipeline each
        row is a **personal concept vector (PCV)**; in the lexical-norm mode
        each row is a single word embedding.
    y : ndarray of shape (n,)
        Outcome values, numeric (continuous case) or object (group case),
        with NaN / empty entries dropped.

    Builds document vectors from corpus + lexicon, then fit with a backend:

    >>> emb = Embeddings.load("model.ssdembed")
    >>> corpus = Corpus(texts, lang="pl")
    >>> ssd = SSD(emb, corpus, y, lexicon)
    >>> pls = ssd.fit_pls()
    >>> pls.r2, pls.pvalue
    >>> pls.top_words(20)
    >>>
    >>> pcaols = ssd.fit_ols()
    >>> pcaols.r2
    >>>
    >>> groups = ssd.fit_groups()          # y as categorical labels
    >>> groups = ssd.fit_groups(median_split=True)  # median split
    """

    def __init__(
        self,
        embeddings: Embeddings,
        corpus: Corpus,
        y,
        lexicon: Sequence[str] | set[str] | None = None,
        *,
        window: int = 3,
        sif_a: float = 1e-3,
        use_full_doc: bool = False,
    ) -> None:
        """Build a **personal concept vector (PCV)** for each document via
        SIF-weighted aggregation of local contexts around lexicon occurrences,
        preparing data for PLS or PCA+OLS fitting.

        Parameters
        ----------
        embeddings : Embeddings
            Word embeddings instance providing the vector space.
        corpus : Corpus
            Tokenized corpus (``Corpus`` instance) aligned with ``y``.
        y : array-like of float
            Outcome variable. Entries with NaN are silently dropped together
            with the corresponding documents.
        lexicon : sequence or set of str, optional
            Seed words used for context-window extraction. Required in the
            default seed-context mode; ignored when ``use_full_doc=True``.
        window : int, default 3
            Context window size (tokens) around each seed word.
        sif_a : float, default 1e-3
            SIF smoothing parameter for document-vector weighting.
        use_full_doc : bool, default False
            If True, use full-document vectors instead of seed-windowed
            contexts.

        Raises
        ------
        ValueError
            If ``len(y) != len(corpus)``, or if ``lexicon`` is empty and
            ``use_full_doc=False``.
        """
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if sif_a <= 0:
            raise ValueError(f"sif_a must be > 0, got {sif_a}")

        if not embeddings.l2_normalized:
            raise RuntimeError(
                "SSD requires L2-normalised embeddings. Call "
                "emb.normalize(l2=True, abtt=N) before constructing SSD."
            )

        if getattr(embeddings, "_partial", False) and not getattr(embeddings, "_corpus_attached", False):
            # Materialise the corpus tail into a private view; never mutate the
            # caller's embedding, so one partial emb stays reusable across corpora.
            embeddings = embeddings.with_corpus(corpus)

        self.embeddings = embeddings
        self.corpus = corpus
        self.lexicon = set(lexicon) if lexicon is not None else set()
        if not use_full_doc and not self.lexicon:
            raise ValueError(
                "lexicon is required when use_full_doc=False "
                "(seed-context mode needs seed words)."
            )
        self.window = window
        self.sif_a = sif_a
        self.lang = getattr(corpus, "lang", None)

        # Build doc vectors
        docs = corpus.docs

        # Accept both numeric and categorical y
        y_raw = np.asarray(y)
        try:
            y_num = np.asarray(y, dtype=float)
            is_numeric = True
        except (ValueError, TypeError):
            y_num = None
            is_numeric = False

        if len(y_raw) != len(docs):
            raise ValueError(
                f"len(y)={len(y_raw)} != len(docs)={len(docs)}. "
                "y and corpus must have the same length."
            )

        # Filter invalid y entries
        if is_numeric:
            finite = np.isfinite(y_num)
            if not finite.all():
                docs = [d for d, m in zip(docs, finite) if m]
                y_num = y_num[finite]
            y_clean = y_num
        else:
            # For categorical: drop None, "", NaN
            valid = _categorical_mask(y_raw)
            if not valid.all():
                docs = [d for d, v in zip(docs, valid) if v]
                y_raw = y_raw[valid]
            y_clean = y_raw

        X, keep = build_and_normalize_doc_vectors(
            docs, embeddings, self.lexicon,
            window=window, sif_a=sif_a, use_full_doc=use_full_doc,
        )

        self._keep_mask = keep
        self.n_raw = len(keep)
        self.n_kept = int(keep.sum())
        self.n_dropped = self.n_raw - self.n_kept

        self.x = np.asarray(X, dtype=np.float64)
        self.y = y_clean[keep]
        self.is_numeric = is_numeric

    # ── Shared helpers ─────────────────────────────────────────

    def _compute_fit_stats(
        self, ys: np.ndarray, y_pred: np.ndarray, p: int,
    ) -> dict:
        """Compute R², R²_adj, and F p-value from predictions."""
        n = len(ys)
        resid = ys - y_pred
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))

        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        r2_adj = (
            1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)
            if n - p - 1 > 0 else float("nan")
        )

        msr = (ss_tot - ss_res) / max(p, 1)
        mse = ss_res / (n - p - 1) if n - p - 1 > 0 else float("inf")
        f_stat_val = msr / mse if np.isfinite(mse) and mse > 0 else 0.0
        f_pvalue = (
            f_sf(f_stat_val, p, n - p - 1)
            if np.isfinite(mse) and n - p - 1 > 0
            else float("nan")
        )

        return {
            "r2": r2,
            "r2_adj": r2_adj,
            "f_pvalue": f_pvalue,
        }

    def _orient_beta(
        self, beta: np.ndarray, ys: np.ndarray,
    ) -> np.ndarray:
        """Orient beta so higher alignment → higher outcome."""
        yhat_std = (self.x @ beta).ravel()
        if float(np.std(yhat_std)) > 0:
            c = float(np.corrcoef(ys, yhat_std)[0, 1])
            corr = c if np.isfinite(c) else 0.0
        else:
            corr = 0.0
        if corr < 0:
            beta = -beta
        return beta

    def _base_result_kwargs(self) -> dict:
        """Common kwargs for result object construction.

        Does not include _y_mean / _y_scale — callers add those
        after computing standardization in their fit method.
        """
        return {
            "embeddings": self.embeddings,
            "corpus": self.corpus,
            "lexicon": self.lexicon,
            "window": self.window,
            "sif_a": self.sif_a,
            "lang": self.lang,
            "x": self.x,
            "keep_mask": self._keep_mask,
            "n_raw": self.n_raw,
            "n_kept": self.n_kept,
            "n_dropped": self.n_dropped,
            "y": self.y,
        }

    # ── PLS backend ────────────────────────────────────────────

    def fit_pls(
        self,
        *,
        k: int | Literal["auto"] = "auto",
        k_max: int = 5,
        n_splits: int = 50,
        random_state: int = 2137,
        verbose: bool = False,
    ):
        """Fit PLS1 NIPALS and return PLSResult.

        Significance is always tested with the split_nb method
        (Lenartowicz, 2026). For ``k="auto"`` the reported p-value is the
        **same sample k=1 confirmatory** test — independent of the
        ``find_k_optimal`` selection that picks ``k_star`` for the actual
        fit. When ``k_star == 1`` the find_k chain entry at index 0 is
        reused as the k=1 same sample p (it is the same statistic).

        Parameters
        ----------
        k : int or ``"auto"``, default ``"auto"``
            Number of PLS components.

            - ``int`` — fit at exactly this k; p-value comes from
              ``plskit.pls1_confirmatory_test`` at this k.
            - ``"auto"`` — let ``plskit.pls1_find_k_optimal`` pick
              ``k_star`` (selector ``"r2_se"``); p-value is the same sample
              k=1 confirmatory split_nb statistic.
        k_max : int, default 5
            Cap for ``k="auto"``. Further clamped to ``min(k_max, n-1, D)``.
            Ignored when ``k`` is an int.
        n_splits : int, default 50
            Random splits for the split_nb test.
        random_state : int, default 2137
            Random seed.
        verbose : bool

        Returns
        -------
        PLSResult
        """
        import plskit

        from ssdiff.results.continuous_result import PLSResult

        if not self.is_numeric:
            raise ValueError(
                "fit_pls() requires numeric y. This SSD was constructed with "
                "categorical labels — use fit_groups() instead."
            )

        y_std = float(np.std(self.y))
        if y_std <= 1e-12:
            raise ValueError(
                f"fit_pls() requires y with nonzero variance (std={y_std:.3g}). "
                "PLS1 is undefined when y is constant."
            )

        ys_2d, _y_mean, _y_scale = standardize(self.y.reshape(-1, 1))
        ys = ys_2d.ravel()
        Xs, X_mean, X_scale = standardize(self.x)

        n_kept, D = Xs.shape

        find_k_raw = None
        if k == "auto":
            eff_k_max = max(min(int(k_max), n_kept - 1, D), 1)
            if verbose:
                print(
                    f"[fit_pls] auto-selecting k via find_k_optimal "
                    f"(selector=r2_se, k_max={eff_k_max})"
                )
            find_k_raw = plskit.pls1_find_k_optimal(
                Xs, ys, eff_k_max,
                selector="r2_se",
                diagnostic="split_nb",
                args={"n_folds": 5, "n_splits": int(n_splits)},
                pre_standardized=True,
                seed=random_state,
                verbose=verbose,
            )
            k_star = int(find_k_raw.k_star)
            if verbose:
                print(f"[fit_pls] k_star = {k_star}")
            if k_star == 1:
                pvalue = float(find_k_raw.pvalues[0])
                if verbose:
                    print(
                        f"[fit_pls] k_star = 1 — reusing chain[0] as the "
                        f"k=1 p (split_nb)"
                    )
                    print(f"[fit_pls] p-value (k=1) = {pvalue:.6g}")
            else:
                if verbose:
                    chain = ", ".join(
                        f"{p:.3g}" for p in find_k_raw.pvalues[:k_star]
                    )
                    path_max = float(np.nanmax(find_k_raw.pvalues[:k_star]))
                    print(
                        f"[fit_pls] diagnostic chain (split_nb, k=1..{k_star}): "
                        f"{chain}"
                    )
                    print(
                        f"        path-max: {path_max:.3g}  "
                        f"[same-sample]"
                    )
                    print(
                        f"[fit_pls] running k=1 confirmatory test "
                        f"(split_nb, n_splits={n_splits})"
                    )
                r = plskit.pls1_confirmatory_test(
                    Xs, ys, 1,
                    method="split_nb",
                    args={"n_splits": int(n_splits)},
                    pre_standardized=True,
                    seed=random_state,
                    verbose=verbose,
                )
                pvalue = float(r.pvalue)
                if verbose:
                    print(f"[fit_pls] p-value (k=1) = {pvalue:.6g}")
            n_comp = k_star
            p_at_k = 1
        elif isinstance(k, int):
            if k < 1:
                raise ValueError(f"k must be >= 1, got {k}")
            n_comp = min(int(k), n_kept - 1, D)
            if n_comp < 1:
                raise ValueError(
                    f"Cannot fit PLS: need at least 2 samples and 1 feature, "
                    f"got n={n_kept}, features={D}, requested k={k}"
                )
            r = plskit.pls1_confirmatory_test(
                Xs, ys, n_comp,
                method="split_nb",
                args={"n_splits": int(n_splits)},
                pre_standardized=True,
                seed=random_state,
                verbose=verbose,
            )
            pvalue = float(r.pvalue)
            p_at_k = n_comp
        else:
            raise ValueError(f"k must be int or 'auto', got {k!r}")

        m = plskit.pls1_fit(
            Xs, ys, k=n_comp, pre_standardized=True, seed=random_state,
        )
        T, P, W, Q, coef = m.T, m.P, m.W, m.Q, m.coef

        y_pred = Xs @ coef
        stats = self._compute_fit_stats(ys, y_pred, W.shape[1])

        scale = np.where(X_scale > 1e-12, X_scale, 1.0)
        beta = coef / scale
        beta = self._orient_beta(beta, ys)

        kwargs = self._base_result_kwargs()
        kwargs["_y_mean"] = _y_mean
        kwargs["_y_scale"] = _y_scale

        test_info = {
            "pvalue": pvalue,
            "n_splits": int(n_splits),
            "random_state": random_state,
        }

        return PLSResult(
            beta=beta,
            pvalue=pvalue,
            r2=stats["r2"],
            fit_info={
                "n_components": n_comp,
                "p_at_k": p_at_k,
                "n_splits": int(n_splits),
                "random_state": random_state,
            },
            raw_diagnostics={
                "find_k_result": find_k_raw,
                "component_scores": T,
                "component_weights": W,
            },
            test_name="split_nb",
            test_info=test_info,
            **kwargs,
        )

    # ── Multi-dimensional PLS backend ──────────────────────────

    def fit_multipls(
        self,
        *,
        k: int | Literal["auto"] = "auto",
        k_max: int = 5,
        rotate: Literal["raw", "varimax"] = "varimax",
        rotation_vocab: int | None = 50_000,
        n_splits: int = 50,
        random_state: int = 2137,
        verbose: bool = False,
    ):
        """Fit PLS1, rotate the W-subspace, return a :class:`MultiPLSResult`.

        Container-level p-value follows :meth:`fit_pls` (same sample k=1 for
        ``k="auto"``, confirmatory at ``k`` for ``k=int``). In addition,
        each rotated dim leaf gains a diagnostic ``pvalue`` field: for
        ``k="auto"`` these are read off the ``find_k_optimal`` chain;
        for ``k=int`` they come from ``n_comp`` individual
        ``pls1_confirmatory_test(ki)`` calls (ki=1..n_comp). In both
        cases the chain is remapped via ``mpls_fit``'s ``order``
        permutation.

        Parameters
        ----------
        k : int or ``"auto"``, default ``"auto"``
            Same semantics as :meth:`fit_pls`. Raises if NIPALS deflation
            produces fewer than the resolved count.
        k_max : int, default 5
            Cap for ``k="auto"``. Clamped to ``min(k_max, n-1, D)``.
        rotate : {"raw", "varimax"}, default "varimax"
            Rotation applied to the W-subspace.
        rotation_vocab : int or None, default 50_000
            Number of leading vocabulary rows fed to varimax as the
            simple-structure target. Vocab rows are assumed to be
            frequency-ranked (the word2vec/GloVe convention), so the
            top-N is the well-estimated head; the long tail of rare-word
            vectors is noisy and pulls varimax toward fitting noise.
            ``None`` uses the full vocabulary. Clamped silently to the
            available row count. No-op for ``rotate="raw"`` (which
            doesn't touch the embedding matrix). Matches the 50k
            convention shared with ``cluster_top_neighbors`` and
            ``Embeddings.load(ram_efficient=True)``.
        n_splits : int, default 50
            Random splits for the split_nb test.
        random_state, verbose
            Same as :meth:`fit_pls`.

        Returns
        -------
        MultiPLSResult
        """
        import plskit

        from ssdiff.backends.pls import mpls_fit
        from ssdiff.results.multi_pls_result import MultiPLSResult

        if not self.is_numeric:
            raise ValueError(
                "fit_multipls() requires numeric y. This SSD was constructed "
                "with categorical labels — use fit_groups() instead."
            )
        if self.embeddings is None:
            raise RuntimeError(
                "fit_multipls() rotates against the embedding vocabulary, "
                "which requires Embeddings; none attached. Call result.attach("
                "embeddings=...) or re-construct SSD with embeddings."
            )

        Xs, X_mean, X_scale = standardize(self.x)
        ys_2d, _, _ = standardize(self.y.reshape(-1, 1))
        ys = ys_2d.ravel()
        n_kept, D = Xs.shape

        if k == "auto":
            eff_k_max = max(min(int(k_max), n_kept - 1, D), 1)
            if verbose:
                print(
                    f"[fit_multipls] auto-selecting k via find_k_optimal "
                    f"(selector=r2_se, k_max={eff_k_max})"
                )
            find_k_raw = plskit.pls1_find_k_optimal(
                Xs, ys, eff_k_max,
                selector="r2_se",
                diagnostic="split_nb",
                args={"n_folds": 5, "n_splits": int(n_splits)},
                pre_standardized=True,
                seed=random_state,
                verbose=verbose,
            )
            k_star = int(find_k_raw.k_star)
            if verbose:
                print(f"[fit_multipls] k_star = {k_star}")
            if k_star == 1:
                pvalue = float(find_k_raw.pvalues[0])
                if verbose:
                    print(
                        f"[fit_multipls] k_star = 1 — reusing chain[0] as "
                        f"the k=1 p (split_nb)"
                    )
                    print(f"[fit_multipls] p-value (k=1) = {pvalue:.6g}")
            else:
                if verbose:
                    chain = ", ".join(
                        f"{p:.3g}" for p in find_k_raw.pvalues[:k_star]
                    )
                    print(
                        f"[fit_multipls] diagnostic chain "
                        f"(split_nb, k=1..{k_star}): {chain}"
                    )
                    print(
                        f"[fit_multipls] running k=1 confirmatory test "
                        f"(split_nb, n_splits={n_splits})"
                    )
                r = plskit.pls1_confirmatory_test(
                    Xs, ys, 1,
                    method="split_nb",
                    args={"n_splits": int(n_splits)},
                    pre_standardized=True,
                    seed=random_state,
                    verbose=verbose,
                )
                pvalue = float(r.pvalue)
                if verbose:
                    print(f"[fit_multipls] p-value (k=1) = {pvalue:.6g}")
            n_comp = k_star
            p_at_k = 1
            chain_pvalues = np.asarray(
                find_k_raw.pvalues[:n_comp], dtype=float,
            )
        elif isinstance(k, int):
            if k < 1:
                raise ValueError(f"k must be >= 1, got {k}")
            n_comp = min(int(k), n_kept - 1, D)
            if n_comp < 1:
                raise ValueError(
                    f"Cannot fit PLS: need at least 2 samples and 1 feature, "
                    f"got n={n_kept}, features={D}, requested k={k}"
                )
            r = plskit.pls1_confirmatory_test(
                Xs, ys, n_comp,
                method="split_nb",
                args={"n_splits": int(n_splits)},
                pre_standardized=True,
                seed=random_state,
                verbose=verbose,
            )
            pvalue = float(r.pvalue)
            p_at_k = n_comp
            chain_pvalues = np.array([
                float(plskit.pls1_confirmatory_test(
                    Xs, ys, ki,
                    method="split_nb",
                    args={"n_splits": int(n_splits)},
                    pre_standardized=True,
                    seed=random_state,
                    verbose=False,
                ).pvalue)
                for ki in range(1, n_comp + 1)
            ], dtype=float)
        else:
            raise ValueError(f"k must be int or 'auto', got {k!r}")

        emb = self.embeddings.vectors
        emb_dtype = emb.dtype
        X_mean_v = X_mean.astype(emb_dtype, copy=False)
        scale_v = X_scale.astype(emb_dtype, copy=False)

        if rotation_vocab is None:
            n_rot = int(emb.shape[0])
        else:
            n_rot = min(int(rotation_vocab), int(emb.shape[0]))
        emb_rot = emb if n_rot == emb.shape[0] else emb[:n_rot]

        def project_target(W: np.ndarray) -> np.ndarray:
            M = (W / scale_v[:, None]).astype(emb_dtype, copy=False)
            offset = X_mean_v @ M
            L = emb_rot @ M
            L -= offset
            return L

        mp_out = mpls_fit(
            Xs, ys,
            n_components=n_comp, rotate=rotate,
            E_target=project_target,
        )
        mp_out["rotation_meta"]["rotation_vocab"] = (
            None if rotation_vocab is None else n_rot
        )

        order = np.asarray(mp_out["order"], dtype=int)
        per_dim_pvalues = chain_pvalues[order]

        y_pred = Xs @ mp_out["beta_combined"]
        stats = self._compute_fit_stats(ys, y_pred, n_comp)
        r2 = stats["r2"]

        test_info = {
            "pvalue": pvalue,
            "n_splits": int(n_splits),
            "random_state": random_state,
        }

        return MultiPLSResult(
            x=self.x, y=self.y,
            W=mp_out["W"], P=mp_out["P"], Q=mp_out["Q"],
            W_rot=mp_out["W_rot"], T_rot=mp_out["T_rot"],
            beta_combined=mp_out["beta_combined"],
            n_components=n_comp,
            rotation_meta=mp_out["rotation_meta"],
            r2=r2,
            per_dim_pvalues=per_dim_pvalues,
            p_at_k=p_at_k,
            test_name="split_nb",
            test_info=test_info,
            embeddings=self.embeddings,
            corpus=self.corpus,
            lexicon=self.lexicon,
            window=self.window,
            sif_a=self.sif_a,
            lang=self.lang,
            random_state=random_state,
        )

    # ── PCA + OLS backend ─────────────────────────────────────

    def fit_ols(
        self,
        *,
        fixed_k: int | None = None,
        k_min: int = 2,
        k_max: int = 120,
        k_step: int = 2,
        verbose: bool = False,
        pca_sweep_args: dict | None = None,
    ):
        """Fit PCA + OLS and return PCAOLSResult.

        Parameters
        ----------
        fixed_k : int or None
            Fixed number of PCA components. None = auto-select via sweep.
        k_min, k_max, k_step : int
            Range for PCA-K sweep when fixed_k is None.
        verbose : bool
            Print progress.
        pca_sweep_args : dict, optional
            Extra keyword arguments forwarded to
            :func:`ssdiff.backends.pca_sweep.pca_sweep` when ``fixed_k is None``.
            Useful for tuning the sweep's clustering / scoring (e.g.
            ``cluster_topn``, ``cluster_k_min``, ``cluster_k_max``,
            ``weight_by_size``, ``auck_radius``) or for enabling table output
            (``save_tables``, ``out_dir``, ``prefix``).
            Keys that ``fit_ols`` derives internally (``Xs``, ``X_scale``,
            ``x``, ``ys``, ``embeddings``, ``pca_k_values``, ``verbose``,
            ``lang``) cannot be overridden and raise ``ValueError`` if passed.
            Ignored when ``fixed_k`` is set.

        Returns
        -------
        PCAOLSResult
        """
        from ssdiff.results.continuous_result import PCAOLSResult

        if not self.is_numeric:
            raise ValueError(
                "fit_ols() requires numeric y. This SSD was constructed with "
                "categorical labels — use fit_groups() instead."
            )

        # Standardize y (deferred from __init__)
        ys_2d, _y_mean, _y_scale = standardize(self.y.reshape(-1, 1))
        ys = ys_2d.ravel()

        # Standardize X
        Xs, X_mean, X_scale = standardize(self.x)

        if fixed_k is None:
            from ssdiff.backends.pca_sweep import pca_sweep
            reserved = {
                "Xs", "X_scale", "x", "ys", "embeddings",
                "pca_k_values", "verbose", "lang",
            }
            extra = dict(pca_sweep_args) if pca_sweep_args else {}
            collisions = reserved & extra.keys()
            if collisions:
                raise ValueError(
                    f"pca_sweep_args cannot override fit_ols-managed keys: "
                    f"{sorted(collisions)}. Use the corresponding fit_ols "
                    f"parameter (k_min/k_max/k_step, verbose) instead."
                )
            sweep_result = pca_sweep(
                Xs=Xs,
                X_scale=X_scale,
                x=self.x,
                ys=ys,
                embeddings=self.embeddings,
                pca_k_values=list(range(k_min, k_max + 1, k_step)),
                verbose=verbose,
                lang=self.lang,
                **extra,
            )
            n_pca = sweep_result.best_k
        else:
            n_pca = int(fixed_k)
            sweep_result = None

        max_comp = min(n_pca, Xs.shape[0], Xs.shape[1])

        # PCA via SVD
        z, components, _ = pca_fit_transform(Xs, max_comp)

        # OLS in PCA space
        w_reg = np.linalg.solve(z.T @ z, z.T @ ys)
        y_pred = z @ w_reg
        stats = self._compute_fit_stats(ys, y_pred, max_comp)

        # Back-project to doc space
        beta_std = components.T @ w_reg
        scale = np.where(X_scale > 1e-12, X_scale, 1.0)
        beta = beta_std / scale

        # Orient beta
        beta = self._orient_beta(beta, ys)

        kwargs = self._base_result_kwargs()
        kwargs["_y_mean"] = _y_mean
        kwargs["_y_scale"] = _y_scale

        _sweep_rows = None
        if sweep_result is not None:
            _sweep_rows = [
                {**r, "k": r["PCA_K"]} for r in sweep_result.df_joined
            ]

        return PCAOLSResult(
            beta=beta,
            pvalue=stats["f_pvalue"],
            r2=stats["r2"],
            r2_adj=stats["r2_adj"],
            fit_info={
                "n_components": n_pca,
                "k_min": k_min if sweep_result is not None else None,
                "k_max": k_max if sweep_result is not None else None,
                "k_step": k_step if sweep_result is not None else None,
                "best_k": sweep_result.best_k if sweep_result is not None else None,
                "pca_k_source": "sweep" if fixed_k is None else "fixed",
            },
            raw_diagnostics={
                "sweep_result": sweep_result,
                "pca_components": components,
                "pca_weights": w_reg,
            },
            sweep=_sweep_rows,
            test_name="f_test",
            test_info={"pvalue": stats["f_pvalue"]},
            **kwargs,
        )

    # ── Group comparison backend ──────────────────────────────

    def fit_groups(
        self,
        *,
        median_split: bool = False,
        n_perm: int = 5000,
        correction: Literal["holm", "bonferroni", "fdr_bh", "none"] = "holm",
        random_state: int = 2137,
        verbose: bool = False,
    ):
        """Fit group comparison using y as group labels.

        Parameters
        ----------
        median_split : bool, default False
            If True, split y into "low"/"high" at the median.
            If False, treat y values as categorical group labels.
        n_perm : int, default 5000
            Number of permutations for significance tests.
        correction : str, default "holm"
            P-value correction method: "holm", "bonferroni", "fdr_bh",
            or "none".
        random_state : int, default 2137
            Random seed for reproducibility.
        verbose : bool
            Print progress and diagnostics.

        Returns
        -------
        GroupResult

        Raises
        ------
        ValueError
            If fewer than 2 groups remain after filtering.
        """
        from ssdiff.backends.group import (
            filter_small_groups,
            unified_permutation_test,
        )
        from ssdiff.backends.group import (
            median_split as _median_split,
        )
        from ssdiff.results.group_result import GroupResult
        from ssdiff.results.schema import Pair

        # Local copies — never mutate self.x or self.y
        x_local = self.x.copy()

        if median_split:
            groups_local = _median_split(self.y, random_state=random_state)
        else:
            groups_local = np.asarray(self.y, dtype=object)

        # Small-group filtering (only when not median_split)
        n_group_dropped = 0
        if not median_split:
            x_local, groups_local, n_group_dropped = filter_small_groups(
                x_local, groups_local,
            )

        group_labels = sorted(set(groups_local), key=str)
        G = len(group_labels)
        if G < 2:
            raise ValueError(
                f"Need at least 2 groups after filtering, got {G}"
            )

        # Run unified permutation test
        test_result = unified_permutation_test(
            x_local, groups_local, group_labels,
            n_perm=n_perm,
            correction=correction,
            random_state=random_state,
            verbose=verbose,
        )

        # Build list[Pair] from backend's pairwise dict
        pairs = []
        for (g1, g2), pw in test_result["pairwise"].items():
            pairs.append(Pair(
                contrast=f"{g1}_{g2}",
                g1=str(g1),
                g2=str(g2),
                T=float(pw["T"]),
                p_raw=float(pw["p_raw"]),
                p_corrected=float(pw["p_corrected"]),
                cohens_d=float(pw["cohens_d"]),
                n_g1=int(pw["n_g1"]),
                n_g2=int(pw["n_g2"]),
                contrast_norm=float(pw["contrast_norm"]),
            ))

        # words / clusters / snippets are computed lazily on access by
        # GroupResult, using (embeddings, x, groups) + per-pair gradient.
        return GroupResult(
            G=G,
            n_kept=len(x_local),
            n_perm=n_perm,
            correction=correction,
            random_state=random_state,
            omnibus_T=float(test_result["omnibus_T"]),
            omnibus_p=float(test_result["omnibus_p"]),
            pairs=pairs,
            embeddings=self.embeddings,
            corpus=self.corpus,
            x=x_local,
            groups=groups_local,
            lang=self.lang,
            lexicon=self.lexicon,
            window=self.window,
            sif_a=self.sif_a,
        )

    def __repr__(self) -> str:
        dim = self.x.shape[1] if self.x.ndim == 2 else 0
        header = (
            f"SSD  n={self.n_kept:,}/{self.n_raw:,}  "
            f"D={dim}  |L|={len(self.lexicon)}  lang={self.lang}"
        )
        arrays = "  arrays:  .x  .y"
        methods = "  methods: .fit_pls()  .fit_multipls()  .fit_ols()  .fit_groups()"
        return "\n".join([header, arrays, methods])

    def _repr_html_(self) -> str:
        """Render a plain-text repr inside a ``<pre>`` block for Jupyter."""
        import html as _html
        return f"<pre>{_html.escape(repr(self))}</pre>"
