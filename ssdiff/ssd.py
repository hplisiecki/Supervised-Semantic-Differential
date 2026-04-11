"""SSD: Supervised Semantic Differential — continuous outcome analysis."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ssdiff.corpus import Corpus
from ssdiff.embeddings import Embeddings
from ssdiff.utils.math import f_sf, pca_fit_transform, standardize
from ssdiff.utils.vectors import build_and_normalize_doc_vectors


class SSD:
    """Supervised Semantic Differential — continuous outcome.

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
        lexicon: Sequence[str] | set[str],
        *,
        window: int = 3,
        sif_a: float = 1e-3,
        use_full_doc: bool = False,
    ) -> None:
        """Build document vectors from corpus and lexicon, preparing data for
        PLS or PCA+OLS fitting.

        Parameters
        ----------
        embeddings : Embeddings
            Word embeddings instance providing the vector space.
        corpus : Corpus
            Tokenized corpus (``Corpus`` instance) aligned with ``y``.
        y : array-like of float
            Outcome variable. Entries with NaN are silently dropped together
            with the corresponding documents.
        lexicon : sequence or set of str
            Seed words used for context-window extraction.
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
            If ``len(y) != len(corpus)``.
        """
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if sif_a <= 0:
            raise ValueError(f"sif_a must be > 0, got {sif_a}")

        self.kv = embeddings
        self.lexicon = set(lexicon)
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
            valid = np.array([
                v is not None and v != ""
                and (not isinstance(v, float) or np.isfinite(v))
                for v in y_raw
            ], dtype=bool)
            if not valid.all():
                docs = [d for d, v in zip(docs, valid) if v]
                y_raw = y_raw[valid]
            y_clean = y_raw

        X, keep = build_and_normalize_doc_vectors(
            docs, embeddings, self.lexicon,
            window=window, sif_a=sif_a, use_full_doc=use_full_doc,
        )

        self.keep_mask = keep
        self.n_raw = len(keep)
        self.n_kept = int(keep.sum())
        self.n_dropped = self.n_raw - self.n_kept

        y_kept = y_clean[keep]
        self.x = np.asarray(X, dtype=np.float64)
        self.y_kept = y_kept
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
            "kv": self.kv,
            "lexicon": self.lexicon,
            "window": self.window,
            "sif_a": self.sif_a,

            "lang": self.lang,
            "x": self.x,
            "keep_mask": self.keep_mask,
            "n_raw": self.n_raw,
            "n_kept": self.n_kept,
            "n_dropped": self.n_dropped,
            "y_kept": self.y_kept,
        }

    # ── PLS backend ────────────────────────────────────────────

    def fit_pls(
        self,
        *,
        n_components: int | str = 1,
        cv_folds: int = 10,
        use_1se: bool = True,
        pca_preprocess: int | str | None = None,
        p_method: str | None = "auto",
        n_perm: int = 1000,
        n_splits: int = 50,
        split_ratio: float = 0.5,
        random_state: int = 2137,
        verbose: bool = False,
    ):
        """Fit PLS1 NIPALS and return PLSResult.

        Parameters
        ----------
        n_components : int or "auto"
            Number of PLS components. Default 1. "auto" = select via CV.
        cv_folds : int
            Number of CV folds for component selection.
        use_1se : bool
            Use 1-SE rule for parsimonious component selection.
        pca_preprocess : int or str or None
            Optional PCA dim reduction before PLS (e.g., 50 or "var95").
        p_method : str or None, default "auto"
            Significance test method:

            - ``"auto"`` — ``"split"`` when n_components=1,
              ``"perm"`` otherwise.
            - ``"perm"`` — permutation test on cross-validated R².
            - ``"split"`` — split-half test with overlap-corrected
              t-test (Lenartowicz, 2026).
            - ``"split_cal"`` — permutation-calibrated split-half test.
            - ``None`` — skip significance testing (p-value = NaN).
        n_perm : int
            Permutation iterations for ``"perm"`` and ``"split_cal"``.
        n_splits : int
            Number of random splits for ``"split"`` and ``"split_cal"``.
        split_ratio : float
            Train fraction for ``"split"`` and ``"split_cal"``.
        random_state : int
            Random seed.
        verbose : bool
            Print progress.

        Returns
        -------
        PLSResult
        """
        from ssdiff.backends.pls import pls1_cv_select, pls1_fit
        from ssdiff.results import PLSResult

        if not self.is_numeric:
            raise ValueError(
                "fit_pls() requires numeric y. This SSD was constructed with "
                "categorical labels — use fit_groups() instead."
            )

        # Standardize y (deferred from __init__)
        ys_2d, _y_mean, _y_scale = standardize(self.y_kept.reshape(-1, 1))
        ys = ys_2d.ravel()

        # Standardize X
        Xs, X_mean, X_scale = standardize(self.x)

        # Optional PCA preprocessing
        if pca_preprocess is not None:
            n, D = Xs.shape
            if isinstance(pca_preprocess, str) and pca_preprocess.startswith("var"):
                try:
                    target = float(pca_preprocess[3:]) / 100.0
                except ValueError:
                    raise ValueError(
                        f"pca_preprocess={pca_preprocess!r} must be 'varNN' "
                        f"where NN is a number (e.g. 'var95')"
                    ) from None
                max_k = min(n - 1, D)
                Z_full, _, evr_full = pca_fit_transform(Xs, max_k)
                cum_var = np.cumsum(evr_full)
                pca_k = min(int(np.searchsorted(cum_var, target) + 1), max_k)
            else:
                pca_k = int(pca_preprocess)
            pca_k = min(pca_k, n - 1, D)
            Z_pca, pca_comps, _ = pca_fit_transform(Xs, pca_k)
            X_for_pls = Z_pca
            pca_preprocess_components = pca_comps
        else:
            X_for_pls = Xs
            pca_k = None
            pca_preprocess_components = None

        # Component selection
        if n_components is None or n_components == "auto":
            cv_result = pls1_cv_select(
                self.x, self.y_kept,
                max_components=15,
                n_folds=cv_folds,
                seed=random_state,
                use_1se_rule=use_1se,
                verbose=verbose,
                pca_k=pca_k,
            )
            n_comp = cv_result.best_n_components
            cv_scores = cv_result.cv_scores
        else:
            n_comp = int(n_components)
            cv_result = None
            cv_scores = None

        # Fit PLS
        n = X_for_pls.shape[0]
        max_comp = min(n_comp, n - 1, X_for_pls.shape[1])
        T, P, W, Q, coef = pls1_fit(X_for_pls, ys, max_comp)
        actual_comp = W.shape[1]

        # Statistics
        y_pred = X_for_pls @ coef
        stats = self._compute_fit_stats(ys, y_pred, actual_comp)

        # Back-project to embedding space
        if pca_preprocess_components is not None:
            coef_emb = pca_preprocess_components.T @ coef
        else:
            coef_emb = coef
        scale = np.where(X_scale > 1e-12, X_scale, 1.0)
        beta = coef_emb / scale

        # Orient beta
        beta = self._orient_beta(beta, ys)

        # Resolve p_method
        resolved = p_method
        if resolved == "auto":
            resolved = "split" if n_comp == 1 else "perm"

        perm_null = None
        split_mean_r = None

        if resolved == "perm":
            from ssdiff.backends.pls import pls1_permutation_test
            p_val, _, cv_r2_null = pls1_permutation_test(
                self.x, self.y_kept, n_comp,
                n_perm=n_perm, seed=random_state, verbose=verbose,
                pca_k=pca_k,
            )
            pvalue = p_val
            perm_null = cv_r2_null
        elif resolved == "split":
            from ssdiff.backends.pls import pls1_split_test
            pvalue, split_mean_r = pls1_split_test(
                self.x, self.y_kept, n_comp,
                n_splits=n_splits, split_ratio=split_ratio,
                seed=random_state, pca_k=pca_k,
            )
        elif resolved == "split_cal":
            from ssdiff.backends.pls import pls1_split_test_calibrated
            pvalue, split_mean_r = pls1_split_test_calibrated(
                self.x, self.y_kept, n_comp,
                n_splits=n_splits, split_ratio=split_ratio,
                n_perm=n_perm, seed=random_state, pca_k=pca_k,
                verbose=verbose,
            )
        elif resolved is None:
            pvalue = float("nan")
        else:
            raise ValueError(
                f"Unknown p_method {p_method!r}. "
                "Choose 'perm', 'split', 'split_cal', or None."
            )

        kwargs = self._base_result_kwargs()
        kwargs["_y_mean"] = _y_mean
        kwargs["_y_scale"] = _y_scale

        return PLSResult(
            n_components=n_comp,
            cv_result=cv_result,
            cv_scores=cv_scores,
            perm_null=perm_null,
            pca_k=pca_k,
            p_method=resolved,
            split_mean_r=split_mean_r,
            random_state=random_state,
            n_perm=n_perm,
            n_splits=n_splits,
            split_ratio=split_ratio,
            beta=beta,
            pvalue=pvalue,
            r2=stats["r2"],
            **kwargs,
        )

    # ── PCA + OLS backend ─────────────────────────────────────

    def fit_ols(
        self,
        *,
        n_components: int | None = None,
        k_min: int = 20,
        k_max: int = 120,
        k_step: int = 2,
        verbose: bool = False,
    ):
        """Fit PCA + OLS and return PCAOLSResult.

        Parameters
        ----------
        n_components : int or None
            Number of PCA components. None = auto-select via sweep.
        k_min, k_max, k_step : int
            Range for PCA-K sweep when n_components is None.
        verbose : bool
            Print progress.

        Returns
        -------
        PCAOLSResult
        """
        from ssdiff.results import PCAOLSResult

        if not self.is_numeric:
            raise ValueError(
                "fit_ols() requires numeric y. This SSD was constructed with "
                "categorical labels — use fit_groups() instead."
            )

        # Standardize y (deferred from __init__)
        ys_2d, _y_mean, _y_scale = standardize(self.y_kept.reshape(-1, 1))
        ys = ys_2d.ravel()

        # Standardize X
        Xs, X_mean, X_scale = standardize(self.x)

        if n_components is None:
            from ssdiff.backends.pca_sweep import pca_sweep
            sweep_result = pca_sweep(
                Xs=Xs,
                X_scale=X_scale,
                x=self.x,
                ys=ys,
                kv=self.kv,
                pca_k_values=list(range(k_min, k_max + 1, k_step)),
                verbose=verbose,
                lang=self.lang,
            )
            n_pca = sweep_result.best_k
        else:
            n_pca = int(n_components)
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

        return PCAOLSResult(
            n_components=n_pca,
            sweep_result=sweep_result,
            k_min=k_min if sweep_result is not None else None,
            k_max=k_max if sweep_result is not None else None,
            k_step=k_step if sweep_result is not None else None,
            beta=beta,
            pvalue=stats["f_pvalue"],
            r2=stats["r2"],
            r2_adj=stats["r2_adj"],
            **kwargs,
        )

    # ── Group comparison backend ──────────────────────────────

    def fit_groups(
        self,
        *,
        median_split: bool = False,
        n_perm: int = 5000,
        correction: str = "holm",
        random_state: int = 2137,
    ):
        """Fit group comparison using y_kept as group labels.

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
        from ssdiff.results import GroupResult

        # Local copies — never mutate self.x or self.y_kept
        x_local = self.x.copy()

        if median_split:
            groups_local = _median_split(self.y_kept, random_state=random_state)
        else:
            groups_local = np.asarray(self.y_kept, dtype=object)

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
        )

        return GroupResult(
            kv=self.kv,
            lexicon=self.lexicon,
            window=self.window,
            sif_a=self.sif_a,
            lang=self.lang,
            x=x_local,
            groups_kept=groups_local,
            keep_mask=self.keep_mask,
            n_raw=self.n_raw,
            n_kept=len(x_local),
            n_dropped=self.n_dropped,
            n_group_dropped=n_group_dropped,
            omnibus_T=test_result["omnibus_T"],
            omnibus_p=test_result["omnibus_p"],
            pairwise=test_result["pairwise"],
            group_labels=group_labels,
            G=G,
            n_perm=n_perm,
            correction=correction,
            random_state=random_state,
        )

    def __repr__(self) -> str:
        return f"SSD(n_kept={self.n_kept}, n_dropped={self.n_dropped})"
