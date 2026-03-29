"""SSD: Supervised Semantic Differential — continuous outcome analysis."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ssdlite.embeddings import Embeddings
from ssdlite.corpus import Corpus
from ssdlite.utils.math import standardize, pca_fit_transform, f_sf
from ssdlite.utils.vectors import build_and_normalize_doc_vectors


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
    >>> pcaols = ssd.fit_ols()  # requires scikit-learn
    >>> pcaols.r2
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
        use_unit_beta: bool = True,
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
        use_unit_beta : bool, default True
            If True, unit-norm the beta vector before neighbor search.

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
        self.use_unit_beta = use_unit_beta
        self.lang = getattr(corpus, "lang", None) or "pl"

        # Build doc vectors
        y = np.asarray(y, dtype=float)
        docs = corpus.docs

        if len(y) != len(docs):
            raise ValueError(
                f"len(y)={len(y)} != len(docs)={len(docs)}. "
                "y and corpus must have the same length."
            )

        # Filter NaN y
        finite = np.isfinite(y)
        if not finite.all():
            docs = [d for d, m in zip(docs, finite) if m]
            y = y[finite]

        X, keep = build_and_normalize_doc_vectors(
            docs, embeddings, self.lexicon,
            window=window, sif_a=sif_a, use_full_doc=use_full_doc,
        )

        self.keep_mask = keep
        self.n_raw = len(keep)
        self.n_kept = int(keep.sum())
        self.n_dropped = self.n_raw - self.n_kept

        y_kept = y[keep]
        self.x = np.asarray(X, dtype=np.float64)
        self.y_kept = y_kept

        # Standardize y
        ys_2d, self._y_mean, self._y_scale = standardize(y_kept.reshape(-1, 1))
        self.ys = ys_2d.ravel()

    # ── Shared helpers ─────────────────────────────────────────

    def _compute_fit_stats(self, y_pred: np.ndarray, p: int) -> dict:
        """Compute R², R²_adj, and F p-value from predictions."""
        ys = self.ys
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

    def _orient_beta(self, beta: np.ndarray) -> np.ndarray:
        """Orient beta so higher alignment → higher outcome."""
        yhat_std = (self.x @ beta).ravel()
        if float(np.std(yhat_std)) > 0:
            c = float(np.corrcoef(self.ys, yhat_std)[0, 1])
            corr = c if np.isfinite(c) else 0.0
        else:
            corr = 0.0
        if corr < 0:
            beta = -beta
        return beta

    def _base_result_kwargs(self) -> dict:
        """Common kwargs for result object construction."""
        return {
            "kv": self.kv,
            "lexicon": self.lexicon,
            "window": self.window,
            "sif_a": self.sif_a,
            "use_unit_beta": self.use_unit_beta,
            "lang": self.lang,
            "x": self.x,
            "keep_mask": self.keep_mask,
            "n_raw": self.n_raw,
            "n_kept": self.n_kept,
            "n_dropped": self.n_dropped,
            "y_kept": self.y_kept,
            "_y_mean": self._y_mean,
            "_y_scale": self._y_scale,
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
        random_state: int = 42,
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
        from ssdlite.backends.pls import pls1_fit, pls1_cv_select
        from ssdlite.results import PLSResult

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
        T, P, W, Q, coef = pls1_fit(X_for_pls, self.ys, max_comp)
        actual_comp = W.shape[1]

        # Statistics
        y_pred = X_for_pls @ coef
        stats = self._compute_fit_stats(y_pred, actual_comp)

        # Back-project to embedding space
        if pca_preprocess_components is not None:
            coef_emb = pca_preprocess_components.T @ coef
        else:
            coef_emb = coef
        scale = np.where(X_scale > 1e-12, X_scale, 1.0)
        beta = coef_emb / scale

        # Orient beta
        beta = self._orient_beta(beta)

        # Resolve p_method
        resolved = p_method
        if resolved == "auto":
            resolved = "split" if n_comp == 1 else "perm"

        perm_null = None
        split_mean_r = None

        if resolved == "perm":
            from ssdlite.backends.pls import pls1_permutation_test
            p_val, _, cv_r2_null = pls1_permutation_test(
                self.x, self.y_kept, n_comp,
                n_perm=n_perm, seed=random_state, verbose=verbose,
                pca_k=pca_k,
            )
            pvalue = p_val
            perm_null = cv_r2_null
        elif resolved == "split":
            from ssdlite.backends.pls import pls1_split_test
            pvalue, split_mean_r = pls1_split_test(
                self.x, self.y_kept, n_comp,
                n_splits=n_splits, split_ratio=split_ratio,
                seed=random_state, pca_k=pca_k,
            )
        elif resolved == "split_cal":
            from ssdlite.backends.pls import pls1_split_test_calibrated
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

        return PLSResult(
            n_components=n_comp,
            cv_result=cv_result,
            cv_scores=cv_scores,
            perm_null=perm_null,
            pca_k=pca_k,
            p_method=resolved,
            split_mean_r=split_mean_r,
            beta=beta,
            pvalue=pvalue,
            r2=stats["r2"],
            r2_adj=stats["r2_adj"],
            **self._base_result_kwargs(),
        )

    # ── PCA + OLS backend (sklearn) ───────────────────────────

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

        Uses sklearn StandardScaler and PCA to match the official ssdiff
        algorithm exactly.  Requires scikit-learn.

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
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from ssdlite.results import PCAOLSResult

        # Standardize X (sklearn, matches official)
        scaler_X = StandardScaler()
        Xs = scaler_X.fit_transform(self.x)

        if n_components is None:
            from ssdlite.backends.pca_sweep import pca_sweep
            sweep_result = pca_sweep(
                Xs=Xs,
                X_scale=scaler_X.scale_,
                x=self.x,
                ys=self.ys,
                kv=self.kv,
                use_unit_beta=self.use_unit_beta,
                pca_k_values=list(range(k_min, k_max + 1, k_step)),
                verbose=verbose,
                lang=self.lang,
            )
            n_pca = sweep_result.best_k
        else:
            n_pca = int(n_components)
            sweep_result = None

        max_comp = min(n_pca, Xs.shape[0], Xs.shape[1])

        # sklearn PCA (matches official)
        pca = PCA(n_components=max_comp, svd_solver="full")
        z = pca.fit_transform(Xs)

        # OLS in PCA space (normal equations, matches official)
        w_reg = np.linalg.solve(z.T @ z, z.T @ self.ys)
        y_pred = z @ w_reg
        stats = self._compute_fit_stats(y_pred, max_comp)

        # Back-project to doc space (matches official)
        beta_std = pca.components_.T @ w_reg
        scale = np.where(scaler_X.scale_ > 1e-12, scaler_X.scale_, 1.0)
        beta = beta_std / scale

        # Orient beta
        beta = self._orient_beta(beta)

        return PCAOLSResult(
            n_components=n_pca,
            sweep_result=sweep_result,
            beta=beta,
            pvalue=stats["f_pvalue"],
            r2=stats["r2"],
            r2_adj=stats["r2_adj"],
            **self._base_result_kwargs(),
        )

    def __repr__(self) -> str:
        return f"SSD(n_kept={self.n_kept}, n_dropped={self.n_dropped})"
