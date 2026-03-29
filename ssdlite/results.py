"""Result objects returned by SSD.fit_pls() and SSD.fit_ols()."""

from __future__ import annotations

import numpy as np

from ssdlite.utils.math import unit_vector
from ssdlite.utils.neighbors import filtered_neighbors, cluster_top_neighbors


class _SSDResultBase:
    """Shared interpretation API for all SSD result types.

    Duck-types with what snippets_along_beta expects:
    .kv, .beta_unit, .beta, .use_unit_beta, .lexicon, .window, .sif_a
    """

    def __init__(
        self,
        *,
        kv,
        lexicon: set,
        window: int,
        sif_a: float,
        use_unit_beta: bool,
        lang: str = "pl",
        x: np.ndarray,
        keep_mask: np.ndarray,
        n_raw: int,
        n_kept: int,
        n_dropped: int,
        y_kept: np.ndarray,
        _y_mean: np.ndarray,
        _y_scale: np.ndarray,
        beta: np.ndarray,
        r2: float,
        r2_adj: float,
        pvalue: float,
    ):
        self.kv = kv
        self.lexicon = lexicon
        self.window = window
        self.sif_a = sif_a
        self.use_unit_beta = use_unit_beta
        self.lang = lang

        self.x = x
        self.keep_mask = keep_mask
        self.n_raw = n_raw
        self.n_kept = n_kept
        self.n_dropped = n_dropped

        self.y_kept = y_kept
        self._y_mean = _y_mean
        self._y_scale = _y_scale

        self.beta = beta
        self.r2 = r2
        self.r2_adj = r2_adj
        self.pvalue = pvalue

        # Derived
        self.beta_unit = unit_vector(beta)
        self.beta_norm = float(np.linalg.norm(beta))

        # Cluster placeholders
        self.pos_clusters_raw = None
        self.neg_clusters_raw = None

        # Effect sizes
        self._compute_effect_sizes()

    def _compute_effect_sizes(self):
        """Compute effect-size calibration attributes from the fitted model.

        Sets the following instance attributes:

        - ``y_mean`` / ``y_std`` : outcome moments in the original scale.
        - ``cos_align`` : per-document cosine alignment to ``beta_unit``.
        - ``y_corr_pred`` : absolute Pearson correlation between observed
          and predicted *y*.
        - ``delta`` : predicted change in *y* per +0.10 cosine shift
          (``0.10 * ||beta|| * y_std``).
        - ``iqr_effect`` : predicted change in *y* across the inter-quartile
          range of cosine alignment (``IQR(cos) * ||beta|| * y_std``).
        """
        self.y_mean = float(self._y_mean[0])
        self.y_std = float(self._y_scale[0])

        # Per-doc cosine alignment to beta_unit
        x_norms = np.sqrt(np.einsum("ij,ij->i", self.x, self.x))[:, None]
        x_norms = np.maximum(x_norms, 1e-12)
        self.cos_align = ((self.x / x_norms) @ self.beta_unit).ravel()

        # |corr(y, prediction)| — scale-invariant
        yhat = (self.x @ self.beta).ravel()
        denom = float(np.std(self.y_kept) * np.std(yhat))
        if denom > 0:
            c = float(np.corrcoef(self.y_kept, yhat)[0, 1])
            self.y_corr_pred = abs(c) if np.isfinite(c) else 0.0
        else:
            self.y_corr_pred = 0.0

        # Effect per +0.10 cosine in raw y units
        self.delta = 0.10 * self.beta_norm * self.y_std

        # IQR(cos) effect in raw y units
        q75, q25 = np.percentile(self.cos_align, [75, 25])
        self.iqr_effect = float(q75 - q25) * self.beta_norm * self.y_std

    def _base_summary_lines(self) -> list[str]:
        """Common summary lines shared by PLS and PCAOLS results."""
        lines = [
            f"Docs:  {self.n_kept} kept / {self.n_raw} total ({self.n_dropped} dropped)",
        ]
        lines.append(f"R² = {self.r2:.4f}   R²_adj = {self.r2_adj:.4f}")
        lines.append("")
        lines.append("Effect sizes:")
        lines.append(f"  ‖β‖ (SD(y) per +1.0 cos) = {self.beta_norm:.4f}")
        lines.append(f"  Δy per +0.10 cos         = {self.delta:.4f}")
        lines.append(f"  IQR(cos) effect on y     = {self.iqr_effect:.4f}")
        lines.append(f"  Corr(y, ŷ)               = {self.y_corr_pred:.4f}")
        return lines

    def summary(self) -> str:
        """Human-readable model summary."""
        return "\n".join(self._base_summary_lines())

    def top_words(self, n: int = 20) -> list[dict]:
        """Top neighbor words on both poles of beta.

        Parameters
        ----------
        n : int, default 20
            Number of neighbors per pole.

        Returns
        -------
        list[dict]
            Each dict has keys: ``side`` (``"pos"`` or ``"neg"``),
            ``rank`` (int, 1-based), ``word`` (str), ``cos`` (float).
            Positive-pole entries come first, then negative.
        """
        b = self.beta_unit if self.use_unit_beta else self.beta
        out = []
        for side, vec in [("pos", b), ("neg", -b)]:
            pairs = filtered_neighbors(self.kv, vec, topn=n, lang=self.lang)
            for rank, (word, cos) in enumerate(pairs, 1):
                out.append({"side": side, "rank": rank, "word": word, "cos": float(cos)})
        return out

    def neighbors(self, side: str = "pos", n: int = 20) -> list[tuple[str, float]]:
        """Top cosine neighbors to +beta (pos) or -beta (neg).

        Parameters
        ----------
        side : str, default ``"pos"``
            ``"pos"`` for neighbors of +beta, ``"neg"`` for neighbors of -beta.
        n : int, default 20
            Number of neighbors to return.

        Returns
        -------
        list[tuple[str, float]]
            ``(word, cosine)`` tuples sorted by descending similarity.
        """
        b = self.beta_unit if self.use_unit_beta else self.beta
        vec = b if side == "pos" else -b
        return filtered_neighbors(self.kv, vec, topn=n, lang=self.lang)

    def cluster_neighbors(
        self,
        side: str = "pos",
        *,
        topn: int = 100,
        k: int | None = None,
        k_min: int = 2,
        k_max: int = 10,
        random_state: int = 13,
        min_cluster_size: int = 2,
    ) -> list[dict]:
        """Cluster top neighbors into interpretable themes.

        Uses K-Means on the top-*topn* cosine neighbors to group them
        into semantically coherent clusters.

        Parameters
        ----------
        side : str, default ``"pos"``
            ``"pos"`` for +beta neighbors, ``"neg"`` for -beta neighbors.
        topn : int, default 100
            Size of the candidate neighbor pool to cluster.
        k : int or None, default None
            Fixed number of clusters.  If ``None``, automatically selects
            the best *k* within the ``[k_min, k_max]`` range.
        k_min : int, default 2
            Minimum *k* for automatic selection.
        k_max : int, default 10
            Maximum *k* for automatic selection.
        random_state : int, default 13
            Random seed for K-Means.
        min_cluster_size : int, default 2
            Clusters smaller than this are discarded.

        Returns
        -------
        list[dict]
            Each dict has keys: ``id`` (int), ``size`` (int),
            ``centroid_cos_beta`` (float), ``coherence`` (float),
            ``words`` (list[str]).  Sorted by centroid alignment to beta.
        """
        b = self.beta_unit if self.use_unit_beta else self.beta
        clusters = cluster_top_neighbors(
            self.kv, b, use_unit_beta=self.use_unit_beta,
            topn=topn, k=k, k_min=k_min, k_max=k_max,
            random_state=random_state, min_cluster_size=min_cluster_size,
            side=side, lang=self.lang,
        )
        if side == "pos":
            self.pos_clusters_raw = clusters
        else:
            self.neg_clusters_raw = clusters
        return clusters

    def doc_scores(self) -> dict:
        """Per-document alignment scores and predictions.

        Returns
        -------
        dict
            - ``keep_mask`` : bool array, shape ``(n_raw,)``.  Which
              documents from the original corpus were kept after
              preprocessing.
            - ``cos_align`` : float array, shape ``(n_kept,)``.  Cosine
              alignment of each kept document to ``beta_unit``.
            - ``score_std`` : float array, shape ``(n_kept,)``.
              Standardized predicted scores (``X @ beta``).
            - ``yhat_raw`` : float array, shape ``(n_kept,)``.  Predicted
              outcome in the original scale
              (``y_mean + y_std * score_std``).
        """
        score_std = (self.x @ self.beta).astype(np.float64)
        yhat_raw = self.y_mean + self.y_std * score_std

        return {
            "keep_mask": self.keep_mask.copy(),
            "cos_align": self.cos_align.copy(),
            "score_std": score_std,
            "yhat_raw": yhat_raw,
        }

    def extreme_docs(
        self, k: int = 50, by: str = "predicted",
    ) -> list[dict]:
        """Select top-k and bottom-k documents by predicted or observed outcome.

        Parameters
        ----------
        k : int
            Number of extremes per side.
        by : str
            ``"predicted"`` (rank by model prediction) or ``"observed"``
            (rank by true outcome).

        Returns
        -------
        list[dict]
            Each dict: ``{idx, y_true, yhat, cos, side}``.
            ``idx`` is in kept-doc space (0 to ``n_kept - 1``).
        """
        if by not in ("predicted", "observed"):
            raise ValueError(f"`by` must be 'predicted' or 'observed', got {by!r}")

        yhat = (self.y_mean + self.y_std * (self.x @ self.beta).ravel())
        y_true = self.y_kept
        cos = self.cos_align

        signal = yhat if by == "predicted" else y_true
        k = max(0, min(k, len(signal) // 2))
        if k == 0:
            return []

        bot_idx = np.argpartition(signal, k)[:k]
        bot_sorted = bot_idx[np.argsort(signal[bot_idx])]
        top_idx = np.argpartition(signal, len(signal) - k)[-k:]
        top_sorted = top_idx[np.argsort(-signal[top_idx])]

        out = []
        for idx in bot_sorted:
            out.append({
                "idx": int(idx), "y_true": float(y_true[idx]),
                "yhat": float(yhat[idx]), "cos": float(cos[idx]),
                "side": "bottom",
            })
        for idx in top_sorted:
            out.append({
                "idx": int(idx), "y_true": float(y_true[idx]),
                "yhat": float(yhat[idx]), "cos": float(cos[idx]),
                "side": "top",
            })
        return out

    def snippets_extreme(
        self,
        pre_docs,
        *,
        k: int = 50,
        by: str = "predicted",
        top_per_side: int = 200,
        **kwargs,
    ) -> dict:
        """Extract text snippets from extreme documents.

        Chains ``extreme_docs()`` -> subset ``pre_docs`` -> ``snippets()``.

        Parameters
        ----------
        pre_docs : list[PreprocessedDoc]
            Full preprocessed documents (from ``Corpus.pre_docs``).
        k, by :
            Forwarded to ``extreme_docs()``.
        top_per_side, **kwargs :
            Forwarded to ``snippets_along_beta()``.
        """
        from ssdlite.utils.snippets import snippets_along_beta

        extremes = self.extreme_docs(k=k, by=by)
        if not extremes:
            return {"pos": [], "neg": []}

        # Map kept-space indices -> corpus-space indices
        kept_indices = set(d["idx"] for d in extremes)
        corpus_positions = np.where(self.keep_mask)[0]
        corpus_indices = {int(corpus_positions[i]) for i in kept_indices}

        subset = [
            doc for i, doc in enumerate(pre_docs)
            if i in corpus_indices
        ]

        return snippets_along_beta(
            pre_docs=subset, ssd=self,
            top_per_side=top_per_side, **kwargs,
        )

    def misdiagnosed(
        self, k: int = 20, side: str = "both",
    ) -> list[dict]:
        """Documents where model predictions diverge most from observed.

        Parameters
        ----------
        k : int
            Number of docs per side.
        side : str
            ``"both"`` (k over + k under), ``"over"`` (model over-predicts),
            ``"under"`` (model under-predicts).

        Returns
        -------
        list[dict]
            Each dict: ``{idx, y_true, yhat, cos, residual, side}``.
            ``residual = yhat - y_true`` (positive = over-predicted).
        """
        if side not in ("both", "over", "under"):
            raise ValueError(f"`side` must be 'both', 'over', or 'under', got {side!r}")

        yhat = (self.y_mean + self.y_std * (self.x @ self.beta).ravel())
        y_true = self.y_kept
        cos = self.cos_align
        residual = yhat - y_true

        def _top_k_by(arr, k_sel):
            k_sel = max(0, min(k_sel, len(arr)))
            if k_sel == 0:
                return np.array([], dtype=int)
            idx = np.argpartition(arr, len(arr) - k_sel)[-k_sel:]
            return idx[np.argsort(-arr[idx])]

        def _build(indices, label):
            return [
                {
                    "idx": int(i), "y_true": float(y_true[i]),
                    "yhat": float(yhat[i]), "cos": float(cos[i]),
                    "residual": float(residual[i]), "side": label,
                }
                for i in indices
            ]

        out = []
        if side in ("both", "over"):
            over_idx = _top_k_by(residual, k)
            out.extend(_build(over_idx, "over"))
        if side in ("both", "under"):
            under_idx = _top_k_by(-residual, k)
            out.extend(_build(under_idx, "under"))
        return out

    def snippets(self, pre_docs, *, top_per_side: int = 200, **kwargs) -> dict:
        """Extract text snippets aligned with beta.

        Parameters
        ----------
        pre_docs : list[PreprocessedDoc]
            Preprocessed documents (from Corpus.pre_docs).
        top_per_side : int
            Number of top snippets per side.
        **kwargs
            Forwarded to snippets_along_beta.
        """
        from ssdlite.utils.snippets import snippets_along_beta
        return snippets_along_beta(
            pre_docs=pre_docs, ssd=self,
            top_per_side=top_per_side, **kwargs,
        )


class PLSResult(_SSDResultBase):
    """Result from SSD.fit_pls() -- PLS1 NIPALS fit.

    Attributes
    ----------
    n_components : int
        Number of PLS latent components used.
    r2 : float
        Coefficient of determination on the training data.
    r2_adj : float
        Adjusted R-squared (corrected for number of components).
    pvalue : float
        P-value from the chosen significance test (NaN if skipped).
    p_method : str or None
        Which significance test produced ``pvalue``
        (``"perm"``, ``"split"``, ``"split_cal"``, or ``None``).
    beta : np.ndarray
        Raw regression weight vector in embedding space.
    beta_unit : np.ndarray
        Unit-length direction of ``beta``.
    cv_result : object or None
        Full cross-validation result object.
    cv_scores : dict or None
        Per-component CV R-squared scores (``{n_comp: r2_cv}``).
    perm_null : np.ndarray or None
        Null-distribution R-squared values from the permutation test.
    split_mean_r : float or None
        Mean Pearson r from split-half test (when ``p_method``
        is ``"split"`` or ``"split_cal"``).
    pca_k : int or None
        Number of PCA components used for dimensionality reduction
        before PLS (``None`` if PCA pre-reduction was not applied).

    Inherited from ``_SSDResultBase``
    ----------------------------------
    delta : float
        Predicted change in *y* per +0.10 cosine shift.
    iqr_effect : float
        Predicted change in *y* across the IQR of cosine alignment.
    y_corr_pred : float
        Absolute correlation between observed and predicted *y*.
    cos_align : np.ndarray
        Per-document cosine alignment to ``beta_unit``.
    """

    def __init__(
        self,
        *,
        n_components: int,
        cv_result,
        cv_scores: dict | None,
        perm_null: np.ndarray | None,
        pca_k: int | None = None,
        p_method: str | None = None,
        split_mean_r: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_components = n_components
        self.cv_result = cv_result
        self.cv_scores = cv_scores
        self.perm_null = perm_null
        self.pca_k = pca_k
        self.p_method = p_method
        self.split_mean_r = split_mean_r

    def summary(self) -> str:
        title = "SSD Model Summary (PLS)"
        sep = "─" * len(title)
        lines = [title, sep]

        lines.append(f"Backend: PLS ({self.n_components} components)")
        lines.extend(self._base_summary_lines())

        if np.isfinite(self.pvalue):
            label = self.p_method or "unknown"
            if label == "perm" and self.perm_null is not None:
                label = f"perm, {len(self.perm_null)} iter"
            lines.append("")
            lines.append(f"p-value = {self.pvalue:.4g} ({label})")
            if self.split_mean_r is not None:
                lines.append(f"split mean r = {self.split_mean_r:.4f}")

        if self.cv_scores:
            best_cv = max(self.cv_scores.values())
            lines.append(f"Best CV R² = {best_cv:.4f}")

        return "\n".join(lines)

    def split_test(
        self,
        n_splits: int = 50,
        split_ratio: float = 0.5,
        seed: int = 42,
        method: str = "split",
        n_perm: int = 200,
    ) -> dict:
        """Split-half significance test.

        Parameters
        ----------
        n_splits : int
            Number of random train/test splits.
        split_ratio : float
            Fraction used for training.
        seed : int
            Random seed.
        method : ``"split"`` | ``"split_cal"``
            Aggregation strategy:

            - ``"split"`` — overlap-corrected t-test on Fisher-z
              transformed correlations. Same cost as a single split run,
              corrects for split overlap
              (Lenartowicz, 2026).
            - ``"split_cal"`` — Permutation-calibrated: builds exact null
              distribution by re-running the full split procedure on
              permuted y. Exact FPR control but ~n_perm× slower
              (Lenartowicz, 2026).
        n_perm : int
            Permutations for ``"split_cal"`` method (ignored otherwise).

        Returns
        -------
        dict
            Keys: ``"pvalue"`` and ``"mean_r"``.
        """
        if method == "split":
            from ssdlite.backends.pls import pls1_split_test

            p_split, mean_r = pls1_split_test(
                self.x, self.y_kept, self.n_components,
                n_splits=n_splits, split_ratio=split_ratio,
                seed=seed, pca_k=self.pca_k,
            )
            return {"pvalue": p_split, "mean_r": mean_r}

        if method == "split_cal":
            from ssdlite.backends.pls import pls1_split_test_calibrated

            p_cal, mean_r = pls1_split_test_calibrated(
                self.x, self.y_kept, self.n_components,
                n_splits=n_splits, split_ratio=split_ratio,
                n_perm=n_perm, seed=seed, pca_k=self.pca_k,
            )
            return {"pvalue": p_cal, "mean_r": mean_r}

        raise ValueError(
            f"Unknown method {method!r}. Choose 'split' or 'split_cal'."
        )

    def __repr__(self) -> str:
        return (
            f"PLSResult(n_components={self.n_components}, r2={self.r2:.4f}, "
            f"pvalue={self.pvalue:.4g}, n_kept={self.n_kept})"
        )


class PCAOLSResult(_SSDResultBase):
    """Result from SSD.fit_ols() -- PCA + OLS fit.

    Attributes
    ----------
    n_components : int
        Number of PCA components retained before OLS regression.
    r2 : float
        Coefficient of determination on the training data.
    r2_adj : float
        Adjusted R-squared (corrected for number of components).
    pvalue : float
        F-test p-value for the overall regression.
    beta : np.ndarray
        Raw regression weight vector in embedding space.
    beta_unit : np.ndarray
        Unit-length direction of ``beta``.
    sweep_result : object or None
        Result from the component-sweep selection procedure.

    Inherited from ``_SSDResultBase``
    ----------------------------------
    delta : float
        Predicted change in *y* per +0.10 cosine shift.
    iqr_effect : float
        Predicted change in *y* across the IQR of cosine alignment.
    y_corr_pred : float
        Absolute correlation between observed and predicted *y*.
    cos_align : np.ndarray
        Per-document cosine alignment to ``beta_unit``.
    """

    def __init__(self, *, n_components: int, sweep_result=None, **kwargs):
        super().__init__(**kwargs)
        self.n_components = n_components
        self.sweep_result = sweep_result

    def summary(self) -> str:
        title = "SSD Model Summary (PCA+OLS)"
        sep = "─" * len(title)
        lines = [title, sep]

        lines.append(f"Backend: PCA+OLS ({self.n_components} components)")
        lines.extend(self._base_summary_lines())

        p_str = f"{self.pvalue:.4g}" if np.isfinite(self.pvalue) else "n/a"
        lines.append("")
        lines.append(f"p-value = {p_str}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"PCAOLSResult(n_components={self.n_components}, r2={self.r2:.4f}, "
            f"pvalue={self.pvalue:.4g}, n_kept={self.n_kept})"
        )
