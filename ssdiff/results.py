"""Result objects returned by SSD.fit_pls(), SSD.fit_ols(), and SSD.fit_groups()."""

from __future__ import annotations

from collections import namedtuple
from typing import Literal

import numpy as np

from ssdiff.utils.math import unit_vector
from ssdiff.utils.neighbors import cluster_top_neighbors, filtered_neighbors

# ---------------------------------------------------------------------------
# Citation
# ---------------------------------------------------------------------------

CITATION = (
    "Please cite as: Plisiecki, H., Lenartowicz, P., Pokropek, A., "
    "Małyska, K., & Flakus, M. (2025). Supervised Semantic Differential: "
    "Extracting Dimensions of Meaning from Word Embeddings. "
    "PsyArXiv. https://doi.org/10.31234/osf.io/gvrsb_v1"
)

# ---------------------------------------------------------------------------
# Report formatting helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> str:
    return f"\n── {title} {'─' * max(1, 48 - len(title))}"


def _fmt_top_words(words: list[dict], n: int) -> str:
    """Format top_words() output as side-by-side pos/neg table."""
    pos = [w for w in words if w["side"] == "pos"][:n]
    neg = [w for w in words if w["side"] == "neg"][:n]
    rows = max(len(pos), len(neg))

    lines = [_section(f"Top Words (n={n})")]
    lines.append(f"  {'+ pole':<28s}{'− pole':<28s}")
    for i in range(rows):
        left = f"  {pos[i]['rank']:>2}. {pos[i]['word']:<16s} {pos[i]['cos']:+.3f}" if i < len(pos) else " " * 28
        right = f"  {neg[i]['rank']:>2}. {neg[i]['word']:<16s} {neg[i]['cos']:+.3f}" if i < len(neg) else ""
        lines.append(f"{left}{right}")
    return "\n".join(lines)


def _fmt_clusters(clusters: list[dict], side: str) -> str:
    """Format cluster_neighbors() output."""
    label = "+β" if side == "pos" else "−β"
    lines = [_section(f"Clusters ({label}, {len(clusters)} clusters)")]
    for c in clusters:
        words_str = ", ".join(
            f"{w['word']} ({w['cos_beta']:+.3f})" for w in c["words"][:8]
        )
        if len(c["words"]) > 8:
            words_str += f", ... (+{len(c['words']) - 8})"
        lines.append(
            f"  [{c['id']}] {c['size']} words, "
            f"coherence={c['coherence']:.2f}, cos(β)={c['centroid_cos_beta']:+.3f}"
        )
        lines.append(f"      {words_str}")
    return "\n".join(lines)


def _fmt_extreme_docs(docs: list[dict], k: int) -> str:
    """Format extreme_docs() output."""
    top = [d for d in docs if d["side"] == "top"]
    bot = [d for d in docs if d["side"] == "bottom"]
    lines = [_section(f"Extreme Documents (k={k})")]
    lines.append("  Highest predicted:")
    for d in top:
        lines.append(
            f"    #{d['idx']:<5d}  y={d['y_true']:+.3f}  "
            f"ŷ={d['yhat']:+.3f}  cos={d['cos']:+.3f}"
        )
    lines.append("  Lowest predicted:")
    for d in bot:
        lines.append(
            f"    #{d['idx']:<5d}  y={d['y_true']:+.3f}  "
            f"ŷ={d['yhat']:+.3f}  cos={d['cos']:+.3f}"
        )
    return "\n".join(lines)


def _fmt_misdiagnosed(docs: list[dict], k: int) -> str:
    """Format misdiagnosed() output."""
    over = [d for d in docs if d["side"] == "over"]
    under = [d for d in docs if d["side"] == "under"]
    lines = [_section(f"Misdiagnosed (k={k})")]
    if over:
        lines.append("  Over-predicted:")
        for d in over:
            lines.append(
                f"    #{d['idx']:<5d}  y={d['y_true']:+.3f}  "
                f"ŷ={d['yhat']:+.3f}  residual={d['residual']:+.3f}"
            )
    if under:
        lines.append("  Under-predicted:")
        for d in under:
            lines.append(
                f"    #{d['idx']:<5d}  y={d['y_true']:+.3f}  "
                f"ŷ={d['yhat']:+.3f}  residual={d['residual']:+.3f}"
            )
    return "\n".join(lines)


def _rolling_median(x: np.ndarray, window: int = 7) -> np.ndarray:
    """Rolling median with NaN-awareness. Used for sweep plot smoothing."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = np.full(n, np.nan)
    half = window // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        w = x[lo:hi]
        w = w[np.isfinite(w)]
        if len(w):
            out[i] = float(np.median(w))
    return out


# ---------------------------------------------------------------------------
# _Interpretable mixin — shared interpretation API
# ---------------------------------------------------------------------------

class _Interpretable:
    """Anything with embeddings + beta_unit + lang can interpret.

    Subclasses must set: embeddings, beta_unit, lexicon, window, sif_a, lang.
    """

    result_type: str

    def _require_embeddings(self, method_name: str) -> None:
        """Raise if embeddings are not attached (needed for fresh computation)."""
        if self.embeddings is None:
            raise RuntimeError(
                f"{method_name}() requires embeddings but none are attached. "
                f"Re-attach with: result.embeddings = Embeddings.load('path/to/model')"
            )

    def top_words(self, n: int = 20, *, recompute: bool = False) -> list[dict]:
        """Top neighbor words on both poles of beta.

        Returns
        -------
        list[dict]
            Each dict has keys: ``side``, ``rank``, ``word``, ``cos``.
        """
        if not recompute and self._cached_top_words is not None:
            if n <= self._cached_top_words_n:
                return [w for w in self._cached_top_words if w["rank"] <= n]
            # Requested more than cached — fall through to recompute
        self._require_embeddings("top_words")
        b = self.beta_unit
        out = []
        for side, vec in [("pos", b), ("neg", -b)]:
            pairs = filtered_neighbors(self.embeddings, vec, topn=n, lang=self.lang)
            for rank, (word, cos) in enumerate(pairs, 1):
                out.append({"side": side, "rank": rank, "word": word, "cos": float(cos)})
        self._cached_top_words = out
        self._cached_top_words_n = n
        return out

    def neighbors(self, side: Literal["pos", "neg"] = "pos", n: int = 20,
                  *, recompute: bool = False) -> list[tuple[str, float]]:
        """Top cosine neighbors to +beta (pos) or -beta (neg)."""
        words = self.top_words(n=n, recompute=recompute)
        return [(w["word"], w["cos"]) for w in words if w["side"] == side]

    def cluster_neighbors(
        self,
        side: Literal["pos", "neg"] = "pos",
        *,
        recompute: bool = False,
        topn: int = 100,
        k: int | None = None,
        k_min: int = 2,
        k_max: int = 10,
        random_state: int = 2137,
        min_cluster_size: int = 2,
    ) -> list[dict]:
        """Cluster top neighbors into interpretable themes."""
        attr = "_cached_clusters_pos" if side == "pos" else "_cached_clusters_neg"
        cached = getattr(self, attr, None)
        if not recompute and cached is not None:
            return cached
        self._require_embeddings("cluster_neighbors")
        b = self.beta_unit
        clusters = cluster_top_neighbors(
            self.embeddings, b,
            topn=topn, k=k, k_min=k_min, k_max=k_max,
            random_state=random_state, min_cluster_size=min_cluster_size,
            side=side, lang=self.lang,
        )
        setattr(self, attr, clusters)
        return clusters

    def snippets(self, pre_docs=None, *, recompute: bool = False,
                 top_per_side: int = 200, **kwargs) -> dict:
        """Extract text snippets aligned with beta."""
        if not recompute and self._cached_snippets is not None:
            if top_per_side <= self._cached_snippets_top:
                return self._cached_snippets
            # Requested more than cached — fall through to recompute
        if pre_docs is None:
            if self._cached_snippets is not None:
                return self._cached_snippets
            raise ValueError("No cached snippets and no pre_docs provided")
        self._require_embeddings("snippets")
        from ssdiff.utils.snippets import snippets_along_beta
        result = snippets_along_beta(
            pre_docs=pre_docs, ssd=self,
            top_per_side=top_per_side, **kwargs,
        )
        self._cached_snippets = result
        self._cached_snippets_top = top_per_side
        return result

    def cluster_snippets(
        self,
        pre_docs=None,
        pos_clusters=None,
        neg_clusters=None,
        *,
        recompute: bool = False,
        top_per_cluster: int = 100,
        **kwargs,
    ) -> dict:
        """Extract text snippets scored against cluster centroids.

        Parameters
        ----------
        pre_docs : list[PreprocessedDoc] | None
            Preprocessed documents. If None, returns cached results.
        pos_clusters, neg_clusters : list[dict] | None
            Cluster dicts from ``cluster_neighbors()``.  If None, falls back
            to cached clusters from prior ``cluster_neighbors()`` calls.
        recompute : bool
            If True, force recomputation even when cached.
        top_per_cluster : int
            Max snippets per cluster (default 100).

        Returns
        -------
        dict with 'pos' and 'neg' lists of snippet dicts, each containing
        a ``centroid_label`` field (e.g. ``"pos_cluster_1"``).
        """
        explicit_clusters = pos_clusters is not None or neg_clusters is not None
        if not recompute and not explicit_clusters and self._cached_cluster_snippets is not None:
            if top_per_cluster <= self._cached_cluster_snippets_top:
                return self._cached_cluster_snippets
        if pre_docs is None:
            if not explicit_clusters and self._cached_cluster_snippets is not None:
                return self._cached_cluster_snippets
            raise ValueError("No cached cluster_snippets and no pre_docs provided")
        self._require_embeddings("cluster_snippets")
        from ssdiff.utils.snippets import cluster_snippets_by_centroids
        pos = pos_clusters if pos_clusters is not None else self._cached_clusters_pos
        neg = neg_clusters if neg_clusters is not None else self._cached_clusters_neg
        result = cluster_snippets_by_centroids(
            pre_docs=pre_docs, ssd=self,
            pos_clusters=pos, neg_clusters=neg,
            top_per_cluster=top_per_cluster, **kwargs,
        )
        self._cached_cluster_snippets = result
        self._cached_cluster_snippets_top = top_per_cluster
        return result


# ---------------------------------------------------------------------------
# _SSDResultBase — regression results (PLS / PCA+OLS)
# ---------------------------------------------------------------------------

class _SSDResultBase(_Interpretable):
    """Shared base for regression-type SSD results.

    Duck-types with what snippets_along_beta expects:
    .embeddings, .beta_unit, .beta, .lexicon, .window, .sif_a
    """

    def __init__(
        self,
        *,
        embeddings,
        lexicon: set,
        window: int,
        sif_a: float,
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
        pvalue: float,
        r2_adj: float | None = None,
    ):
        self.embeddings = embeddings
        self.lexicon = lexicon
        self.window = window
        self.sif_a = sif_a
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

        # Cache slots (set by interpretation methods, included in pickle)
        self._cached_top_words: list[dict] | None = None
        self._cached_top_words_n: int = 0
        self._cached_clusters_pos: list[dict] | None = None
        self._cached_clusters_neg: list[dict] | None = None
        self._cached_snippets: dict | None = None
        self._cached_snippets_top: int = 0
        self._cached_cluster_snippets: dict | None = None
        self._cached_cluster_snippets_top: int = 0

        # Effect sizes
        self._compute_effect_sizes()

    def _compute_effect_sizes(self):
        """Compute effect-size calibration attributes from the fitted model."""
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
        if self.r2_adj is not None:
            lines.append(f"R² = {self.r2:.4f}   R²_adj = {self.r2_adj:.4f}")
        else:
            lines.append(f"R² = {self.r2:.4f}")
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

    def report(
        self,
        *,
        top_words: int | None = 5,
        clusters: int | None = None,
        extreme_docs: int | None = None,
        misdiagnosed: int | None = None,
    ) -> str:
        """Print a comprehensive report to console and return it as a string.

        Parameters
        ----------
        top_words : int or None
            Number of top words per side. None to skip.
        clusters : int or None
            Number of top neighbors to cluster (topn). None to skip.
        extreme_docs : int or None
            Number of extreme docs per side (top/bottom). None to skip.
        misdiagnosed : int or None
            Number of misdiagnosed docs per side (over/under). None to skip.
        """
        parts = [self.summary()]

        if top_words is not None:
            parts.append(_fmt_top_words(self.top_words(n=top_words), top_words))

        if clusters is not None:
            for side in ("pos", "neg"):
                cl = self.cluster_neighbors(side=side, topn=clusters)
                if cl:
                    parts.append(_fmt_clusters(cl, side))

        if extreme_docs is not None:
            docs = self.extreme_docs(k=extreme_docs)
            if docs:
                parts.append(_fmt_extreme_docs(docs, extreme_docs))

        if misdiagnosed is not None:
            docs = self.misdiagnosed(k=misdiagnosed)
            if docs:
                parts.append(_fmt_misdiagnosed(docs, misdiagnosed))

        parts.append("─" * 50)
        parts.append(CITATION)

        text = "\n".join(parts)
        print(text)
        return text

    def doc_scores(self) -> dict:
        """Per-document alignment scores and predictions."""
        score_std = (self.x @ self.beta).astype(np.float64)
        yhat_raw = self.y_mean + self.y_std * score_std

        return {
            "idx": np.where(self.keep_mask)[0],
            "cos_align": self.cos_align.copy(),
            "score_std": score_std,
            "yhat_raw": yhat_raw,
        }

    def extreme_docs(
        self, k: int = 50, by: Literal["predicted", "observed"] = "predicted",
    ) -> list[dict]:
        """Select top-k and bottom-k documents by predicted or observed outcome."""
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
        """Extract text snippets from extreme documents."""
        from ssdiff.utils.snippets import snippets_along_beta

        extremes = self.extreme_docs(k=k, by=by)
        if not extremes:
            return {"pos": [], "neg": []}

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
        self, k: int = 20, side: Literal["both", "over", "under"] = "both",
    ) -> list[dict]:
        """Documents where model predictions diverge most from observed."""
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

    def save(self, path) -> None:
        """Save result to pickle, stripping large recomputable objects.

        Keeps: beta, statistics, cached interpretation data, x.
        Strips: embeddings (reload from file), perm_null, sweep_result, cv_result.

        Cached interpretation (top_words, clusters, snippets) is included —
        the loaded result can serve these from cache without embeddings.
        To recompute, re-attach embeddings and call with recompute=True.
        """
        import copy
        import pickle
        stripped = copy.copy(self)
        stripped.embeddings = None
        for attr in ("perm_null", "sweep_result", "cv_result"):
            if hasattr(stripped, attr):
                setattr(stripped, attr, None)
        with open(path, "wb") as f:
            pickle.dump(stripped, f, protocol=pickle.HIGHEST_PROTOCOL)


# ---------------------------------------------------------------------------
# PLSResult
# ---------------------------------------------------------------------------

class PLSResult(_SSDResultBase):
    """Result from SSD.fit_pls() -- PLS1 NIPALS fit."""

    result_type = "pls"

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
        random_state: int | None = None,
        n_perm: int | None = None,
        n_splits: int | None = None,
        split_ratio: float | None = None,
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
        self.random_state = random_state
        self.n_perm = n_perm
        self.n_splits = n_splits
        self.split_ratio = split_ratio

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
        method: Literal["split", "split_cal"] = "split",
        n_perm: int = 200,
    ) -> dict:
        """Split-half significance test."""
        if method == "split":
            from ssdiff.backends.pls import pls1_split_test

            p_split, mean_r = pls1_split_test(
                self.x, self.y_kept, self.n_components,
                n_splits=n_splits, split_ratio=split_ratio,
                seed=seed, pca_k=self.pca_k,
            )
            return {"pvalue": p_split, "mean_r": mean_r}

        if method == "split_cal":
            from ssdiff.backends.pls import pls1_split_test_calibrated

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


# ---------------------------------------------------------------------------
# PCAOLSResult
# ---------------------------------------------------------------------------

class PCAOLSResult(_SSDResultBase):
    """Result from SSD.fit_ols() -- PCA + OLS fit."""

    result_type = "pca_ols"

    def __init__(
        self,
        *,
        n_components: int,
        sweep_result=None,
        k_min: int | None = None,
        k_max: int | None = None,
        k_step: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_components = n_components
        self.sweep_result = sweep_result
        self.k_min = k_min
        self.k_max = k_max
        self.k_step = k_step

    def plot_sweep(self, path: str | None = None, *, dpi: int = 300) -> bytes:
        """Render the PCA-K sweep plot as a dual-axis chart."""
        if self.sweep_result is None:
            raise RuntimeError(
                "No sweep data — fit_ols() was called with an explicit "
                "n_components. Re-run with n_components=None to enable "
                "the PCA-K sweep."
            )

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plot_sweep(). "
                "Install it with: pip install ssdiff[plot]"
            ) from None

        import io

        rows = self.sweep_result.df_joined
        x = [r["PCA_K"] for r in rows]
        y_left = np.array([r.get("interp_resid_z", np.nan) for r in rows])
        y_right = np.array([r.get("beta_delta_1_minus_cos", np.nan) for r in rows])

        y_right_smooth = _rolling_median(y_right, window=7)

        fig, ax1 = plt.subplots()

        ax1.plot(x, y_left, marker="o", color="tab:blue",
                 label="detrended interpretability (z)")
        ax1.axhline(0.0, linewidth=1, color="0.6")
        ax1.set_xlabel("PCA_K")
        ax1.set_ylabel("Detrended interpretability (z)", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax1.twinx()
        ax2.plot(x, y_right_smooth, linewidth=2, color="tab:orange",
                 label="beta change (smoothed 1-cos)")
        ax2.set_ylabel("Beta change (smoothed 1 - cosine)", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        ax1.axvline(self.sweep_result.best_k, color="red", linewidth=2,
                     label=f"best K = {self.sweep_result.best_k}")

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        png = buf.getvalue()

        if path is not None:
            with open(path, "wb") as f:
                f.write(png)
        else:
            plt.show()

        plt.close(fig)
        return png

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


# ---------------------------------------------------------------------------
# GroupResult
# ---------------------------------------------------------------------------

_ContrastProxy = namedtuple(
    "_ContrastProxy", "embeddings beta_unit beta lexicon window sif_a lang x",
)


class GroupResult(_Interpretable):
    """Result from SSD.fit_groups() — group comparison via permutation tests.

    Stores pairwise data in a dict keyed by (g1, g2). Each value holds:
    beta_unit, T, p_raw, p_corrected, cohens_d, n_g1, n_g2, contrast_norm.
    """

    result_type = "group"

    def __init__(
        self,
        *,
        embeddings,
        lexicon: set,
        window: int,
        sif_a: float,
        lang: str,
        x: np.ndarray,
        groups_kept: np.ndarray,
        keep_mask: np.ndarray,
        n_raw: int,
        n_kept: int,
        n_dropped: int,
        n_group_dropped: int,
        omnibus_T: float,
        omnibus_p: float,
        pairwise: dict,
        group_labels: list,
        G: int,
        n_perm: int,
        correction: str,
        random_state: int | None = None,
        _original_omnibus: dict | None = None,
        _original_group_labels: list | None = None,
    ):
        self.embeddings = embeddings
        self.lexicon = lexicon
        self.window = window
        self.sif_a = sif_a
        self.lang = lang

        self.x = x
        self.groups_kept = groups_kept
        self.keep_mask = keep_mask
        self.n_raw = n_raw
        self.n_kept = n_kept
        self.n_dropped = n_dropped
        self.n_group_dropped = n_group_dropped

        self.omnibus_T = omnibus_T
        self.omnibus_p = omnibus_p
        self.pairwise = pairwise
        self.group_labels = group_labels
        self.G = G
        self.n_perm = n_perm
        self.correction = correction
        self.random_state = random_state

        # For filtered results: keep original omnibus for display
        self._original_omnibus = _original_omnibus
        self._original_group_labels = _original_group_labels

        # Cache slots (set by interpretation methods, included in pickle)
        self._cached_top_words: list[dict] | None = None
        self._cached_top_words_n: int = 0
        self._cached_clusters_pos: list[dict] | None = None
        self._cached_clusters_neg: list[dict] | None = None
        self._cached_snippets: dict | None = None
        self._cached_snippets_top: int = 0
        self._cached_cluster_snippets: dict | None = None
        self._cached_cluster_snippets_top: int = 0

        # beta_unit not meaningful for multi-contrast — set to None
        # Interpretation methods override to loop over pairs
        self.beta_unit = None

    # ── Interpretation overrides ──────────────────────────────

    def _make_proxy(self, pair_key) -> _ContrastProxy:
        """Build a lightweight proxy for a single contrast."""
        pair = self.pairwise[pair_key]
        return _ContrastProxy(
            embeddings=self.embeddings,
            beta_unit=pair["beta_unit"],
            beta=pair["beta_unit"],
            lexicon=self.lexicon,
            window=self.window,
            sif_a=self.sif_a,
            lang=self.lang,
            x=self.x,
        )

    def _contrast_label(self, pair_key) -> str:
        return f"{pair_key[0]} vs {pair_key[1]}"

    def top_words(self, n: int = 20, *, recompute: bool = False) -> list[dict]:
        """Top neighbor words for all contrasts.

        Returns list[dict] with added ``contrast`` key.
        """
        if not recompute and self._cached_top_words is not None:
            if n <= self._cached_top_words_n:
                return [w for w in self._cached_top_words if w["rank"] <= n]
        self._require_embeddings("top_words")
        out = []
        for pair_key in self.pairwise:
            proxy = self._make_proxy(pair_key)
            label = self._contrast_label(pair_key)
            b = proxy.beta_unit
            for side, vec in [("pos", b), ("neg", -b)]:
                pairs = filtered_neighbors(proxy.embeddings, vec, topn=n, lang=proxy.lang)
                for rank, (word, cos) in enumerate(pairs, 1):
                    out.append({
                        "contrast": label,
                        "side": side,
                        "rank": rank,
                        "word": word,
                        "cos": float(cos),
                    })
        self._cached_top_words = out
        self._cached_top_words_n = n
        return out

    def neighbors(self, side: Literal["pos", "neg"] = "pos", n: int = 20,
                  *, recompute: bool = False) -> list[tuple[str, str, float]]:
        """Neighbors for all contrasts.

        Returns list of (contrast_label, word, cosine) tuples for all pairs.
        """
        words = self.top_words(n=n, recompute=recompute)
        return [(w["contrast"], w["word"], w["cos"]) for w in words if w["side"] == side]

    def cluster_neighbors(
        self,
        side: Literal["pos", "neg"] = "pos",
        *,
        recompute: bool = False,
        **kwargs,
    ) -> list[dict]:
        """Cluster neighbors for all contrasts.

        Returns list[dict] with added ``contrast`` key.
        """
        attr = "_cached_clusters_pos" if side == "pos" else "_cached_clusters_neg"
        cached = getattr(self, attr, None)
        if not recompute and cached is not None:
            return cached
        self._require_embeddings("cluster_neighbors")
        out = []
        for pair_key in self.pairwise:
            proxy = self._make_proxy(pair_key)
            label = self._contrast_label(pair_key)
            clusters = cluster_top_neighbors(
                proxy.embeddings, proxy.beta_unit,
                side=side, lang=proxy.lang, **kwargs,
            )
            for c in clusters:
                c["contrast"] = label
                out.append(c)
        setattr(self, attr, out)
        return out

    def snippets(self, pre_docs=None, *, recompute: bool = False,
                 top_per_side: int = 200, **kwargs) -> dict:
        """Snippets for all contrasts.

        Returns dict with ``contrast`` key added to each snippet dict.
        """
        if not recompute and self._cached_snippets is not None:
            if top_per_side <= self._cached_snippets_top:
                return self._cached_snippets
        if pre_docs is None:
            if self._cached_snippets is not None:
                return self._cached_snippets
            raise ValueError("No cached snippets and no pre_docs provided")
        self._require_embeddings("snippets")
        from ssdiff.utils.snippets import snippets_along_beta

        all_pos, all_neg = [], []
        for pair_key in self.pairwise:
            proxy = self._make_proxy(pair_key)
            label = self._contrast_label(pair_key)
            result = snippets_along_beta(
                pre_docs=pre_docs, ssd=proxy,
                top_per_side=top_per_side, **kwargs,
            )
            for s in result.get("pos", []):
                s["contrast"] = label
                all_pos.append(s)
            for s in result.get("neg", []):
                s["contrast"] = label
                all_neg.append(s)
        out = {"pos": all_pos, "neg": all_neg}
        self._cached_snippets = out
        self._cached_snippets_top = top_per_side
        return out

    def cluster_snippets(
        self,
        pre_docs=None,
        pos_clusters=None,
        neg_clusters=None,
        *,
        recompute: bool = False,
        top_per_cluster: int = 100,
        **kwargs,
    ) -> dict:
        """Cluster snippets for all contrasts.

        Returns dict with ``contrast`` key added to each snippet dict.
        """
        explicit_clusters = pos_clusters is not None or neg_clusters is not None
        if not recompute and not explicit_clusters and self._cached_cluster_snippets is not None:
            if top_per_cluster <= self._cached_cluster_snippets_top:
                return self._cached_cluster_snippets
        if pre_docs is None:
            if not explicit_clusters and self._cached_cluster_snippets is not None:
                return self._cached_cluster_snippets
            raise ValueError("No cached cluster_snippets and no pre_docs provided")
        self._require_embeddings("cluster_snippets")
        from ssdiff.utils.snippets import cluster_snippets_by_centroids

        all_pos, all_neg = [], []
        for pair_key in self.pairwise:
            proxy = self._make_proxy(pair_key)
            label = self._contrast_label(pair_key)
            # Filter cached clusters belonging to this contrast
            pos_c = pos_clusters or [
                c for c in (self._cached_clusters_pos or [])
                if c.get("contrast") == label
            ]
            neg_c = neg_clusters or [
                c for c in (self._cached_clusters_neg or [])
                if c.get("contrast") == label
            ]
            if not pos_c and not neg_c:
                continue
            result = cluster_snippets_by_centroids(
                pre_docs=pre_docs, ssd=proxy,
                pos_clusters=pos_c, neg_clusters=neg_c,
                top_per_cluster=top_per_cluster, **kwargs,
            )
            for s in result.get("pos", []):
                s["contrast"] = label
                all_pos.append(s)
            for s in result.get("neg", []):
                s["contrast"] = label
                all_neg.append(s)
        out = {"pos": all_pos, "neg": all_neg}
        self._cached_cluster_snippets = out
        self._cached_cluster_snippets_top = top_per_cluster
        return out

    def save(self, path) -> None:
        """Save result to pickle, stripping embeddings.

        Cached interpretation (top_words, clusters, snippets) is included.
        """
        import copy
        import pickle
        stripped = copy.copy(self)
        stripped.embeddings = None
        with open(path, "wb") as f:
            pickle.dump(stripped, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ── Results access ────────────────────────────────────────

    def results_table(self) -> list[dict]:
        """Pairwise results as list of dicts."""
        rows = []
        for (g1, g2), r in self.pairwise.items():
            rows.append({
                "group_A": g1, "group_B": g2,
                "n_A": r["n_g1"], "n_B": r["n_g2"],
                "cosine_distance": r["T"],
                "p_raw": r["p_raw"], "p_corrected": r["p_corrected"],
                "cohens_d": r["cohens_d"], "contrast_norm": r["contrast_norm"],
            })
        return rows

    # ── Filtering ─────────────────────────────────────────────

    def filter_groups(self, *labels) -> GroupResult:
        """Return a new GroupResult with a subset of contrasts.

        Accepts any number of group labels. Returns contrasts involving
        all pairs from the given labels.
        """
        for lbl in labels:
            if lbl not in self.group_labels:
                raise ValueError(
                    f"Group '{lbl}' not found. Available: {self.group_labels}"
                )

        label_set = set(labels)
        # Select pairwise results where both groups are in label_set
        filtered_pw = {}
        for (g1, g2), r in self.pairwise.items():
            if g1 in label_set and g2 in label_set:
                filtered_pw[(g1, g2)] = r

        # If only 1 label, select all contrasts involving that label
        if len(labels) == 1:
            lbl = labels[0]
            for (g1, g2), r in self.pairwise.items():
                if g1 == lbl or g2 == lbl:
                    filtered_pw[(g1, g2)] = r

        if not filtered_pw:
            raise ValueError(
                f"No pairwise contrasts found for labels {labels}"
            )

        # Subset x and groups to only docs in selected groups
        all_labels = set()
        for g1, g2 in filtered_pw:
            all_labels.add(g1)
            all_labels.add(g2)
        mask = np.isin(self.groups_kept, list(all_labels))
        x_filtered = self.x[mask]
        groups_filtered = self.groups_kept[mask]

        return GroupResult(
            embeddings=self.embeddings,
            lexicon=self.lexicon,
            window=self.window,
            sif_a=self.sif_a,
            lang=self.lang,
            x=x_filtered,
            groups_kept=groups_filtered,
            keep_mask=self.keep_mask,
            n_raw=self.n_raw,
            n_kept=int(mask.sum()),
            n_dropped=self.n_dropped,
            n_group_dropped=self.n_group_dropped,
            omnibus_T=self.omnibus_T,
            omnibus_p=self.omnibus_p,
            pairwise=filtered_pw,
            group_labels=sorted(all_labels),
            G=len(all_labels),
            n_perm=self.n_perm,
            correction=self.correction,
            _original_omnibus={
                "T": self._original_omnibus["T"] if self._original_omnibus else self.omnibus_T,
                "p": self._original_omnibus["p"] if self._original_omnibus else self.omnibus_p,
            },
            _original_group_labels=(
                self._original_group_labels or self.group_labels
            ),
        )

    # ── Display ───────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable group analysis summary."""
        labels_str = ", ".join(str(g) for g in self.group_labels)
        is_filtered = self._original_group_labels is not None

        if is_filtered:
            orig_labels = ", ".join(str(g) for g in self._original_group_labels)
            orig_G = len(self._original_group_labels)
            title = f"Group Analysis Summary (filtered: {labels_str})"
            sep = "─" * len(title)
            lines = [title, sep]
            lines.append(
                f"Groups: {self.G} ({labels_str})  — filtered from "
                f"{orig_G} ({orig_labels})"
            )
            lines.append(
                f"Docs: {self.n_kept} in filtered contrasts / "
                f"{self.n_raw} total"
            )
            orig_T = self._original_omnibus["T"]
            orig_p = self._original_omnibus["p"]
            lines.append(
                f"Omnibus (all {orig_G} groups): "
                f"T = {orig_T:.4f}   p = {orig_p:.4f}"
            )
        else:
            title = "Group Analysis Summary"
            sep = "─" * len(title)
            lines = [title, sep]
            n_kept_str = f"Docs: {self.n_kept} kept / {self.n_raw} total"
            if self.n_group_dropped > 0:
                n_kept_str += f" ({self.n_group_dropped} dropped by small-group filter)"
            lines.append(
                f"Groups: {self.G} ({labels_str})   {n_kept_str}"
            )
            lines.append(
                f"Permutations: {self.n_perm}     Correction: {self.correction.title()}"
            )
            lines.append(
                f"Omnibus: T = {self.omnibus_T:.4f}   p = {self.omnibus_p:.4f}"
            )

        lines.append("")
        lines.append("Pairwise:")
        for (g1, g2), r in self.pairwise.items():
            lines.append(
                f"  {g1} vs {g2}: "
                f"cos_dist={r['T']:.4f}  "
                f"p={r['p_raw']:.4f} (corrected={r['p_corrected']:.4f})  "
                f"d={r['cohens_d']:+.2f}  "
                f"n={r['n_g1']}/{r['n_g2']}"
            )
        return "\n".join(lines)

    def report(
        self,
        *,
        top_words: int | None = 5,
        clusters: int | None = None,
        extreme_docs: int | None = None,
        misdiagnosed: int | None = None,
    ) -> str:
        """Print a comprehensive report to console and return it as a string.

        Parameters
        ----------
        top_words : int or None
            Number of top words per side per contrast. None to skip.
        clusters : int or None
            Number of top neighbors to cluster (topn) per contrast. None to skip.
        extreme_docs : int or None
            Ignored (not applicable to group results).
        misdiagnosed : int or None
            Ignored (not applicable to group results).
        """
        parts = [self.summary()]

        for pair_key in self.pairwise:
            proxy = self._make_proxy(pair_key)
            label = self._contrast_label(pair_key)

            if top_words is not None:
                words = []
                b = proxy.beta_unit
                for side, vec in [("pos", b), ("neg", -b)]:
                    pairs = filtered_neighbors(proxy.embeddings, vec, topn=top_words, lang=proxy.lang)
                    for rank, (word, cos) in enumerate(pairs, 1):
                        words.append({"side": side, "rank": rank, "word": word, "cos": float(cos)})
                parts.append(_fmt_top_words(words, top_words).replace(
                    f"Top Words (n={top_words})",
                    f"Top Words: {label} (n={top_words})",
                ))

            if clusters is not None:
                for side in ("pos", "neg"):
                    cl = cluster_top_neighbors(
                        proxy.embeddings, proxy.beta_unit,
                        topn=clusters, side=side, lang=proxy.lang,
                    )
                    if cl:
                        header = _fmt_clusters(cl, side)
                        header = header.replace(
                            f"Clusters ({'+'  if side == 'pos' else '−'}β",
                            f"Clusters: {label} ({'+'  if side == 'pos' else '−'}β",
                        )
                        parts.append(header)

        parts.append("─" * 50)
        parts.append(CITATION)

        text = "\n".join(parts)
        print(text)
        return text

    def __repr__(self) -> str:
        return (
            f"GroupResult({self.G} groups, n_kept={self.n_kept}, "
            f"omnibus_p={self.omnibus_p:.4f})"
        )
