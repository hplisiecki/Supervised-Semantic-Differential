"""Result objects returned by SSD.fit_pls(), SSD.fit_ols(), and SSD.fit_groups()."""

from __future__ import annotations

from collections import namedtuple
from typing import Literal

import numpy as np

from ssdiff.utils.math import unit_vector
from ssdiff.utils.neighbors import cluster_top_neighbors, filtered_neighbors

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
    """Anything with kv + beta_unit + lang can interpret.

    Subclasses must set: kv, beta_unit, lexicon, window, sif_a, lang.
    """

    result_type: str

    def top_words(self, n: int = 20) -> list[dict]:
        """Top neighbor words on both poles of beta.

        Returns
        -------
        list[dict]
            Each dict has keys: ``side``, ``rank``, ``word``, ``cos``.
        """
        b = self.beta_unit
        out = []
        for side, vec in [("pos", b), ("neg", -b)]:
            pairs = filtered_neighbors(self.kv, vec, topn=n, lang=self.lang)
            for rank, (word, cos) in enumerate(pairs, 1):
                out.append({"side": side, "rank": rank, "word": word, "cos": float(cos)})
        return out

    def neighbors(self, side: Literal["pos", "neg"] = "pos", n: int = 20) -> list[tuple[str, float]]:
        """Top cosine neighbors to +beta (pos) or -beta (neg)."""
        b = self.beta_unit
        vec = b if side == "pos" else -b
        return filtered_neighbors(self.kv, vec, topn=n, lang=self.lang)

    def cluster_neighbors(
        self,
        side: Literal["pos", "neg"] = "pos",
        *,
        topn: int = 100,
        k: int | None = None,
        k_min: int = 2,
        k_max: int = 10,
        random_state: int = 2137,
        min_cluster_size: int = 2,
    ) -> list[dict]:
        """Cluster top neighbors into interpretable themes."""
        b = self.beta_unit
        clusters = cluster_top_neighbors(
            self.kv, b,
            topn=topn, k=k, k_min=k_min, k_max=k_max,
            random_state=random_state, min_cluster_size=min_cluster_size,
            side=side, lang=self.lang,
        )
        if side == "pos":
            self.pos_clusters_raw = clusters
        else:
            self.neg_clusters_raw = clusters
        return clusters

    def snippets(self, pre_docs, *, top_per_side: int = 200, **kwargs) -> dict:
        """Extract text snippets aligned with beta."""
        from ssdiff.utils.snippets import snippets_along_beta
        return snippets_along_beta(
            pre_docs=pre_docs, ssd=self,
            top_per_side=top_per_side, **kwargs,
        )

    def cluster_snippets(
        self,
        pre_docs,
        pos_clusters=None,
        neg_clusters=None,
        *,
        top_per_cluster: int = 100,
        **kwargs,
    ) -> dict:
        """Extract text snippets scored against cluster centroids.

        Parameters
        ----------
        pre_docs : list[PreprocessedDoc]
            Preprocessed documents.
        pos_clusters, neg_clusters : list[dict] | None
            Cluster dicts from ``cluster_neighbors()``.  If None, falls back
            to ``self.pos_clusters_raw`` / ``self.neg_clusters_raw`` (set
            automatically by ``cluster_neighbors()``).
        top_per_cluster : int
            Max snippets per cluster (default 100).

        Returns
        -------
        dict with 'pos' and 'neg' lists of snippet dicts, each containing
        a ``centroid_label`` field (e.g. ``"pos_cluster_1"``).
        """
        from ssdiff.utils.snippets import cluster_snippets_by_centroids
        pos = pos_clusters if pos_clusters is not None else getattr(self, "pos_clusters_raw", None)
        neg = neg_clusters if neg_clusters is not None else getattr(self, "neg_clusters_raw", None)
        return cluster_snippets_by_centroids(
            pre_docs=pre_docs, ssd=self,
            pos_clusters=pos, neg_clusters=neg,
            top_per_cluster=top_per_cluster, **kwargs,
        )


# ---------------------------------------------------------------------------
# _SSDResultBase — regression results (PLS / PCA+OLS)
# ---------------------------------------------------------------------------

class _SSDResultBase(_Interpretable):
    """Shared base for regression-type SSD results.

    Duck-types with what snippets_along_beta expects:
    .kv, .beta_unit, .beta, .lexicon, .window, .sif_a
    """

    def __init__(
        self,
        *,
        kv,
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
        self.kv = kv
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

        # Cluster placeholders
        self.pos_clusters_raw = None
        self.neg_clusters_raw = None

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

        text = "\n".join(parts)
        print(text)
        return text

    def doc_scores(self) -> dict:
        """Per-document alignment scores and predictions."""
        score_std = (self.x @ self.beta).astype(np.float64)
        yhat_raw = self.y_mean + self.y_std * score_std

        return {
            "keep_mask": self.keep_mask.copy(),
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
    "_ContrastProxy", "kv beta_unit beta lexicon window sif_a lang x",
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
        kv,
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
        self.kv = kv
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

        # No cluster caching for GroupResult
        self.pos_clusters_raw = None
        self.neg_clusters_raw = None

        # beta_unit not meaningful for multi-contrast — set to None
        # Interpretation methods override to loop over pairs
        self.beta_unit = None

    # ── Interpretation overrides ──────────────────────────────

    def _make_proxy(self, pair_key) -> _ContrastProxy:
        """Build a lightweight proxy for a single contrast."""
        pair = self.pairwise[pair_key]
        return _ContrastProxy(
            kv=self.kv,
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

    def top_words(self, n: int = 20) -> list[dict]:
        """Top neighbor words for all contrasts.

        Returns list[dict] with added ``contrast`` key.
        """
        out = []
        for pair_key in self.pairwise:
            proxy = self._make_proxy(pair_key)
            label = self._contrast_label(pair_key)
            b = proxy.beta_unit
            for side, vec in [("pos", b), ("neg", -b)]:
                pairs = filtered_neighbors(proxy.kv, vec, topn=n, lang=proxy.lang)
                for rank, (word, cos) in enumerate(pairs, 1):
                    out.append({
                        "contrast": label,
                        "side": side,
                        "rank": rank,
                        "word": word,
                        "cos": float(cos),
                    })
        return out

    def neighbors(self, side: Literal["pos", "neg"] = "pos", n: int = 20) -> list[tuple[str, str, float]]:
        """Neighbors for all contrasts.

        Returns list of (contrast_label, word, cosine) tuples for all pairs.
        """
        out: list[tuple[str, str, float]] = []
        for pair_key in self.pairwise:
            proxy = self._make_proxy(pair_key)
            b = proxy.beta_unit
            vec = b if side == "pos" else -b
            nbrs = filtered_neighbors(proxy.kv, vec, topn=n, lang=proxy.lang)
            label = self._contrast_label(pair_key)
            for word, cos in nbrs:
                out.append((label, word, cos))
        return out

    def cluster_neighbors(
        self,
        side: Literal["pos", "neg"] = "pos",
        **kwargs,
    ) -> list[dict]:
        """Cluster neighbors for all contrasts.

        Returns list[dict] with added ``contrast`` key. No caching.
        """
        out = []
        for pair_key in self.pairwise:
            proxy = self._make_proxy(pair_key)
            label = self._contrast_label(pair_key)
            clusters = cluster_top_neighbors(
                proxy.kv, proxy.beta_unit,
                side=side, lang=proxy.lang, **kwargs,
            )
            for c in clusters:
                c["contrast"] = label
                out.append(c)
        return out

    def snippets(self, pre_docs, *, top_per_side: int = 200, **kwargs) -> dict:
        """Snippets for all contrasts.

        Returns dict with ``contrast`` key added to each snippet dict.
        """
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
        return {"pos": all_pos, "neg": all_neg}

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
            kv=self.kv,
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
                    pairs = filtered_neighbors(proxy.kv, vec, topn=top_words, lang=proxy.lang)
                    for rank, (word, cos) in enumerate(pairs, 1):
                        words.append({"side": side, "rank": rank, "word": word, "cos": float(cos)})
                parts.append(_fmt_top_words(words, top_words).replace(
                    f"Top Words (n={top_words})",
                    f"Top Words: {label} (n={top_words})",
                ))

            if clusters is not None:
                for side in ("pos", "neg"):
                    cl = cluster_top_neighbors(
                        proxy.kv, proxy.beta_unit,
                        topn=clusters, side=side, lang=proxy.lang,
                    )
                    if cl:
                        header = _fmt_clusters(cl, side)
                        header = header.replace(
                            f"Clusters ({'+'  if side == 'pos' else '−'}β",
                            f"Clusters: {label} ({'+'  if side == 'pos' else '−'}β",
                        )
                        parts.append(header)

        text = "\n".join(parts)
        print(text)
        return text

    def __repr__(self) -> str:
        return (
            f"GroupResult({self.G} groups, n_kept={self.n_kept}, "
            f"omnibus_p={self.omnibus_p:.4f})"
        )
