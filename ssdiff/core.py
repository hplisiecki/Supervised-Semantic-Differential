# ssdiff/core.py
from __future__ import annotations
import numpy as np
from typing import Any, List, Union
from gensim.models import KeyedVectors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd


try:
    from scipy import stats as _scistats
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

from .utils import (
    compute_global_sif,
    build_doc_vectors,
    filtered_neighbors,
    load_embeddings,
)




class SSD:
    """
    Analysis backend: builds per-essay vectors from SIF-weighted contexts around seeds,
    drops essays with no seed contexts, fits PCA→OLS→β (back-projected), and exposes
    β, stats, neighbors, and human-readable printouts.

    Args:
        kv: Preloaded KeyedVectors or path to embeddings file.
        docs: List of documents, each as a list of tokens (strings).
        y: Array-like of outcome variable (continuous).
        lexicon: Set or list of seed words (strings) for the concept.
        l2_normalize_docs: Whether to L2-normalize per-essay vectors before ABTT.
        use_unit_beta: Whether to return unit vector for β̂ (True) or raw magnitude (False). Raw magnitude biases the effect size estimates.
        N_PCA: Number of PCA components to retain before regression.
    """

    def __init__(
        self,
        kv: Union[KeyedVectors, str],
        docs: List[Any],
        y: np.ndarray,
        lexicon: Any,
        *,
        l2_normalize_docs: bool = True,
        use_unit_beta: bool = True,
        N_PCA = 20,
        window: int = 3,
        SIF_a: float = 1e-3,
    ) -> None:
        self.kv = kv if isinstance(kv, KeyedVectors) else load_embeddings(kv)
        self.docs = docs
        self.y = np.asarray(y)
        self.lexicon = set(list(lexicon))

        self.pos_clusters_raw = None   # type: list[dict] | None
        self.neg_clusters_raw = None   # type: list[dict] | None

        self.window = window
        self.SIF_a = SIF_a

        wc, tot = compute_global_sif(self.docs)

        # Per-essay matrix (drops essays w/o seed contexts)
        X_raw, keep = build_doc_vectors(
            self.docs, self.kv, self.lexicon,
            global_wc=wc, total_tokens=tot, window=self.window, sif_a=self.SIF_a,
        )
        if not np.any(keep):
            raise ValueError("No essays contain the lexicon for this concept; nothing to fit.")
        self.keep_mask = keep
        self.n_raw = len(self.docs)
        self.n_kept = int(keep.sum())
        self.n_dropped = self.n_raw - self.n_kept
        self.docs_kept = [d for d, k in zip(self.docs, keep) if k]
        self.y_kept = self.y[keep]
        X = X_raw

        # Optional row-L2 + doc-level ABTT
        if l2_normalize_docs:
            X = self._row_l2_normalize(X)
        self._abtt_mu_docs = np.zeros(X.shape[1], dtype=np.float64)
        self._abtt_P_docs = np.eye(X.shape[1], dtype=np.float64)
        self.x = X

        # Standardize & PCA
        self.scaler_X = StandardScaler()
        self.Xs = self.scaler_X.fit_transform(self.x)
        self.scaler_y = StandardScaler()
        self.ys = self.scaler_y.fit_transform(self.y_kept.reshape(-1, 1)).ravel()
        self.pca = PCA(n_components=N_PCA, svd_solver="full")
        self.z = self.pca.fit_transform(self.Xs)

        # Fit β in doc space
        self.use_unit_beta = use_unit_beta
        self.beta = self._fit_beta()
        self.beta_unit = self._unit(self.beta) if self.use_unit_beta else self.beta

        # Calibration & diagnostics
        self._calibrate_effect()



    # ---------- Public API ----------
    def nbrs(self, sign: int = +1, n: int = 16, restrict_vocab: int = 10000):
        """Return list of (word, cosine, shift) for nearest neighbors to ±β̂ using base model (kv)."""
        b = self.beta_unit if self.use_unit_beta else self.beta
        vec = b if sign > 0 else -b
        out = []
        for w, sim in filtered_neighbors(self.kv, vec, topn=n, restrict=restrict_vocab):
            out.append((w, sim, float(self.kv[w].dot(vec))))
        return out

    def print_model_stats(self) -> None:
        """Pretty printer for model diagnostics."""
        print(f"Kept {self.n_kept} / {self.n_raw} essays (dropped {self.n_dropped} with no seed occurrence).")
        print("Model Statistics:")
        print(f"R² = {self.r2:.4f}")
        print(f"F-statistic = {self.f_stat:.4f}")
        ptxt = f"{self.f_pvalue:.6f}" if np.isfinite(self.f_pvalue) else "n/a (SciPy missing)"
        print(f"F-test p-value = {ptxt}")
        print("\nCalibration:")
        print(f"‖β‖ (SD(y) per +1.0 cosine) = {self.beta_norm_stdCN:.4f}")
        print(f"Δy per +0.10 cosine         = {self.delta_per_0p10_raw:.4f}")
        print(f"IQR(cos) effect (raw y)     = {self.iqr_effect_raw:.4f}")
        print(f"Corr(y, Xβ)                 = {self.y_corr_pred:.4f}")

    import pandas as pd

    def top_words(self, n: int = 10, *, verbose: bool = False) -> pd.DataFrame:
        """
        Return a DataFrame of the top-N neighbors on both poles.
        Columns: side ['pos','neg'], rank (1..N), word, cos, shift, shift_signed

        - cos: cosine similarity to the queried pole (+β̂ for 'pos', −β̂ for 'neg')
        - shift: dot(word, pole_vector) → always positive within each pole
        - shift_signed: dot(word, +β̂_unit) → positive for 'pos' words, negative for 'neg' words
        """
        b = self.beta_unit if getattr(self, "use_unit_beta", True) else self.beta
        
        rows = []
        for side, vec in (("pos", b), ("neg", -b)):
            pairs = filtered_neighbors(self.kv, vec, topn=n)
            for rank, (w, sim) in enumerate(pairs, start=1):
                rows.append({
                    "side": side,
                    "rank": rank,
                    "word": w,
                    "cos": float(sim),
                })

        df = pd.DataFrame(rows, columns=["side", "rank", "word", "cos"])

        if verbose:
            # Pretty print (unchanged look-and-feel)
            print("\n--- Words aligned with β̂ ---")
            print(f"{'Word':<18} {'cos':>7} {'shift':>9}")
            for _, r in df[df["side"] == "pos"].sort_values("rank").iterrows():
                print(f"{r['word']:<18} {r['cos']:>7.4f}")

            print("\n--- Words opposed to β̂ ---")
            print(f"{'Word':<18} {'cos':>7} {'shift':>9}")
            for _, r in df[df["side"] == "neg"].sort_values("rank").iterrows():
                print(f"{r['word']:<18} {r['cos']:>7.4f}")

        return df

    # ---------- Internals ----------
    def _fit_beta(self) -> np.ndarray:
        ys = self.ys
        w_reg = np.linalg.solve(self.z.T @ self.z, self.z.T @ ys)
        y_pred = self.z @ w_reg
        resid = ys - y_pred
        n = len(ys)
        p = len(w_reg)
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
        self.r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        if n - p - 1 > 0:
            self.r2_adj = 1.0 - (1.0 - self.r2) * (n - 1) / (n - p - 1)
        else:
            self.r2_adj = np.nan

        msr = (ss_tot - ss_res) / max(p, 1)
        mse = ss_res / (n - p - 1) if (n - p - 1) > 0 else np.inf
        self.f_stat = msr / mse if np.isfinite(mse) and mse > 0 else 0.0
        if _HAS_SCIPY and np.isfinite(mse):
            self.f_pvalue = 1 - _scistats.f.cdf(self.f_stat, p, n - p - 1)
        else:
            self.f_pvalue = np.nan

        beta_std = self.pca.components_.T.dot(w_reg)
        scale = np.where(self.scaler_X.scale_ > 0, self.scaler_X.scale_, 1.0)
        beta_docspace = beta_std / scale
        beta = self._abtt_P_docs.dot(beta_docspace)
        return beta

    def _calibrate_effect(self) -> None:
        """
        Map gradient strength to CN units and orient beta so that
        corr(ys, X @ beta) >= 0. Recompute all diagnostics accordingly.
        """
        # --- Orientation: make sure higher alignment => higher outcome ---
        yhat_std = (self.x @ self.beta).ravel()
        ys_std = self.ys
        denom = float(np.std(ys_std) * np.std(yhat_std))
        corr = float(np.corrcoef(ys_std, yhat_std)[0, 1]) if denom > 0 else 0.0

        if corr < 0:
            # Flip beta and predictions so the direction is consistent
            self.beta = -self.beta
            corr = -corr  # now non-negative

        # Update the unit vector after any potential flip
        self.beta_unit = self._unit(self.beta) if getattr(self, "use_unit_beta", True) else self.beta

        # --- Strength of beta in standardized CN units ---
        self.beta_norm_stdCN = float(np.linalg.norm(self.beta))

        # Outcome scaler (raw CN units)
        self.y_mean = float(getattr(self.scaler_y, "mean_", np.array([0.0]))[0])
        self.y_std = float(getattr(self.scaler_y, "scale_", np.array([1.0]))[0])

        # Cosine alignment for each doc against unit beta
        bu = self._unit(self.beta)
        x_unit = self._row_l2_normalize(self.x)  # safe even if x already L2
        cos_align = (x_unit @ bu).ravel()
        self.cos_align = cos_align

        # Store correlation magnitude (orientation fixed)
        self.y_corr_pred = corr  # equals sqrt(R^2) for standardized y, in [0,1]

        # Effect per +0.10 cosine in raw CN units
        delta_sd_per_0p10 = 0.10 * self.beta_norm_stdCN
        self.delta_per_0p10_raw = delta_sd_per_0p10 * self.y_std

        # IQR effect in raw CN units
        q75, q25 = np.percentile(cos_align, [75, 25])
        iqr_cos = float(q75 - q25)
        self.iqr_effect_raw = iqr_cos * self.beta_norm_stdCN * self.y_std

    # inside class ssd
    def doc_scores(self) -> dict:
        """
        Per-essay scores for documents actually used in the regression (kept mask).
        Returns:
            {
              'keep_mask':   np.bool_[n_docs_raw],  # True = essay kept (has seed contexts)
              'cos_align':   np.float64[n_kept],    # cosine(x_i, beta_unit) in [-1,1]
              'score_std':   np.float64[n_kept],    # x_i · beta  (standardized-y units)
              'yhat_raw':    np.float64[n_kept],    # back to raw y units: y_mean + y_std * score_std
            }
        """
        # unit beta
        bu = self._unit(self.beta)
        # if docs in self.x weren’t unit, get cosine anyway by re-normalizing rows
        x_unit = self._row_l2_normalize(self.x)
        cos_align = (x_unit @ bu).astype(np.float64)

        score_std = (self.x @ self.beta).astype(np.float64)
        yhat_raw = self.y_mean + self.y_std * score_std

        return dict(
            keep_mask=self.keep_mask.copy(),
            cos_align=cos_align,
            score_std=score_std,
            yhat_raw=yhat_raw,
        )

    def cluster_neighbors_sign(
            self,
            *,
            side: str = "pos",
            topn: int = 100,
            k: int | None = None,
            k_min: int = 2,
            k_max: int = 10,
            restrict_vocab: int = 50000,
            random_state: int = 13,
            min_cluster_size: int = 2,
            top_words: int = 10,
            verbose: bool = False,
    ):
        """
        Cluster top neighbors around +β̂ or −β̂.
        Returns (df_clusters, df_members) and stores raw clusters internally
        (self.pos_clusters_raw / self.neg_clusters_raw) for downstream snippets.
        """
        from .clusters import cluster_top_neighbors

        clusters = cluster_top_neighbors(
            self,
            topn=topn,
            k=k,
            k_min=k_min,
            k_max=k_max,
            restrict_vocab=restrict_vocab,
            random_state=random_state,
            min_cluster_size=min_cluster_size,
            side=side,
        )

        # Store raw clusters for later (snippets, etc.)
        if side == "pos":
            self.pos_clusters_raw = clusters
        else:
            self.neg_clusters_raw = clusters

        import pandas as pd

        rows_summary, rows_members = [], []
        for rank, C in enumerate(clusters, start=1):
            # summary row
            top = [w for (w, _ccent, _cbeta) in C["words"][:top_words]]
            rows_summary.append({
                "side": side,
                "cluster_rank": rank,  # key
                "size": C.get("size", len(C["words"])),
                "centroid_cos_beta": C.get("centroid_cos_beta", float("nan")),
                "coherence": C.get("coherence", float("nan")),
                "top_words": ", ".join(top),
            })
            # members
            for (w, ccent, cbeta) in C["words"]:
                rows_members.append({
                    "side": side,
                    "cluster_rank": rank,  # key
                    "word": w,
                    "cos_to_centroid": float(ccent),
                    "cos_to_beta": float(cbeta),
                })

        df_clusters = pd.DataFrame(rows_summary, columns=[
            "side", "cluster_rank", "size", "centroid_cos_beta", "coherence", "top_words"
        ])
        df_members = pd.DataFrame(rows_members, columns=[
            "side", "cluster_rank", "word", "cos_to_centroid", "cos_to_beta"
        ])

        if verbose:
            pole = "+β̂" if side == "pos" else "−β̂"
            title = f"Themes among neighbors of {pole}"
            print(f"\n{title}\n" + "-" * len(title))
            for _, row in df_clusters.sort_values("cluster_rank").iterrows():
                print(f"\nCluster {int(row.cluster_rank)}")
                print(f"  size={int(row.size)}  centroid·β̂={row.centroid_cos_beta:.3f}  coherence={row.coherence:.3f}")
                print(f"  top: {row.top_words}")

        return df_clusters, df_members

    def cluster_neighbors(
        self,
        *,
        topn: int = 100,
        k: int | None = None,
        k_min: int = 2,
        k_max: int = 10,
        restrict_vocab: int = 50000,
        random_state: int = 13,
        min_cluster_size: int = 2,
        top_words: int = 10,
        verbose: bool = False,
    ):
        """
        Convenience: run clustering for both +β̂ and −β̂ and return concatenated DFs.
        Also stores self.pos_clusters_raw / self.neg_clusters_raw.
        """
        import pandas as pd

        df_pos_clusters, df_pos_members = self.cluster_neighbors_sign(
            side="pos",
            topn=topn, k=k, k_min=k_min, k_max=k_max,
            restrict_vocab=restrict_vocab,
            random_state=random_state,
            min_cluster_size=min_cluster_size,
            top_words=top_words,
            verbose = verbose,
        )
        df_neg_clusters, df_neg_members = self.cluster_neighbors_sign(
            side="neg",
            topn=topn, k=k, k_min=k_min, k_max=k_max,
            restrict_vocab=restrict_vocab,
            random_state=random_state,
            min_cluster_size=min_cluster_size,
            top_words=top_words,
            verbose = verbose,
        )

        df_clusters = pd.concat([df_pos_clusters, df_neg_clusters], ignore_index=True)
        df_members  = pd.concat([df_pos_members,  df_neg_members ], ignore_index=True)
        return df_clusters, df_members

    def cluster_snippets(
            self,
            *,
            pre_docs,
            side: str = "both",  # "pos", "neg", or "both"
            window_sentences: int = 1,  # [sent-1, sent, sent+1]
            seeds=None,  # defaults to self.lexicon
            top_per_cluster: int = 100,
    ) -> dict:
        """
        Collect text snippets (surface sentences) most aligned with each cluster centroid.
        Requires that clustering has been run via `cluster_neighbors_sign(...)` first.

        Returns:
            dict with keys:
                - "pos": DataFrame of positive-side cluster snippets (if requested)
                - "neg": DataFrame of negative-side cluster snippets (if requested)
        """
        # Defensive import (avoid circular)
        from .snippets import cluster_snippets_by_centroids

        # Ensure clusters exist
        need_pos = side in ("pos", "both")
        need_neg = side in ("neg", "both")

        if need_pos and (self.pos_clusters_raw is None or len(self.pos_clusters_raw) == 0):
            raise RuntimeError(
                "Positive-side clusters not available. Run `cluster_neighbors_sign(side='pos')` first."
            )
        if need_neg and (self.neg_clusters_raw is None or len(self.neg_clusters_raw) == 0):
            raise RuntimeError(
                "Negative-side clusters not available. Run `cluster_neighbors_sign(side='neg')` first."
            )

        # Build request
        pos_clusters = self.pos_clusters_raw if need_pos else []
        neg_clusters = self.neg_clusters_raw if need_neg else []
        seeds = set(seeds or getattr(self, "lexicon", set()))

        # Call snippet extractor
        return cluster_snippets_by_centroids(
            pre_docs=pre_docs,
            ssd=self,
            pos_clusters=pos_clusters,
            neg_clusters=neg_clusters,
            token_window = self.window,
            seeds=seeds,
            sif_a=self.SIF_a,
            top_per_cluster=top_per_cluster,
        )

    def beta_snippets(
            self,
            *,
            pre_docs,
            window_sentences: int = 1,
            seeds=None,
            top_per_side: int = 200,
            min_cosine: float | None = None,
    ) -> dict:
        """
        Collect text snippets most aligned with the β̂ direction itself (not cluster centroids).
        Returns dict with:
            - "beta_pos": DataFrame of most positive snippets
            - "beta_neg": DataFrame of most negative snippets
        """
        from .snippets import snippets_along_beta

        seeds = set(seeds or getattr(self, "lexicon", set()))
        return snippets_along_beta(
            pre_docs=pre_docs,
            ssd=self,
            token_window = self.window,
            seeds=seeds,
            sif_a=self.SIF_a,
            top_per_side=top_per_side,
            min_cosine=min_cosine,
        )

    def ssd_scores(
            self,
            include_all: bool = True,
            return_df: bool = True,
            include_true: bool = True,
    ):
        """
        Compute per-document SSD scores from the fitted model.

        Returns, for each original document index (0..n_raw-1):
          - cos:        cosine alignment of the document vector to the unit gradient β̂
          - yhat_std:   predicted outcome in standardized units (X @ β)
          - yhat_raw:   predicted outcome mapped back to original units
          - kept:       whether this doc contributed a valid concept vector (had seed context)
          - (optional) y_true_std, y_true_raw for kept docs; NaN for dropped

        Parameters
        ----------
        include_all : bool
            If True (default), return a row for every original doc index, with NaNs for
            dropped docs. If False, return only kept docs (those used in regression).
        return_df : bool
            If True (default), return a pandas.DataFrame. If False, return a dict of np.ndarrays.
        include_true : bool
            If True (default), include observed outcome values for kept docs
            (`y_true_std`, `y_true_raw`).

        Notes
        -----
        - Assumes `self.x` holds the (optionally row-L2’d and ABTT-processed) document
          vectors for the kept docs (those with self.keep_mask == True).
        - `self.beta` is the regression gradient in document space (standardized-y units).
        - `self.beta_unit` is the unit-norm version of `self.beta`.
        - `self.cos_align` was computed in _calibrate_effect() as
             cos_align = normalize_rows(self.x) @ self.beta_unit
          for the kept docs.
        - Predictions in raw units are formed via the fitted `self.scaler_y`:
             yhat_raw = y_mean + y_std * yhat_std
        """
        import numpy as np
        try:
            import pandas as pd
        except Exception:
            pd = None  # allow non-DataFrame return if pandas not available

        if not hasattr(self, "x") or not hasattr(self, "beta") or not hasattr(self, "keep_mask"):
            raise RuntimeError("Model appears unfitted: missing x/beta/keep_mask. Fit before calling ssd_scores().")

        n_raw = getattr(self, "n_raw", len(self.docs))
        keep = self.keep_mask
        if keep is None or len(keep) != n_raw:
            raise RuntimeError("Invalid or missing keep_mask; cannot map scores back to original doc indices.")

        # Per-kept-doc projections in standardized y-units
        # yhat_std_k: X_k @ β  (vector length = n_kept)
        yhat_std_k = (self.x @ self.beta).astype(float).ravel()

        # Cosine alignment for kept docs (already computed and stored)
        cos_k = np.array(self.cos_align, dtype=float)

        # Raw-scale mapping via fitted scaler on y
        y_mean = float(
            getattr(self, "y_mean", getattr(self, "scaler_y", None).mean_[0] if hasattr(self, "scaler_y") else 0.0))
        y_std = float(
            getattr(self, "y_std", getattr(self, "scaler_y", None).scale_[0] if hasattr(self, "scaler_y") else 1.0))
        if y_std == 0.0:
            y_std = 1.0
        yhat_raw_k = y_mean + y_std * yhat_std_k

        # Prepare full-length arrays (size = n_raw), fill with NaN, then insert kept
        def _full_like(vals_k):
            out = np.full((n_raw,), np.nan, dtype=float)
            out[keep] = vals_k
            return out

        cos_full = _full_like(cos_k)
        yhat_std_full = _full_like(yhat_std_k)
        yhat_raw_full = _full_like(yhat_raw_k)
        kept_full = keep.astype(bool).copy()

        result = {
            "doc_index": np.arange(n_raw, dtype=int),
            "kept": kept_full,
            "cos": cos_full if include_all else cos_k,
            "yhat_std": yhat_std_full if include_all else yhat_std_k,
            "yhat_raw": yhat_raw_full if include_all else yhat_raw_k,
        }

        if include_true and hasattr(self, "y_kept"):
            # Standardized observed = self.ys (length n_kept)
            ys_std_k = np.array(self.ys, dtype=float).ravel()
            y_true_std_full = _full_like(ys_std_k)
            y_true_raw_full = _full_like(y_mean + y_std * ys_std_k)
            if include_all:
                result["y_true_std"] = y_true_std_full
                result["y_true_raw"] = y_true_raw_full
            else:
                result["y_true_std"] = ys_std_k
                result["y_true_raw"] = y_mean + y_std * ys_std_k

        if return_df:
            if pd is None:
                raise RuntimeError(
                    "pandas is required to return a DataFrame. Call with return_df=False or install pandas.")
            cols = ["doc_index", "kept", "cos", "yhat_std", "yhat_raw"]
            if include_true:
                cols += ["y_true_std", "y_true_raw"]
            return pd.DataFrame({c: result[c] for c in cols})
        return result

    @staticmethod
    def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        n = float(np.linalg.norm(v))
        return v / max(n, eps)

    @staticmethod
    def _row_l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, eps)
        return X / norms

    @staticmethod
    def _apply_abtt_matrix(X: np.ndarray, m: int):
        mu = X.mean(axis=0, dtype=np.float64)
        Xc = X - mu
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        top = Vt[:m, :]
        P = np.eye(Vt.shape[1], dtype=np.float64) - top.T @ top
        Xp = Xc @ P
        return Xp, mu, P
