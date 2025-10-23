# ssdiff/clusters.py
from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING
from .utils import filtered_neighbors

if TYPE_CHECKING:
    # Only used for hints; not imported at runtime → no circular import
    from .core import SSD
import warnings
warnings.filterwarnings(
    "ignore",
    message=r"KMeans is known to have a memory leak on Windows with MKL.*",
    category=UserWarning,
    module=r"sklearn\.cluster\._kmeans"
)

def _require_sklearn():
    try:
        import sklearn  # noqa
        from sklearn.cluster import KMeans  # noqa
        from sklearn.metrics import silhouette_score  # noqa
    except Exception:
        raise ImportError("scikit-learn is required for clustering. Install: pip install scikit-learn")

def cluster_top_neighbors(
    ssd: SSD,
    *,
    topn: int = 100,
    k: int | None = None,
    k_min: int = 2,
    k_max: int = 10,
    restrict_vocab: int = 50000,
    random_state: int = 13,
    min_cluster_size: int = 2,
    side: str = "pos",  # "pos" → +β̂, "neg" → −β̂
):
    _require_sklearn()
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    b = ssd.beta_unit if ssd.use_unit_beta else ssd.beta
    vec = b if side == "pos" else -b

    pairs = filtered_neighbors(ssd.kv, vec, topn=topn, restrict=restrict_vocab)
    words = [w for (w, _s) in pairs]
    if len(words) < max(2, k_min):
        raise ValueError("Not enough neighbors to cluster.")

    W = np.vstack([ssd.kv.get_vector(w, norm=True).astype(np.float64) for w in words])  # unit rows

    def choose_k_auto(W, kmin, kmax):
        best_k, best_s = None, -1.0
        upper = min(kmax, max(kmin, W.shape[0]-1))
        for kk in range(max(2, kmin), max(2, upper) + 1):
            km = KMeans(n_clusters=kk, random_state=random_state, n_init="auto")
            labels = km.fit_predict(W)
            if len(set(labels)) <= 1 or np.max(np.bincount(labels)) <= 1:
                continue
            s = silhouette_score(W, labels)
            if s > best_s:
                best_s, best_k = s, kk
        return best_k if best_k is not None else max(2, kmin)

    k_use = int(k) if k is not None else choose_k_auto(W, k_min, k_max)
    km = KMeans(n_clusters=k_use, random_state=random_state, n_init="auto")
    labels = km.fit_predict(W)

    bu = b / max(float(np.linalg.norm(b)), 1e-12)
    clusters = []
    for cid in sorted(set(labels)):
        idx = np.where(labels == cid)[0]
        if len(idx) < min_cluster_size:
            continue
        Wc = W[idx]
        centroid = Wc.mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        cos_beta = float(centroid @ bu)
        cos_to_centroid = (Wc @ centroid).astype(float)
        coherence = float(np.mean(cos_to_centroid))

        rows = []
        for j in idx:
            w = words[j]
            ccent = float(W[j] @ centroid)
            cbeta = float(W[j] @ bu)
            rows.append((w, ccent, cbeta))
        rows.sort(key=lambda t: t[1], reverse=True)

        clusters.append({
            "id": int(cid),
            "size": int(len(idx)),
            "centroid_cos_beta": cos_beta,
            "coherence": coherence,
            "words": rows,
        })

    clusters.sort(key=lambda C: C["centroid_cos_beta"], reverse=True)

    return clusters

