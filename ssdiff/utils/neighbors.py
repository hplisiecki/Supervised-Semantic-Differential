"""Neighbor search and clustering in embedding space."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ssdiff.lang_config import get_config

from .math import kmeans, kmeans_auto_k, unit_vector


def filtered_neighbors(
    embeddings,
    vec,
    topn: int = 20,
    cand: int = 2000,
    restrict: int = 10000,
    lang: str = "pl",
) -> list[tuple[str, float]]:
    """Return top cosine neighbors, filtering out numbers and capitalized tokens.

    Queries the embedding model for candidate neighbors, then discards tokens
    that contain digits or start with an uppercase letter, returning the first
    ``topn`` clean results.  The filter pattern is language-aware (configured
    in :mod:`ssdiff.lang_config`).

    Parameters
    ----------
    embeddings : Embeddings
        Word embedding model supporting ``similar_by_vector``.
    vec : (D,) ndarray
        Query vector in embedding space.
    topn : int, default 20
        Number of filtered results to return.
    cand : int, default 2000
        Size of the raw candidate pool retrieved from the embedding model.
        Must be >= ``topn`` to ensure enough results survive filtering.
    restrict : int, default 10000
        Restrict the search to the top-N most frequent words in the vocabulary.
    lang : str, default ``"pl"``
        Language code — selects the token-filter regex from
        :func:`lang_config.get_config`.

    Returns
    -------
    list[tuple[str, float]]
        Filtered neighbors as ``(word, cosine_similarity)`` pairs, sorted by
        descending similarity.
    """
    bad_token = get_config(lang).bad_token_re
    nbrs = embeddings.similar_by_vector(vec, topn=cand, restrict_vocab=restrict)
    out = []
    for w, sim in nbrs:
        if not bad_token.match(w):
            out.append((w, sim))
            if len(out) >= topn:
                break
    return out


def cluster_top_neighbors(
    embeddings,
    beta: np.ndarray,
    *,
    topn: int = 100,
    k: int | None = None,
    k_min: int = 2,
    k_max: int = 10,
    restrict_vocab: int = 50000,
    random_state: int = 2137,
    min_cluster_size: int = 2,
    side: Literal["pos", "neg"] = "pos",
    lang: str = "pl",
) -> list[dict]:
    """Cluster the top vocabulary neighbors of +/-beta into interpretable themes.

    Retrieves the *topn* closest words to the positive or negative end of the
    gradient *beta*, embeds them, and groups them with pure-numpy KMeans (either
    a fixed *k* or automatic selection via silhouette score).

    Parameters
    ----------
    embeddings : Embeddings
        Word embedding model supporting ``similar_by_vector`` and
        ``get_vector``.
    beta : ndarray of shape (D,)
        Gradient / loading vector from a fitted SSD model.
    topn : int, default 100
        Number of filtered neighbors to retrieve and cluster.
    k : int or None, default None
        Fixed number of clusters.  If None, *k* is chosen automatically in
        ``[k_min, k_max]`` by maximising the silhouette score.
    k_min : int, default 2
        Minimum number of clusters when *k* is selected automatically.
    k_max : int, default 10
        Maximum number of clusters when *k* is selected automatically.
    restrict_vocab : int, default 50000
        Restrict neighbor search to the top-N most frequent vocabulary words.
    random_state : int, default 2137
        Random seed for KMeans reproducibility.
    min_cluster_size : int, default 2
        Clusters with fewer than this many words are silently dropped.
    side : "pos" or "neg", default "pos"
        Which end of *beta* to query.  ``"pos"`` queries ``+beta``;
        ``"neg"`` queries ``-beta``.
    lang : str, default "pl"
        Language code for the token-filter regex (see
        :func:`~ssdiff.lang_config.get_config`).

    Returns
    -------
    list[dict]
        Cluster dicts, each with keys:
        ``id``, ``size``, ``centroid_cos_beta``, ``coherence``, ``words``.
        Sorted by ``centroid_cos_beta`` descending for ``"pos"`` and
        ascending for ``"neg"``.

    Raises
    ------
    ValueError
        If fewer than ``max(2, k_min)`` valid neighbors are found.
    """
    bu = unit_vector(beta)
    vec = bu if side == "pos" else -bu

    pairs = filtered_neighbors(embeddings, vec, topn=topn, restrict=restrict_vocab, lang=lang)
    words = [w for (w, _s) in pairs]
    if len(words) < max(2, k_min):
        raise ValueError("Not enough neighbors to cluster.")

    W = np.vstack(
        [embeddings.get_vector(w, norm=True).astype(np.float64) for w in words]
    )

    if k is not None:
        k_clamped = min(int(k), len(words))
        labels, centers, inertia = kmeans(W, k=k_clamped, random_state=random_state)
    else:
        labels, centers, inertia, k_use = kmeans_auto_k(
            W, k_min=k_min, k_max=min(k_max, len(words)), random_state=random_state,
        )

    clusters = []
    for cid in sorted(set(labels)):
        idx = np.where(labels == cid)[0]
        if len(idx) < min_cluster_size:
            continue
        Wc = W[idx]
        centroid = Wc.mean(axis=0)
        centroid = unit_vector(centroid)
        cos_beta = float(centroid @ bu)
        cos_to_centroid = (Wc @ centroid).astype(float)
        coherence = float(np.mean(cos_to_centroid))

        rows = []
        for j in idx:
            w = words[j]
            ccent = float(W[j] @ centroid)
            cbeta = float(W[j] @ bu)
            rows.append({"word": w, "cos_centroid": ccent, "cos_beta": cbeta})
        rows.sort(key=lambda t: t["cos_centroid"], reverse=True)

        clusters.append({
            "id": int(cid),
            "size": len(idx),
            "centroid_cos_beta": cos_beta,
            "coherence": coherence,
            "words": rows,
        })

    if side == "pos":
        clusters.sort(key=lambda c: c["centroid_cos_beta"], reverse=True)
    else:
        clusters.sort(key=lambda c: c["centroid_cos_beta"], reverse=False)

    return clusters
