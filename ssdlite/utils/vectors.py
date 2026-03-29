"""SIF-weighted document vector construction."""

from __future__ import annotations

import numpy as np

from .math import l2_normalize_rows_inplace


def compute_global_sif(sentences: list[list[str]]) -> tuple[dict[str, int], int]:
    """Compute word counts for SIF weighting.

    Parameters
    ----------
    sentences : list[list[str]]
        Tokenized sentences (each sentence is a list of token strings).

    Returns
    -------
    word_counts : dict[str, int]
        Mapping from each word to its total count across all sentences.
    total_tokens : int
        Sum of all word counts.
    """
    wc: dict[str, int] = {}
    for sent in sentences:
        for t in sent:
            wc[t] = wc.get(t, 0) + 1
    return wc, sum(wc.values())


def _occ_vectors_in_doc(doc, kv, lexicon, wc, tot, window, sif_a):
    """SIF-averaged context vectors for each seed occurrence in a doc."""
    occ = []
    D = kv.vector_size
    for i, token in enumerate(doc):
        if token not in lexicon:
            continue
        start, end = max(0, i - window), min(len(doc), i + window + 1)
        sum_v = np.zeros(D, dtype=np.float64)
        w_sum = 0.0
        for j in range(start, end):
            if j == i:
                continue
            c = doc[j]
            if c not in kv:
                continue
            a = sif_a / (sif_a + wc.get(c, 0) / max(tot, 1))
            sum_v += a * kv[c]
            w_sum += a
        if w_sum > 0:
            occ.append(sum_v / w_sum)
    return occ


def _full_doc_vector(tokens, kv, wc, tot, sif_a) -> np.ndarray | None:
    """SIF-weighted mean of all tokens in a doc (no lexicon filtering)."""
    D = kv.vector_size
    sum_v = np.zeros(D, dtype=np.float64)
    w_sum = 0.0
    for c in tokens:
        if c not in kv:
            continue
        a = sif_a / (sif_a + wc.get(c, 0) / max(tot, 1))
        sum_v += a * kv[c]
        w_sum += a
    if w_sum == 0.0:
        return None
    return (sum_v / w_sum).astype(np.float64)


def build_doc_vectors(
    docs,
    kv,
    lexicon,
    global_wc,
    total_tokens,
    window,
    sif_a,
    *,
    mode: str = "seed",
) -> tuple[np.ndarray, np.ndarray]:
    """Build SIF-weighted document vectors.

    Parameters
    ----------
    docs : list[list[str]] or list[list[list[str]]]
        Flat token lists or grouped profiles.
    kv : Embeddings
        Word embeddings.
    lexicon : set[str]
        Seed words.
    global_wc, total_tokens : word counts from compute_global_sif.
    window : int
        Context window around seed words.
    sif_a : float
        SIF smoothing parameter.
    mode : "seed" | "full"
        "seed" uses lexicon-seeded contexts; "full" uses entire document.

    Returns
    -------
    X : (n_kept, D) float64 document vectors
    keep_mask : (n_docs,) boolean mask
    """
    if mode not in {"seed", "full"}:
        raise ValueError("mode must be 'seed' or 'full'.")

    use_seeds = mode == "seed"
    X_list = []
    keep_mask = []

    # Detect flat vs grouped docs
    is_flat = None
    for item in docs:
        if item:
            is_flat = isinstance(item[0], str)
            break
    if is_flat is None:
        return np.zeros((0, kv.vector_size), dtype=np.float64), np.zeros(0, dtype=bool)

    if is_flat:
        for d in docs:
            if not d:
                keep_mask.append(False)
                continue
            if use_seeds:
                occ = _occ_vectors_in_doc(d, kv, lexicon, global_wc, total_tokens, window, sif_a)
                if not occ:
                    keep_mask.append(False)
                else:
                    keep_mask.append(True)
                    X_list.append(np.mean(occ, axis=0).astype(np.float64))
            else:
                v = _full_doc_vector(d, kv, global_wc, total_tokens, sif_a)
                if v is None:
                    keep_mask.append(False)
                else:
                    keep_mask.append(True)
                    X_list.append(v)
    else:
        for posts in docs:
            if not posts:
                keep_mask.append(False)
                continue
            if use_seeds:
                occ_all = []
                for p in posts:
                    if p:
                        occ_all.extend(
                            _occ_vectors_in_doc(p, kv, lexicon, global_wc, total_tokens, window, sif_a)
                        )
                if not occ_all:
                    keep_mask.append(False)
                else:
                    keep_mask.append(True)
                    X_list.append(np.mean(occ_all, axis=0).astype(np.float64))
            else:
                tokens_all = [t for p in posts for t in p]
                v = _full_doc_vector(tokens_all, kv, global_wc, total_tokens, sif_a)
                if v is None:
                    keep_mask.append(False)
                else:
                    keep_mask.append(True)
                    X_list.append(v)

    X = np.vstack(X_list) if X_list else np.zeros((0, kv.vector_size), dtype=np.float64)
    return X, np.array(keep_mask, dtype=bool)


def build_and_normalize_doc_vectors(
    docs,
    kv,
    lexicon,
    window: int = 3,
    sif_a: float = 1e-3,
    use_full_doc: bool = False,
    l2_normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute SIF word counts, build document vectors, and optionally L2-normalize.

    High-level entry point that chains ``compute_global_sif`` and
    ``build_doc_vectors``, then optionally normalizes rows to unit length.

    Parameters
    ----------
    docs : list[list[str]] or list[list[list[str]]]
        Tokenized documents. Either flat token lists or grouped profiles
        (list of posts per document).
    kv : Embeddings
        Word embedding model providing vector lookups.
    lexicon : set[str]
        Seed words used for context-window extraction in "seed" mode.
    window : int, default 3
        Context window half-width around each seed word.
    sif_a : float, default 1e-3
        SIF smoothing parameter (lower values give more aggressive
        down-weighting of frequent words).
    use_full_doc : bool, default False
        If True, use the entire document for vectorization ("full" mode)
        instead of only seed-word contexts ("seed" mode).
    l2_normalize : bool, default True
        If True, L2-normalize each document vector to unit length.

    Returns
    -------
    X : (n_kept, D) float64
        Document vectors for documents that produced valid representations.
    keep_mask : (n_docs,) bool ndarray
        Boolean mask indicating which input documents yielded a valid vector.
    """
    # Flatten docs for SIF counting
    flat_sents = []
    for item in docs:
        if not item:
            continue
        if isinstance(item[0], str):
            flat_sents.append(item)
        else:
            for sub in item:
                if sub:
                    flat_sents.append(sub)

    global_wc, total_tokens = compute_global_sif(flat_sents)
    mode = "full" if use_full_doc else "seed"

    X, keep_mask = build_doc_vectors(
        docs, kv, lexicon, global_wc, total_tokens, window, sif_a, mode=mode,
    )

    if l2_normalize and X.shape[0] > 0:
        l2_normalize_rows_inplace(X)

    return X, keep_mask
