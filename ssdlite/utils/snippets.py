# ===== ssdlite/utils/snippets.py =====
"""
Snippet extraction along beta and cluster centroids.

Ported from ssdiff.snippets — all pandas usage replaced with plain Python types.
Mathematical logic is identical.
"""
from __future__ import annotations
from typing import List, Iterable, Iterator, Tuple, Optional, Dict, Any
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from ssdlite.utils.math import unit_vector
from ssdlite.utils.text import PreprocessedDoc
from ssdlite.utils.vectors import compute_global_sif


def _centroid_unit_from_cluster_words(words: list, kv) -> np.ndarray:
    """Compute unit centroid vector from cluster words.

    Accepts both dict format (from cluster_top_neighbors: {"word": ..., ...})
    and tuple format ((word_str, score, rank)).
    """
    vecs = []
    for entry in words:
        w = entry["word"] if isinstance(entry, dict) else entry[0]
        if w in kv:
            vecs.append(kv.get_vector(w, norm=True))
    if not vecs:
        return np.zeros(kv.vector_size, dtype=np.float64)
    c = np.mean(np.vstack(vecs), axis=0)
    return unit_vector(c)


# a lightweight "doc-like" view used for both single-doc and per-post in profiles
class _DocLike:
    __slots__ = (
        "profile_id",
        "post_id",
        "sents_surface",
        "doc_lemmas",
        "token_to_sent",
    )

    def __init__(
        self,
        profile_id: int,
        post_id: int,
        sents_surface: List[str],
        doc_lemmas: List[str],
        token_to_sent: List[int],
    ) -> None:
        self.profile_id = profile_id  # for PreprocessedDoc: equals doc_id
        self.post_id = post_id  # for PreprocessedDoc: 0
        self.sents_surface = sents_surface
        self.doc_lemmas = doc_lemmas
        self.token_to_sent = token_to_sent


def _iter_doclikes(
    pre_docs: List[PreprocessedDoc],
) -> Iterator[_DocLike]:
    """
    Iterate over all docs as _DocLike objects.
    Yields _DocLike objects for each doc.
    """
    if not pre_docs:
        return
    for di, P in enumerate(pre_docs):
        yield _DocLike(
            profile_id=di,
            post_id=0,
            sents_surface=P.sents_surface,
            doc_lemmas=P.doc_lemmas,
            token_to_sent=P.token_to_sent,
        )


def _build_global_sif(pre_docs) -> Tuple[dict[str, int], int]:
    """Build global word counts and total token count from pre_docs."""
    flat = [D.doc_lemmas for D in _iter_doclikes(pre_docs)]
    return compute_global_sif(flat)


def _make_snippet_anchor(
    D: _DocLike, i: int, start_tok: int, end_tok: int
) -> tuple[str, int, int]:
    """
    Create snippet anchor text for occurrence at token i with context [start_tok, end_tok].
    Returns (snippet_anchor, start_sent_idx, end_sent_idx).
    """
    s_idx = D.token_to_sent[i] if i < len(D.token_to_sent) else 0
    start_tok = max(0, min(start_tok, len(D.doc_lemmas) - 1))
    end_tok = max(0, min(end_tok, len(D.doc_lemmas) - 1))
    start_sent = (
        D.token_to_sent[start_tok] if start_tok < len(D.token_to_sent) else s_idx
    )
    end_sent = D.token_to_sent[end_tok] if end_tok < len(D.token_to_sent) else s_idx

    if start_sent == s_idx and end_sent == s_idx:
        return D.sents_surface[s_idx], s_idx, s_idx

    if start_sent < s_idx:
        prev_idx = s_idx - 1
        if prev_idx >= 0:
            return (
                (D.sents_surface[prev_idx] + " " + D.sents_surface[s_idx]).strip(),
                prev_idx,
                s_idx,
            )

    next_idx = s_idx + 1
    if next_idx < len(D.sents_surface):
        return (
            (D.sents_surface[s_idx] + " " + D.sents_surface[next_idx]).strip(),
            s_idx,
            next_idx,
        )

    return D.sents_surface[s_idx], s_idx, s_idx


# ---------- vectorized per-doc precompute ----------
def _precompute_doc_arrays(
    kv,
    D: _DocLike,
    sif_a: float,
    global_wc: dict[str, int],
    total_tokens: int,
) -> Dict[str, Any]:
    """
    Build SIF-weighted token matrix once per doc:
      - w      : (N,) SIF weights
      - V      : (N, d) unit vectors (zeros for OOV)
      - W      : (N, d) = w[:,None]*V
      - CW     : (N+1, d) cumulative sum with leading zero row
    """
    toks = D.doc_lemmas
    N = len(toks)
    d = int(kv.vector_size)
    if N == 0:
        return dict(N=0)

    # weights
    w = np.fromiter(
        (sif_a / (sif_a + global_wc.get(t, 0) / max(total_tokens, 1)) for t in toks),
        dtype=np.float64,
        count=N,
    )

    # vectors (unit rows; zero if OOV)
    V = np.zeros((N, d), dtype=np.float64)
    hit = np.zeros(N, dtype=bool)
    for i, t in enumerate(toks):
        if t in kv:
            V[i] = kv.get_vector(t, norm=True)
            hit[i] = True

    W = V * w[:, None]
    CW = np.vstack([np.zeros((1, d), dtype=np.float64), np.cumsum(W, axis=0)])
    return dict(
        N=N,
        toks=toks,
        w=w,
        V=V,
        W=W,
        CW=CW,
        hit=hit,
        token_to_sent=D.token_to_sent,
        sents_surface=D.sents_surface,
        profile_id=D.profile_id,
        post_id=D.post_id,
    )


def _occ_vec(
    CW: np.ndarray, W: np.ndarray, i: int, L: int, R: int
) -> Optional[np.ndarray]:
    """Inclusive [L,R], exclude center i. Returns unit vector or None if zero."""
    L = max(0, L)
    R = min(W.shape[0] - 1, R)
    if R < L:
        return None
    S = CW[R + 1] - CW[L] - W[i]
    n = float(np.linalg.norm(S))
    if n <= 1e-12:
        return None
    return S / n


def _collect_occurrences_for_doc(
    DA: Dict[str, Any],
    seeds_set: set[str],
    token_window: int,
) -> Optional[Dict[str, Any]]:
    """Return per-doc occurrences with their prebuilt snippet metadata and occ vectors."""
    if not DA or DA.get("N", 0) == 0:
        return None
    toks = DA["toks"]
    idxs = [i for i, t in enumerate(toks) if t in seeds_set]
    if not idxs:
        return None

    occ_vecs = []
    meta = []  # tuples for later row construction
    for i in idxs:
        L, R = i - token_window, i + token_window
        v = _occ_vec(DA["CW"], DA["W"], i, L, R)
        if v is None:
            continue
        snippet_anchor, s_min, s_max = _make_snippet_anchor(
            _DL_proxy(DA), i, max(0, L), min(len(toks) - 1, R)
        )
        occ_vecs.append(v)
        meta.append((i, toks[i], L, R, s_min, s_max, snippet_anchor))
    if not occ_vecs:
        return None

    occ_mat = np.vstack(occ_vecs)  # (m, d)
    essay_surface = " ".join(DA["sents_surface"])
    essay_lemmas = " ".join(toks)

    return dict(
        profile_id=DA["profile_id"],
        post_id=DA["post_id"],
        occ_mat=occ_mat,
        meta=meta,
        essay_surface=essay_surface,
        essay_lemmas=essay_lemmas,
    )


# small helper to reuse the existing _make_snippet_anchor signature without copying arrays around
class _DL_proxy(_DocLike):
    def __init__(self, DA: Dict[str, Any]) -> None:
        self.profile_id = DA["profile_id"]
        self.post_id = DA["post_id"]
        self.sents_surface = DA["sents_surface"]
        self.doc_lemmas = DA["toks"]
        self.token_to_sent = DA["token_to_sent"]


# ---------- parallel driver ----------
def _precompute_all_docs(
    pre_docs: List[PreprocessedDoc],
    kv,
    sif_a,
    global_wc,
    total_tokens,
    n_jobs: int,
) -> List[Dict[str, Any]]:
    """
    Precompute per-doc arrays in parallel.
    Returns list of per-doc dicts with precomputed arrays.
    """
    doclikes = list(_iter_doclikes(pre_docs))
    # Threaded precompute (BLAS inside get_vector won't block; read-only kv is safe)
    results: List[Optional[Dict[str, Any]]] = [None] * len(doclikes)
    with ThreadPoolExecutor(
        max_workers=(None if n_jobs in (-1, 0, None) else int(n_jobs))
    ) as ex:
        futs = {
            ex.submit(
                _precompute_doc_arrays, kv, D, sif_a, global_wc, total_tokens
            ): idx
            for idx, D in enumerate(doclikes)
        }
        for f in as_completed(futs):
            idx = futs[f]
            results[idx] = f.result()
    return [r for r in results if r is not None and r.get("N", 0) > 0]


def _collect_all_occurrences(
    doc_arrays: List[Dict[str, Any]],
    seeds_set: set[str],
    token_window: int,
    n_jobs: int,
    progress: bool = False,
    desc: str = "Snippets: occurrences",
) -> List[Dict[str, Any]]:
    """
    Collect all occurrences from all docs based on seeds_set and token_window.
    """
    try:
        from tqdm.auto import tqdm as _tqdm
    except Exception:
        _tqdm = None
    iterator = doc_arrays
    if progress and _tqdm is not None and sys.stderr is not None:
        iterator = _tqdm(iterator, total=len(doc_arrays), desc=desc)

    out: List[Dict[str, Any]] = []
    # Threading again (occ computations are mostly BLAS on arrays)
    with ThreadPoolExecutor(
        max_workers=(None if n_jobs in (-1, 0, None) else int(n_jobs))
    ) as ex:
        futs = [
            ex.submit(_collect_occurrences_for_doc, DA, seeds_set, token_window)
            for DA in iterator
        ]
        for f in as_completed(futs):
            res = f.result()
            if res is not None:
                out.append(res)
    return out


def _collect_sentence_occurrences_for_doc(
    DA: Dict[str, Any],
    token_window: int,
) -> Optional[Dict[str, Any]]:
    """
    Fallback mode when we have *no seeds* (no lexicon).
    Treat each sentence as a single 'occurrence':
      - sentence vector = SIF-weighted mean of all its tokens
      - snippet_anchor  = that sentence
    """
    if not DA or DA.get("N", 0) == 0:
        return None

    toks = DA["toks"]
    N = len(toks)
    if N == 0:
        return None

    W = DA["W"]  # (N, d), already SIF-weighted
    token_to_sent = DA["token_to_sent"]
    sents_surface = DA["sents_surface"]

    if not token_to_sent or not sents_surface:
        return None

    # gather token indices per sentence
    n_sents = max(token_to_sent) + 1 if token_to_sent else 0
    sent_to_indices: List[List[int]] = [[] for _ in range(n_sents)]
    for i, s_idx in enumerate(token_to_sent):
        if 0 <= s_idx < n_sents:
            sent_to_indices[s_idx].append(i)

    occ_vecs = []
    meta = []

    for s_idx, idxs in enumerate(sent_to_indices):
        if not idxs:
            continue

        # sum SIF-weighted vectors for tokens in this sentence
        sum_vec = W[idxs].sum(axis=0)
        n = float(np.linalg.norm(sum_vec))
        if n <= 1e-12:
            continue

        v = sum_vec / n

        start_tok = min(idxs)
        end_tok = max(idxs)
        snippet = sents_surface[s_idx] if s_idx < len(sents_surface) else ""

        # meta fields mirror the seed-based variant:
        # (i, seed, L, R, s_min, s_max, snippet)
        occ_vecs.append(v)
        meta.append((start_tok, "<SENT>", start_tok, end_tok, s_idx, s_idx, snippet))

    if not occ_vecs:
        return None

    occ_mat = np.vstack(occ_vecs)
    essay_surface = " ".join(sents_surface)
    essay_lemmas = " ".join(toks)

    return dict(
        profile_id=DA["profile_id"],
        post_id=DA["post_id"],
        occ_mat=occ_mat,
        meta=meta,
        essay_surface=essay_surface,
        essay_lemmas=essay_lemmas,
    )


def _collect_sentence_occurrences(
    doc_arrays: List[Dict[str, Any]],
    token_window: int,
    n_jobs: int,
    progress: bool = False,
    desc: str = "Snippets: sentences",
) -> List[Dict[str, Any]]:
    """
    Fallback collector when seeds_set is empty:
    one occurrence per sentence per doc.
    """
    try:
        from tqdm.auto import tqdm as _tqdm
    except Exception:
        _tqdm = None

    iterator = doc_arrays
    if progress and _tqdm is not None and sys.stderr is not None:
        iterator = _tqdm(iterator, total=len(doc_arrays), desc=desc)

    out: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(
        max_workers=(None if n_jobs in (-1, 0, None) else int(n_jobs))
    ) as ex:
        futs = [
            ex.submit(_collect_sentence_occurrences_for_doc, DA, token_window)
            for DA in iterator
        ]
        for f in as_completed(futs):
            res = f.result()
            if res is not None:
                out.append(res)
    return out


#####################
def _collect_doc_occurrences_for_doc(
    DA: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Fallback mode when we want *one occurrence per whole text*.

    Vector = SIF-weighted mean over ALL tokens in the doc.
    Snippet = concatenation of all sentences (the whole text).
    """
    if not DA or DA.get("N", 0) == 0:
        return None

    toks = DA["toks"]
    if not toks:
        return None

    W = DA["W"]  # (N, d), SIF-weighted token vectors
    sum_vec = W.sum(axis=0)
    n = float(np.linalg.norm(sum_vec))
    if n <= 1e-12:
        return None

    v = sum_vec / n

    start_tok = 0
    end_tok = len(toks) - 1
    sents_surface = DA["sents_surface"]
    snippet = " ".join(sents_surface) if sents_surface else ""

    # meta format compatible with seed/sentence modes:
    # (i, seed, L, R, s_min, s_max, snippet)
    occ_mat = v.reshape(1, -1)
    meta = [
        (
            0,  # i (we can treat as first token)
            "<DOC>",  # seed placeholder
            start_tok,
            end_tok,
            0,  # start_sent_idx
            max(0, len(sents_surface) - 1),  # end_sent_idx
            snippet,
        )
    ]

    essay_surface = snippet
    essay_lemmas = " ".join(toks)

    return dict(
        profile_id=DA["profile_id"],
        post_id=DA["post_id"],
        occ_mat=occ_mat,
        meta=meta,
        essay_surface=essay_surface,
        essay_lemmas=essay_lemmas,
    )


def _collect_doc_occurrences(
    doc_arrays: List[Dict[str, Any]],
    n_jobs: int,
    progress: bool = False,
    desc: str = "Snippets: docs",
) -> List[Dict[str, Any]]:
    """
    One occurrence per doc/post, based on *whole text*.
    """
    try:
        from tqdm.auto import tqdm as _tqdm
    except Exception:
        _tqdm = None

    iterator = doc_arrays
    if progress and _tqdm is not None and sys.stderr is not None:
        iterator = _tqdm(iterator, total=len(doc_arrays), desc=desc)

    out: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(
        max_workers=(None if n_jobs in (-1, 0, None) else int(n_jobs))
    ) as ex:
        futs = [ex.submit(_collect_doc_occurrences_for_doc, DA) for DA in iterator]
        for f in as_completed(futs):
            res = f.result()
            if res is not None:
                out.append(res)
    return out


############################


# ---------- sorting / top-k helpers (replacing pandas) ----------
def _sort_rows_by_cosine_desc(rows: list[dict]) -> list[dict]:
    """Sort list of dicts by 'cosine' key, descending."""
    return sorted(rows, key=lambda r: r["cosine"], reverse=True)


def _sort_rows_by_label_and_cosine(rows: list[dict]) -> list[dict]:
    """Sort by centroid_label ascending, then cosine descending."""
    return sorted(rows, key=lambda r: (r["centroid_label"], -r["cosine"]))


def _top_per_group(rows: list[dict], group_key: str, n: int) -> list[dict]:
    """Return the top-n rows per group (rows must already be sorted by cosine desc within groups)."""
    counts: dict[str, int] = {}
    out: list[dict] = []
    for r in rows:
        g = r[group_key]
        counts[g] = counts.get(g, 0) + 1
        if counts[g] <= n:
            out.append(r)
    return out


# ---------- public API ----------
def cluster_snippets_by_centroids(
    *,
    pre_docs: List[PreprocessedDoc],
    ssd,  # fitted SSD (must expose kv and lexicon)
    pos_clusters: List[dict] | None,  # clusters from +beta
    neg_clusters: List[dict] | None,  # clusters from -beta
    token_window: int = 3,
    seeds: Iterable[str] | None = None,
    sif_a: float = 1e-3,
    global_wc: dict[str, int] | None = None,
    total_tokens: int | None = None,
    top_per_cluster: int = 100,
    n_jobs: int = -1,
    progress: bool = True,
) -> dict[str, list[dict]]:
    """
    Extract snippets scored against cluster centroids from fitted SSD model.

    Returns dict with 'pos' and 'neg' keys, each a list of dicts.
    Each dict has keys:
        centroid_label, profile_id, post_id, cosine, seed,
        start_token_idx, end_token_idx, start_sent_idx, end_sent_idx,
        snippet_anchor, essay_text_surface, essay_text_lemmas

    Parameters
    ----------
    pre_docs : List[PreprocessedDoc]
        Preprocessed documents to extract snippets from.
    ssd : fitted SSD model
        Duck-typed object exposing .kv (Embeddings), .beta_unit, .beta,
        .use_unit_beta, .lexicon, .window, .sif_a.
    pos_clusters : List[dict] | None
        List of clusters from +beta side (each cluster is a dict with 'words' key).
    neg_clusters : List[dict] | None
        List of clusters from -beta side (each cluster is a dict with 'words' key).
    token_window : int, optional
        Token window size around seed words for occurrence context (default is 3).
    seeds : Iterable[str] | None, optional
        Iterable of seed words to look for in documents. If None, uses ssd.lexicon.
    sif_a : float, optional
        SIF weighting parameter (default is 1e-3).
    global_wc : dict[str, int] | None, optional
        Global word counts. If None, will be computed from pre_docs.
    total_tokens : int | None, optional
        Total token count. If None, will be computed from pre_docs.
    top_per_cluster : int, optional
        Number of top snippets to return per cluster (default is 100).
    n_jobs : int, optional
        Number of parallel jobs to use (default is -1, meaning all available cores).
    progress : bool, optional
        Whether to display a progress bar (default is True).

    Returns
    -------
    dict[str, list[dict]]
        Dictionary with 'pos' and 'neg' lists of snippet dicts.
    """

    if global_wc is None or total_tokens is None:
        global_wc, total_tokens = _build_global_sif(pre_docs)

    kv = ssd.kv
    seeds_set = set(seeds or getattr(ssd, "lexicon", []))

    # targets (cluster centroids)
    targets: List[np.ndarray] = []
    labels: List[str] = []

    def _add_side(clusters: Optional[List[dict]], side: str):
        if not clusters:
            return
        for rank, C in enumerate(clusters, start=1):
            uC = _centroid_unit_from_cluster_words(C["words"], kv)
            if uC.shape[0] and np.any(uC):
                targets.append(uC)
                labels.append(f"{side}_cluster_{rank}")

    _add_side(pos_clusters, "pos")
    _add_side(neg_clusters, "neg")
    if not targets:
        return {"pos": [], "neg": []}

    T = np.vstack(targets)  # (K, d)

    # 1) precompute per-doc arrays once (parallel)
    doc_arrays = _precompute_all_docs(
        pre_docs, kv, sif_a, global_wc, total_tokens, n_jobs
    )

    # 2) collect occurrences:
    #    - if we have seeds -> seed-based windows (old behavior)
    #    - if no seeds      -> fallback to *whole-text* occurrences
    if seeds_set:
        occs = _collect_all_occurrences(
            doc_arrays,
            seeds_set,
            token_window,
            n_jobs,
            progress,
        )
    else:
        occs = _collect_doc_occurrences(
            doc_arrays,
            n_jobs,
            progress,
            desc="Snippets: docs",
        )

    # 3) score all occurrences against all targets in one BLAS call
    rows: list[dict] = []
    for occ in occs:
        Omat = occ["occ_mat"]  # (m, d)
        C = Omat @ T.T  # (m, K)
        for r, (i, seed, L, R, smin, smax, snippet) in enumerate(occ["meta"]):
            cos_row = C[r]  # (K,)
            for k, lab in enumerate(labels):
                rows.append(
                    dict(
                        centroid_label=lab,
                        profile_id=occ["profile_id"],
                        post_id=occ["post_id"],
                        cosine=float(cos_row[k]),
                        seed=seed,
                        start_token_idx=max(0, L),
                        end_token_idx=max(0, R),
                        start_sent_idx=smin,
                        end_sent_idx=smax,
                        snippet_anchor=snippet,
                        essay_text_surface=occ["essay_surface"],
                        essay_text_lemmas=occ["essay_lemmas"],
                    )
                )

    if not rows:
        return {"pos": [], "neg": []}

    # sort by centroid_label asc, cosine desc — then take top_per_cluster per group
    rows = _sort_rows_by_label_and_cosine(rows)
    rows = _top_per_group(rows, "centroid_label", top_per_cluster)

    pos_rows = [r for r in rows if r["centroid_label"].startswith("pos_")]
    neg_rows = [r for r in rows if not r["centroid_label"].startswith("pos_")]
    return {"pos": pos_rows, "neg": neg_rows}


def snippets_along_beta(
    *,
    pre_docs: List[PreprocessedDoc],
    ssd,  # fitted SSD (must expose beta_unit and kv)
    token_window: int = 3,
    seeds: Iterable[str] | None = None,
    sif_a: float = 1e-3,
    global_wc: dict[str, int] | None = None,
    total_tokens: int | None = None,
    top_per_side: int = 200,
    min_cosine: float | None = None,
    n_jobs: int = -1,
    progress: bool = True,
) -> dict[str, list[dict]]:
    """
    Extract snippets scored along +/-beta from fitted SSD model.

    Returns dict with 'pos' and 'neg' keys, each a list of dicts.
    Each dict has keys:
        side_label, profile_id, post_id, cosine, seed,
        start_token_idx, end_token_idx, start_sent_idx, end_sent_idx,
        snippet_anchor, essay_text_surface, essay_text_lemmas

    Parameters
    ----------
    pre_docs : List[PreprocessedDoc]
        Preprocessed documents to extract snippets from.
    ssd : fitted SSD model
        Duck-typed object exposing .kv (Embeddings), .beta_unit, .beta,
        .use_unit_beta, .lexicon, .window, .sif_a.
    token_window : int, optional
        Token window size around seed occurrences, by default 3.
    seeds : Iterable[str] | None, optional
        Iterable of seed lemmas to focus on. If None, uses ssd.lexicon.
    sif_a : float, optional
        SIF weighting parameter, by default 1e-3.
    global_wc : dict[str, int] | None, optional
        Global word counts. If None, computed from pre_docs.
    total_tokens : int | None, optional
        Total token count. If None, computed from pre_docs.
    top_per_side : int, optional
        Number of top snippets to return per side (positive/negative), by default 200.
    min_cosine : float | None, optional
        Minimum cosine threshold to include a snippet. If None, no thresholding.
    n_jobs : int, optional
        Number of parallel jobs, by default -1 (all available).
    progress : bool, optional
        Whether to show progress bars, by default True.

    Returns
    -------
    dict[str, list[dict]]
        Dictionary with 'pos' and 'neg' lists of snippet dicts.
    """

    if global_wc is None or total_tokens is None:
        global_wc, total_tokens = _build_global_sif(pre_docs)

    kv = ssd.kv
    b_unit = unit_vector(getattr(ssd, "beta_unit", getattr(ssd, "beta")))
    seeds_set = set(seeds or getattr(ssd, "lexicon", []))

    # 1) precompute per-doc arrays once (parallel)
    doc_arrays = _precompute_all_docs(
        pre_docs, kv, sif_a, global_wc, total_tokens, n_jobs
    )

    # 2) collect occurrences:
    #    - seed-based if seeds_set not empty
    #    - sentence-based fallback if no seeds
    if seeds_set:
        occs = _collect_all_occurrences(
            doc_arrays, seeds_set, token_window, n_jobs, progress
        )
    else:
        occs = _collect_sentence_occurrences(doc_arrays, token_window, n_jobs, progress)

    # 3) score against +/-beta in one go
    rows_pos: list[dict] = []
    rows_neg: list[dict] = []
    for occ in occs:
        Omat = occ["occ_mat"]  # (m, d)
        cos_pos = Omat @ b_unit
        cos_neg = -cos_pos

        for r, (i, seed, L, R, smin, smax, snippet) in enumerate(occ["meta"]):
            cp = float(cos_pos[r])
            cn = float(cos_neg[r])

            if (min_cosine is None) or (cp >= min_cosine):
                rows_pos.append(
                    dict(
                        side_label="beta_pos",
                        profile_id=occ["profile_id"],
                        post_id=occ["post_id"],
                        cosine=cp,
                        seed=seed,
                        start_token_idx=max(0, L),
                        end_token_idx=max(0, R),
                        start_sent_idx=smin,
                        end_sent_idx=smax,
                        snippet_anchor=snippet,
                        essay_text_surface=occ["essay_surface"],
                        essay_text_lemmas=occ["essay_lemmas"],
                    )
                )
            if (min_cosine is None) or (cn >= min_cosine):
                rows_neg.append(
                    dict(
                        side_label="beta_neg",
                        profile_id=occ["profile_id"],
                        post_id=occ["post_id"],
                        cosine=cn,
                        seed=seed,
                        start_token_idx=max(0, L),
                        end_token_idx=max(0, R),
                        start_sent_idx=smin,
                        end_sent_idx=smax,
                        snippet_anchor=snippet,
                        essay_text_surface=occ["essay_surface"],
                        essay_text_lemmas=occ["essay_lemmas"],
                    )
                )

    def _finalize(rows: list[dict]) -> list[dict]:
        if not rows:
            return []
        rows = _sort_rows_by_cosine_desc(rows)
        if top_per_side is not None:
            rows = rows[:top_per_side]
        return rows

    return {"pos": _finalize(rows_pos), "neg": _finalize(rows_neg)}
