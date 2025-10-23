import copy
import numpy as np
from gensim.models import KeyedVectors
from typing import List, Tuple, Dict, Set, Union
import re

_bad_token = re.compile(r".*\d|^[A-ZĄĆĘŁŃÓŚŹŻ]")

def normalize_kv(
    kv: KeyedVectors,
    *,
    l2: bool = True,
    abtt_m: int = 0,
    re_normalize: bool = True,
) -> KeyedVectors:
    """
    Return a NEW KeyedVectors with optional:
      1) L2 normalization of rows
      2) ABTT: center & remove top-m PCs
      3) re-normalize rows (recommended)
    """
    keys = list(kv.index_to_key)
    V = kv.get_normed_vectors().astype(np.float64) if l2 else kv.vectors.astype(np.float64)

    if abtt_m > 0:
        mu = V.mean(axis=0)
        Vc = V - mu
        # SVD on feature space
        U, S, Vt = np.linalg.svd(Vc, full_matrices=False)
        m = min(abtt_m, Vt.shape[0])
        top = Vt[:m, :]                 # (m, d)
        P = np.eye(Vt.shape[1]) - top.T @ top
        V = Vc @ P

    if re_normalize:
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        V = V / norms

    kv_t = KeyedVectors(vector_size=V.shape[1])
    kv_t.add_vectors(keys, V.astype(np.float32))
    kv_t.fill_norms()
    return kv_t

def compute_global_sif(sentences: List[List[str]]) -> Tuple[Dict[str, int], int]:
    wc: Dict[str, int] = {}
    for sent in sentences:
        for t in sent:
            wc[t] = wc.get(t, 0) + 1
    return wc, sum(wc.values())


def build_doc_vectors(
    docs, kv, lexicon, global_wc, total_tokens, window, sif_a
):
    X_list = []
    keep_mask = []
    for doc in docs:
        v = _doc_vector(doc, kv, lexicon, global_wc, total_tokens, window, sif_a)
        if v is None:
            keep_mask.append(False)
        else:
            keep_mask.append(True)
            X_list.append(v)
    X = np.vstack(X_list) if X_list else np.zeros((0, kv.vector_size), dtype=np.float64)
    return X, np.array(keep_mask, dtype=bool)


def _doc_vector(
    doc, kv, lexicon, wc, tot, window, sif_a
) -> np.ndarray | None:
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
            a = sif_a / (sif_a + wc.get(c, 0) / tot)
            sum_v += a * kv[c]
            w_sum += a
        if w_sum > 0:
            occ.append(sum_v / w_sum)

    if not occ:
        return None

    return np.mean(occ, axis=0).astype(np.float64)


def filtered_neighbors(
    kv: KeyedVectors,
    vec: Union[List[float], np.ndarray],
    topn: int = 20,
    cand: int = 2000,
    restrict: int = 10000,
):
    nbrs = kv.similar_by_vector(vec, topn=cand, restrict_vocab=restrict)
    out = []
    for w, sim in nbrs:
        if not _bad_token.match(w):
            out.append((w, sim))
            if len(out) >= topn:
                break
    return out

# utils/embeddings.py
import os
import gzip
from gensim.models import KeyedVectors

def _first_line_tokens(path: str) -> list[str]:
    opener = gzip.open if path.lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        return f.readline().strip().split()

def load_embeddings(path: str) -> KeyedVectors:
    """
    Load a pre-trained static embedding model from the given path.
    Supports:
      - .kv (Gensim KeyedVectors)
      - .bin (word2vec binary)
      - .txt/.vec (word2vec text; auto-detects header; works with many fastText/GloVe-style dumps)
      - optional .gz compression on the above
    """
    low = path.lower()
    ext = os.path.splitext(low)[1]

    # Gensim native format
    if ext == ".kv" or low.endswith(".kv.gz"):
        return KeyedVectors.load(path, mmap="r")

    # word2vec binary
    if ext == ".bin" or low.endswith(".bin.gz"):
        return KeyedVectors.load_word2vec_format(
            path, binary=True, unicode_errors="ignore"
        )

    # word2vec text / fastText .vec / GloVe-like (auto header detection)
    if ext in {".txt", ".vec"} or low.endswith(".txt.gz") or low.endswith(".vec.gz"):
        toks = _first_line_tokens(path)
        has_header = (len(toks) == 2 and toks[0].isdigit() and toks[1].isdigit())
        return KeyedVectors.load_word2vec_format(
            path,
            binary=False,
            unicode_errors="ignore",
            no_header=not has_header
        )

    # Fallback: try Gensim loader (covers rare cases)
    return KeyedVectors.load(path, mmap="r")

