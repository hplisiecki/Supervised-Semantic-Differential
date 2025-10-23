# ===== ssdiff/snippets.py (replace cluster_snippets_by_centroids) =====
from __future__ import annotations
from typing import List, Iterable
import numpy as np
import pandas as pd

from .preprocess import PreprocessedDoc



def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / max(n, eps)

def _centroid_unit_from_cluster_words(words: list[tuple], kv) -> np.ndarray:
    """
    words: [(word, cos_to_centroid, cos_to_beta), ...]
    returns unit centroid of unit word vectors; zeros if empty.
    """
    vecs = []
    for w, *_ in words:
        if w in kv:
            vecs.append(kv.get_vector(w, norm=True))
    if not vecs:
        return np.zeros(kv.vector_size, dtype=np.float64)
    c = np.mean(np.vstack(vecs), axis=0)
    return _unit(c)

def cluster_snippets_by_centroids(
    *,
    pre_docs: List[PreprocessedDoc],
    ssd,                                # fitted ssd
    pos_clusters: List[dict] | None,   # clusters from +β̂
    neg_clusters: List[dict] | None,   # clusters from −β̂
    window_sentences: int = 1,         # take [sent-1, sent, sent+1]
    seeds: Iterable[str] | None = None,
    sif_a: float = 1e-3,
    global_wc: dict[str,int] | None = None,
    total_tokens: int | None = None,
    top_per_cluster: int = 100,
) -> dict[str, pd.DataFrame]:
    """
    For each cluster (positive and/or negative), find occurrences of any seed lemma,
    compute a SIF-weighted *unit* context vector around the occurrence (±3 tokens),
    cosine it with the cluster centroid (unit), and collect the original sentences.

    Returns: {"pos": df_pos, "neg": df_neg} with columns:
        centroid_label, doc_id, cosine, seed, sent_idx_min, sent_idx_max,
        sentence_before, sentence_anchor, sentence_after,
        window_text_surface, window_text_lemmas
    """
    # Build global SIF stats if not provided (over lemma stream)
    if global_wc is None or total_tokens is None:
        wc = {}
        tot = 0
        for P in pre_docs:
            for lem in P.doc_lemmas:
                wc[lem] = wc.get(lem, 0) + 1
                tot += 1
        global_wc, total_tokens = wc, tot

    kv = ssd.kv
    seeds = set(seeds or getattr(ssd, "lexicon", []))

    def score_side(clusters: List[dict] | None, side_label: str) -> pd.DataFrame:
        if not clusters:
            return pd.DataFrame(columns=[
                "centroid_label","doc_id","cosine","seed","sent_idx_min","sent_idx_max",
                "sentence_before","sentence_anchor","sentence_after",
                "window_text_surface","window_text_lemmas"
            ])

        rows = []
        for rank, C in enumerate(clusters, start=1):
            uC = _centroid_unit_from_cluster_words(C["words"], kv)
            if uC.shape[0] == 0 or not np.any(uC):
                continue
            label = f"{side_label}_cluster_{rank}"

            for doc_id, P in enumerate(pre_docs):
                lemmas = P.doc_lemmas
                # indices of any seed occurrence
                idxs = [i for i, t in enumerate(lemmas) if t in seeds]
                if not idxs:
                    continue

                for i in idxs:
                    # SIF-weighted context around this seed (±3 tokens), excluding the seed itself
                    start = max(0, i - 3)
                    end   = min(len(lemmas), i + 3 + 1)
                    sum_v = np.zeros(kv.vector_size, dtype=np.float64)
                    w_sum = 0.0
                    for j in range(start, end):
                        if j == i:
                            continue
                        w = lemmas[j]
                        if w not in kv:
                            continue
                        a = sif_a / (sif_a + global_wc.get(w, 0) / total_tokens)
                        sum_v += a * kv.get_vector(w, norm=True)
                        w_sum += a
                    if w_sum <= 0:
                        continue

                    # UNIT normalize the occurrence context to get a true cosine
                    occ_vec = _unit(sum_v / w_sum)
                    cos = float(occ_vec @ uC)

                    # sentence window from SURFACE text
                    if i >= len(P.token_to_sent):
                        continue
                    s_idx = P.token_to_sent[i]
                    s_min = max(0, s_idx - window_sentences)
                    s_max = min(len(P.sents_surface) - 1, s_idx + window_sentences)

                    sent_before = P.sents_surface[s_idx-1] if s_idx-1 >= 0 else ""
                    sent_anchor = P.sents_surface[s_idx]
                    sent_after  = P.sents_surface[s_idx+1] if (s_idx+1) < len(P.sents_surface) else ""

                    window_surface = " ".join(P.sents_surface[s_min:s_max+1])
                    window_lemmas  = " || ".join(" ".join(P.sents_lemmas[k]) for k in range(s_min, s_max+1))

                    rows.append(dict(
                        centroid_label=label,
                        doc_id=doc_id,
                        cosine=cos,
                        seed=lemmas[i],
                        sent_idx_min=s_min,
                        sent_idx_max=s_max,
                        sentence_before=sent_before,
                        sentence_anchor=sent_anchor,
                        sentence_after=sent_after,
                        window_text_surface=window_surface,
                        window_text_lemmas=window_lemmas,
                    ))

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        # Sort: for both sides, higher cosine means closer to the cluster centroid
        df = df.sort_values(["centroid_label", "cosine"], ascending=[True, False]).reset_index(drop=True)

        # Keep top-K per cluster label
        df = (df.groupby("centroid_label", group_keys=False)
                .head(top_per_cluster)
                .reset_index(drop=True))
        return df

    return {
        "pos": score_side(pos_clusters, "pos"),
        "neg": score_side(neg_clusters, "neg"),
    }


def snippets_along_beta(
    *,
    pre_docs: List[PreprocessedDoc],
    ssd,                                # fitted ssd (must expose beta_unit and kv)
    window_sentences: int = 1,         # take [sent-1, sent, sent+1]
    seeds: Iterable[str] | None = None,
    sif_a: float = 1e-3,
    global_wc: dict[str,int] | None = None,
    total_tokens: int | None = None,
    top_per_side: int = 200,           # how many snippets to keep per side
    min_cosine: float | None = None,   # optional cosine floor (e.g., 0.15)
) -> dict[str, pd.DataFrame]:
    """
    Find SIF-weighted context snippets for each seed occurrence and score them by
    cosine to +β̂ and −β̂ (unit). Returns two DataFrames: 'beta_pos' and 'beta_neg'.

    Columns:
      side_label, doc_id, cosine, seed, sent_idx_min, sent_idx_max,
      sentence_before, sentence_anchor, sentence_after,
      window_text_surface, window_text_lemmas
    """
    # Global SIF stats if not provided
    if global_wc is None or total_tokens is None:
        wc = {}
        tot = 0
        for P in pre_docs:
            for lem in P.doc_lemmas:
                wc[lem] = wc.get(lem, 0) + 1
                tot += 1
        global_wc, total_tokens = wc, tot

    kv = ssd.kv
    b_unit = _unit(getattr(ssd, "beta_unit", getattr(ssd, "beta")))
    seeds = set(seeds or getattr(ssd, "lexicon", []))

    def score_side(target_vec: np.ndarray, side_label: str) -> pd.DataFrame:
        rows = []
        for doc_id, P in enumerate(pre_docs):
            lemmas = P.doc_lemmas
            idxs = [i for i, t in enumerate(lemmas) if t in seeds]
            if not idxs:
                continue

            for i in idxs:
                # SIF-weighted context around seed (±3 tokens), exclude the seed token
                start = max(0, i - 3)
                end   = min(len(lemmas), i + 3 + 1)
                sum_v = np.zeros(kv.vector_size, dtype=np.float64)
                w_sum = 0.0
                for j in range(start, end):
                    if j == i:
                        continue
                    w = lemmas[j]
                    if w not in kv:
                        continue
                    a = sif_a / (sif_a + global_wc.get(w, 0) / total_tokens)
                    sum_v += a * kv.get_vector(w, norm=True)
                    w_sum += a
                if w_sum <= 0:
                    continue

                occ_vec = _unit(sum_v / w_sum)
                cos = float(occ_vec @ target_vec)

                # optional floor
                if (min_cosine is not None) and (cos < min_cosine):
                    continue

                # sentence window in SURFACE text
                if i >= len(P.token_to_sent):
                    continue
                s_idx = P.token_to_sent[i]
                s_min = max(0, s_idx - window_sentences)
                s_max = min(len(P.sents_surface) - 1, s_idx + window_sentences)

                sent_before = P.sents_surface[s_idx-1] if s_idx-1 >= 0 else ""
                sent_anchor = P.sents_surface[s_idx]
                sent_after  = P.sents_surface[s_idx+1] if (s_idx+1) < len(P.sents_surface) else ""

                window_surface = " ".join(P.sents_surface[s_min:s_max+1])
                window_lemmas  = " || ".join(" ".join(P.sents_lemmas[k]) for k in range(s_min, s_max+1))

                rows.append(dict(
                    side_label=side_label,
                    doc_id=doc_id,
                    cosine=cos,
                    seed=lemmas[i],
                    sent_idx_min=s_min,
                    sent_idx_max=s_max,
                    sentence_before=sent_before,
                    sentence_anchor=sent_anchor,
                    sentence_after=sent_after,
                    window_text_surface=window_surface,
                    window_text_lemmas=window_lemmas,
                ))

        if not rows:
            return pd.DataFrame(columns=[
                "side_label","doc_id","cosine","seed","sent_idx_min","sent_idx_max",
                "sentence_before","sentence_anchor","sentence_after",
                "window_text_surface","window_text_lemmas"
            ])

        df = pd.DataFrame(rows)
        # Sort by cosine desc (strongest alignment first)
        df = df.sort_values(["cosine"], ascending=[False]).reset_index(drop=True)
        # Keep top-K
        if top_per_side is not None:
            df = df.head(top_per_side).reset_index(drop=True)
        return df

    df_pos = score_side(b_unit, "beta_pos")
    df_neg = score_side(-b_unit, "beta_neg")

    return {"beta_pos": df_pos, "beta_neg": df_neg}