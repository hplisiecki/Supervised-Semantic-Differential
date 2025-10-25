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
    ssd,                                # fitted SSD (must expose kv and lexicon)
    pos_clusters: List[dict] | None,    # clusters from +β̂ (raw list-of-dicts, not DF)
    neg_clusters: List[dict] | None,    # clusters from −β̂ (raw list-of-dicts, not DF)
    token_window: int = 3,              # ±token_window around seed
    seeds: Iterable[str] | None = None,
    sif_a: float = 1e-3,
    global_wc: dict[str, int] | None = None,
    total_tokens: int | None = None,
    top_per_cluster: int = 100,
) -> dict[str, pd.DataFrame]:
    """
    For each cluster (positive and/or negative), find occurrences of any seed lemma,
    compute a SIF-weighted *unit* context vector around the occurrence (±token_window tokens),
    cosine it with the cluster centroid (unit), and collect an anchor snippet:

      - If the token window is fully inside the anchor sentence → snippet_anchor = anchor sentence.
      - If it crosses a sentence boundary → snippet_anchor = two sentences (prev+anchor or anchor+next).

    Returns: {"pos": df_pos, "neg": df_neg} with columns:
      centroid_label, doc_id, cosine, seed,
      start_token_idx, end_token_idx, start_sent_idx, end_sent_idx,
      snippet_anchor, essay_text_surface, essay_text_lemmas
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

    def make_snippet_anchor(P: PreprocessedDoc, i: int, start_tok: int, end_tok: int) -> tuple[str, int, int]:
        """
        Build snippet_anchor per the rule. Returns (snippet_anchor, sent_idx_min, sent_idx_max).
        """
        # sentence index of the seed
        s_idx = P.token_to_sent[i] if i < len(P.token_to_sent) else 0

        # sentence index of token window edges (clip to valid token range)
        start_tok = max(0, min(start_tok, len(P.doc_lemmas) - 1))
        end_tok   = max(0, min(end_tok,   len(P.doc_lemmas) - 1))

        start_sent = P.token_to_sent[start_tok] if start_tok < len(P.token_to_sent) else s_idx
        end_sent   = P.token_to_sent[end_tok]   if end_tok   < len(P.token_to_sent) else s_idx

        # fully within anchor sentence
        if start_sent == s_idx and end_sent == s_idx:
            snippet = P.sents_surface[s_idx]
            return snippet, s_idx, s_idx

        # crosses into previous sentence?
        if start_sent < s_idx:
            prev_idx = s_idx - 1
            if prev_idx >= 0:
                snippet = (P.sents_surface[prev_idx] + " " + P.sents_surface[s_idx]).strip()
                return snippet, prev_idx, s_idx

        # otherwise crosses into next sentence (or fallback)
        next_idx = s_idx + 1
        if next_idx < len(P.sents_surface):
            snippet = (P.sents_surface[s_idx] + " " + P.sents_surface[next_idx]).strip()
            return snippet, s_idx, next_idx

        # fallback: just the anchor sentence
        return P.sents_surface[s_idx], s_idx, s_idx

    def score_side(clusters: List[dict] | None, side_label: str) -> pd.DataFrame:
        if not clusters:
            return pd.DataFrame(columns=[
                "centroid_label","doc_id","cosine","seed",
                "start_token_idx","end_token_idx","start_sent_idx","end_sent_idx",
                "snippet_anchor","essay_text_surface","essay_text_lemmas"
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

                essay_surface = " ".join(P.sents_surface)
                essay_lemmas  = " ".join(P.doc_lemmas)

                for i in idxs:
                    # SIF-weighted context around this seed (±token_window tokens), excluding the seed itself
                    start = max(0, i - token_window)
                    end   = min(len(lemmas), i + token_window + 1)
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
                    cos = float(occ_vec @ uC)

                    # Build snippet anchor based on whether token window crosses sentence boundary
                    snippet_anchor, s_min, s_max = make_snippet_anchor(P, i, start, end - 1)

                    rows.append(dict(
                        centroid_label=label,
                        doc_id=doc_id,
                        cosine=cos,
                        seed=lemmas[i],
                        start_token_idx=start,
                        end_token_idx=end - 1,
                        start_sent_idx=s_min,
                        end_sent_idx=s_max,
                        snippet_anchor=snippet_anchor,
                        essay_text_surface=essay_surface,
                        essay_text_lemmas=essay_lemmas,
                    ))

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        # Sort: higher cosine → closer to the cluster centroid
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
    ssd,                                # fitted SSD (must expose beta_unit and kv)
    token_window: int = 3,              # ±token_window around seed
    seeds: Iterable[str] | None = None,
    sif_a: float = 1e-3,
    global_wc: dict[str, int] | None = None,
    total_tokens: int | None = None,
    top_per_side: int = 200,            # how many snippets to keep per side
    min_cosine: float | None = None,    # optional cosine floor (e.g., 0.15)
) -> dict[str, pd.DataFrame]:
    """
    For each seed occurrence, compute a SIF-weighted context vector (±token_window tokens),
    cosine it with +β̂ and −β̂ (unit), and collect an anchor snippet:

      - If the token window is fully inside the anchor sentence → snippet_anchor = anchor sentence.
      - If it crosses a sentence boundary → snippet_anchor = two sentences (prev+anchor or anchor+next).

    Returns two DataFrames: 'beta_pos' and 'beta_neg', each with columns:
      side_label, doc_id, cosine, seed,
      start_token_idx, end_token_idx, start_sent_idx, end_sent_idx,
      snippet_anchor, essay_text_surface, essay_text_lemmas
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

    def make_snippet_anchor(P: PreprocessedDoc, i: int, start_tok: int, end_tok: int) -> tuple[str, int, int]:
        s_idx = P.token_to_sent[i] if i < len(P.token_to_sent) else 0
        start_tok = max(0, min(start_tok, len(P.doc_lemmas) - 1))
        end_tok   = max(0, min(end_tok,   len(P.doc_lemmas) - 1))
        start_sent = P.token_to_sent[start_tok] if start_tok < len(P.token_to_sent) else s_idx
        end_sent   = P.token_to_sent[end_tok]   if end_tok   < len(P.token_to_sent) else s_idx

        if start_sent == s_idx and end_sent == s_idx:
            return P.sents_surface[s_idx], s_idx, s_idx

        if start_sent < s_idx:
            prev_idx = s_idx - 1
            if prev_idx >= 0:
                return (P.sents_surface[prev_idx] + " " + P.sents_surface[s_idx]).strip(), prev_idx, s_idx

        next_idx = s_idx + 1
        if next_idx < len(P.sents_surface):
            return (P.sents_surface[s_idx] + " " + P.sents_surface[next_idx]).strip(), s_idx, next_idx

        return P.sents_surface[s_idx], s_idx, s_idx

    def score_side(target_vec: np.ndarray, side_label: str) -> pd.DataFrame:
        rows = []
        for doc_id, P in enumerate(pre_docs):
            lemmas = P.doc_lemmas
            idxs = [i for i, t in enumerate(lemmas) if t in seeds]
            if not idxs:
                continue

            essay_surface = " ".join(P.sents_surface)
            essay_lemmas  = " ".join(P.doc_lemmas)

            for i in idxs:
                # SIF-weighted context around seed (±token_window tokens), excluding the seed token
                start = max(0, i - token_window)
                end   = min(len(lemmas), i + token_window + 1)
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

                if (min_cosine is not None) and (cos < min_cosine):
                    continue

                snippet_anchor, s_min, s_max = make_snippet_anchor(P, i, start, end - 1)

                rows.append(dict(
                    side_label=side_label,
                    doc_id=doc_id,
                    cosine=cos,
                    seed=lemmas[i],
                    start_token_idx=start,
                    end_token_idx=end - 1,
                    start_sent_idx=s_min,
                    end_sent_idx=s_max,
                    snippet_anchor=snippet_anchor,
                    essay_text_surface=essay_surface,
                    essay_text_lemmas=essay_lemmas,
                ))

        if not rows:
            return pd.DataFrame(columns=[
                "side_label","doc_id","cosine","seed",
                "start_token_idx","end_token_idx","start_sent_idx","end_sent_idx",
                "snippet_anchor","essay_text_surface","essay_text_lemmas"
            ])

        df = pd.DataFrame(rows)
        df = df.sort_values(["cosine"], ascending=[False]).reset_index(drop=True)
        if top_per_side is not None:
            df = df.head(top_per_side).reset_index(drop=True)
        return df

    df_pos = score_side(b_unit, "beta_pos")
    df_neg = score_side(-b_unit, "beta_neg")

    return {"beta_pos": df_pos, "beta_neg": df_neg}
