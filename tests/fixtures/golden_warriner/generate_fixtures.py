"""Regenerate the golden-Warriner regression fixtures.

Offline, run-once script — NOT collected by pytest. It builds a small,
committable artifact from in-repo data (``examples/data/``) and records a
golden PLS result. The committed test
(``tests/integration/test_golden_warriner.py``) then runs entirely from these
artifacts: no large embeddings, no spaCy, no network.

Lexical-norm setup (mirrors ``examples/demo_pls.py``): each "doc" is a single
English word, ``y`` = Warriner valence, ``use_full_doc=True`` (no lexicon).
Because the thin embedding is keyed on the exact rated words, the words are
committed as pre-tokenised 1-token docs and looked up directly — no
lemmatisation anywhere.

What it writes into this directory:
  - ``embedding.bin``  — ``N_DOCS`` rated words, GloVe-300 PCA'd to ``DIM`` dims,
                         L2-normalised (word2vec binary).
  - ``corpus.json``    — the words (1-token docs) + valence.
  - ``golden.json``    — frozen PLS result (R², counts, top words).

Crucial invariant: the golden is generated from the embedding **reloaded from
``embedding.bin``** and a ``pretokenized`` corpus — the exact artifacts the
test consumes — so there is zero generator-vs-test divergence.

Inputs (in-repo, relative to the SSDLite root):
  - examples/data/glove_300_en_top20k_ratings.ssdembed
  - examples/data/Ratings_EN.csv

Run from the SSDLite project root:
    python tests/fixtures/golden_warriner/generate_fixtures.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from ssdiff import SSD, Corpus, Embeddings

# --- Configuration ---------------------------------------------------------
N_DOCS = 6000                            # rated words to keep (seeded subsample)
DIM = 50                                 # PCA target dimensionality
SUBSAMPLE_SEED = 2137                    # fixed seed for the word subsample

K = 2                                    # pinned PLS components (deterministic)
N_SPLITS = 20                            # split_nb splits (p-value only; not asserted)
RANDOM_STATE = 42
N_WORDS = 20                             # top words per pole stored in golden

# --- Paths -----------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
SSDLITE_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
EMB_SRC = os.path.join(SSDLITE_ROOT, "examples", "data",
                       "glove_300_en_top20k_ratings.ssdembed")
RATINGS = os.path.join(SSDLITE_ROOT, "examples", "data", "Ratings_EN.csv")

EMB_STEM = os.path.join(HERE, "embedding")      # → embedding.bin
CORPUS_JSON = os.path.join(HERE, "corpus.json")
GOLDEN_JSON = os.path.join(HERE, "golden.json")


def _pca_reduce(X: np.ndarray, dim: int) -> np.ndarray:
    """Center and project rows onto the top-``dim`` principal components."""
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:dim].T


def main() -> None:
    for p in (EMB_SRC, RATINGS):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required input not found: {p}")

    # 1. Rated words present in the source embedding (seeded subsample).
    src = Embeddings.load(EMB_SRC)
    df = pd.read_csv(RATINGS)
    df["word"] = df["word"].astype(str).str.strip()
    df = df[df["word"].apply(lambda w: w in src)].drop_duplicates("word")
    if len(df) > N_DOCS:
        rng = np.random.default_rng(SUBSAMPLE_SEED)
        df = df.iloc[np.sort(rng.choice(len(df), size=N_DOCS, replace=False))]
    words = df["word"].tolist()
    y = df["valence"].to_numpy(dtype=float)
    print(f"[gen] {len(words)} rated words (of {src.vocab_size:,} in source)")

    # 2. Slice → PCA → L2-normalise → save as word2vec binary.
    idx = [src.key_to_index[w] for w in words]
    reduced = _pca_reduce(src.vectors[idx], DIM)
    reduced /= np.linalg.norm(reduced, axis=1, keepdims=True)
    emb = Embeddings(words, reduced.astype(np.float32))
    emb.l2_normalized = True
    emb.save(EMB_STEM, fmt="bin")

    # 3. Reload from disk — golden MUST come from the exact committed artifact.
    emb = Embeddings.load(EMB_STEM + ".bin")
    if not emb.l2_normalized:
        raise RuntimeError("reloaded embedding is not unit-norm; autodetect failed")

    # 4. Persist the corpus (each word is a 1-token doc).
    corpus_data = {
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
        "docs": [[w] for w in words],
        "y": y.tolist(),
    }
    with open(CORPUS_JSON, "w", encoding="utf-8") as f:
        json.dump(corpus_data, f, ensure_ascii=False)

    # 5. Fit and freeze the golden (from reloaded emb + pretokenized corpus).
    corpus = Corpus([[w] for w in words], pretokenized=True, lang="en")
    ssd = SSD(emb, corpus, y, use_full_doc=True)
    res = ssd.fit_pls(k=K, n_splits=N_SPLITS, random_state=RANDOM_STATE)

    golden = {
        "k": K,
        "n_components": int(res.fit_info.n_components),
        "n_raw": int(res.stats.n_raw),
        "n_kept": int(res.stats.n_kept),
        "n_dropped": int(res.stats.n_dropped),
        "r2": float(res.stats.r2),
        "pos_words": [w.word for w in res.words.pos(N_WORDS)],
        "neg_words": [w.word for w in res.words.neg(N_WORDS)],
    }
    with open(GOLDEN_JSON, "w", encoding="utf-8") as f:
        json.dump(golden, f, ensure_ascii=False, indent=2)

    emb_mb = os.path.getsize(EMB_STEM + ".bin") / 1e6
    corpus_mb = os.path.getsize(CORPUS_JSON) / 1e6
    print(f"\n[gen] wrote embedding.bin ({emb_mb:.2f} MB), "
          f"corpus.json ({corpus_mb:.2f} MB), golden.json")
    print(f"[gen] r2={golden['r2']:.4f}  n_kept={golden['n_kept']}/{golden['n_raw']}")
    print(f"[gen] high-valence pos[:5]={golden['pos_words'][:5]}")
    print(f"[gen] low-valence  neg[:5]={golden['neg_words'][:5]}")


if __name__ == "__main__":
    main()
