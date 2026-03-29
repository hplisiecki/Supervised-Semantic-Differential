"""Demo of SSDLite new API features.

Run from SSD repo:
    python demo_new_api.py
"""

import csv
import sys
import numpy as np

sys.path.insert(0, "SSDLite")

from ssdlite import Embeddings, Corpus, SSD

# ── Load data ────────────────────────────────────────────────────
print("Loading embeddings...")
emb = Embeddings.load("Models/glove_800_3_polish_normalized.ssdembed")
print(f"  {len(emb)} words, {emb.vector_size}D")

print("Loading corpus (Kalibra szczepienie)...")
with open("Corpuses/Kalibra/kalibra_szczepienie.csv", "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

texts = [r["szczepienie_open"] for r in rows]
scores = np.array([float(r["szczepienie_closed"]) for r in rows])

lexicon = {"szczepienie", "szczepić", "szczepionka"}

print("Tokenizing...")
corpus = Corpus(texts, lang="pl")
print(f"  {corpus.n_texts} documents")

# ── Continuous SSD ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("SSD — Continuous outcome (szczepienie_closed score)")
print("=" * 60)

ssd = SSD(emb, corpus, scores, lexicon, window = 3)
print(f"\n{ssd}")

# PLS fit
print("\nFitting PLS...")
pls = ssd.fit_pls(n_perm=200, random_state=42)

# 1) Summary
print("\n--- summary() ---")
print(pls.summary())

# 2) Effect sizes
print("\n--- Effect-size attributes ---")
print(f"  beta_norm  = {pls.beta_norm:.4f}")
print(f"  delta      = {pls.delta:.4f}")
print(f"  iqr_effect = {pls.iqr_effect:.4f}")
print(f"  y_corr_pred= {pls.y_corr_pred:.4f}")

# 3) Top words
print("\n--- top_words(10) ---")
words = pls.top_words(10)
print(f"  POS: {', '.join(w['word'] for w in words if w['side'] == 'pos')}")
print(f"  NEG: {', '.join(w['word'] for w in words if w['side'] == 'neg')}")

# 4) Extreme docs
print("\n--- extreme_docs(k=5, by='predicted') ---")
extremes = pls.extreme_docs(k=5, by="predicted")
for d in extremes:
    label = "TOP" if d["side"] == "top" else "BOT"
    text_preview = texts[np.where(ssd.keep_mask)[0][d["idx"]]][:180]
    print(f"  [{label}] yhat={d['yhat']:.2f} y={d['y_true']:.1f} cos={d['cos']:.3f} | {text_preview}...")

# 5) Misdiagnosed
print("\n--- misdiagnosed(k=5) ---")
mis = pls.misdiagnosed(k=5)
for d in mis:
    text_preview = texts[np.where(ssd.keep_mask)[0][d["idx"]]][:180]
    print(f"  [{d['side'].upper():5s}] resid={d['residual']:+.2f} yhat={d['yhat']:.2f} y={d['y_true']:.1f} | {text_preview}...")

# 6) Split test
print("\n--- split_test(n_splits=30) ---")
st = pls.split_test(n_splits=30, seed=42)
print(f"  pvalue = {st['pvalue']:.4f}")
print(f"  mean_r = {st['mean_r']:.4f}")

print("\nDone.")
