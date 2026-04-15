"""Demo of ssdiff API.

    python demo_new_api.py
    python demo_new_api.py --path /path/to/embeddings.ssdembed
"""

import argparse
import csv
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ssdiff import Embeddings, Corpus, SSD

parser = argparse.ArgumentParser()
parser.add_argument("--path", default="Models/glove_800_normalized.ssdembed")
parser.add_argument("--corpus", default="Corpuses/Kalibra/kalibra_szczepienie.csv")
args = parser.parse_args()

# ── Load data ────────────────────────────────────────────────────
emb = Embeddings.load(args.path)
print(emb)

with open(args.corpus, "r", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

texts = [r["szczepienie_open"] for r in rows]
scores = np.array([float(r["szczepienie_closed"]) for r in rows])
lexicon = {"szczepienie", "szczepić", "szczepionka"}

corpus = Corpus(texts, lang="pl")
print(corpus)

# ── Continuous SSD ───────────────────────────────────────────────
ssd = SSD(emb, corpus, scores, lexicon)
print(ssd)

# PLS
pls = ssd.fit_pls(verbose=True)
print(pls)
print()
print(pls.summary())
print()
pls.report(top_words=10, clusters=50, extreme_docs=5, misdiagnosed=5)
print()
print(pls.split_test(n_splits=30))

# ── Group (median split) ────────────────────────────────────────
gr = ssd.fit_groups(median_split=True, verbose=True)
print(gr)
print()
print(gr.summary())
print()
gr.report(top_words=10, clusters=50)

# ── PCA+OLS sweep ───────────────────────────────────────────────
ols = ssd.fit_ols(verbose=True)
print(ols)
print()
print(ols.summary())
print()
ols.report(top_words=10, clusters=50, extreme_docs=5, misdiagnosed=5)
ols.plot_sweep(path="sweep_plot.png")
