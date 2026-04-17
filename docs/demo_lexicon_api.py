"""Demo of the Corpus lexicon API (suggest_lexicon, evaluate_lexicon).

    python demo_lexicon_api.py
    python demo_lexicon_api.py --corpus /path/to/kalibra.csv
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ssdiff import Corpus

parser = argparse.ArgumentParser()
parser.add_argument("--corpus", default="Corpuses/Kalibra/kalibra_szczepienie.csv")
args = parser.parse_args()

# ── Load data ────────────────────────────────────────────────────
with open(args.corpus, "r", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

texts = [r["szczepienie_open"] for r in rows]
scores = np.array([float(r["szczepienie_closed"]) for r in rows])

corpus = Corpus(texts, lang="pl")
print(corpus)
print()

# ── Suggest lexicon ──────────────────────────────────────────────
# Returns a LexiconResult: iterable of dicts, with .tokens and .report()

suggestions = corpus.suggest_lexicon(scores, top_k=15, min_docs=5)
print("=== Suggestions ===")
suggestions.report()
print()

# ── Evaluate a chosen lexicon ────────────────────────────────────
# evaluate_lexicon = token_stats + coverage_summary in one call

lexicon = suggestions.tokens[:5]
print(f"=== Evaluating lexicon: {lexicon} ===")
result = corpus.evaluate_lexicon(scores, lexicon)
result.report()
print()

# ── Categorical mode (median split) ─────────────────────────────
median = np.median(scores)
groups = np.where(scores >= median, "high", "low")

print("=== Categorical suggestions ===")
corpus.suggest_lexicon(groups, top_k=10, min_docs=5, var_type="categorical").report()
print()

print("=== Categorical evaluation ===")
corpus.evaluate_lexicon(groups, lexicon, var_type="categorical").report()
