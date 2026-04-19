"""End-to-end demo of the ssdiff API (SSD + lexicon)."""

import csv
import os
import sys

import numpy as np

from ssdiff import Corpus, Embeddings, SSD

embeddings_path = "Models/glove_800_normalized.ssdembed"
corpus_path = "Corpuses/Kalibra/kalibra_szczepienie.csv"

emb = Embeddings.load(embeddings_path)

with open(corpus_path, "r", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))
texts = [r["szczepienie_open"] for r in rows]
scores = np.array([float(r["szczepienie_closed"]) for r in rows])
lexicon = {"szczepienie", "szczepić", "szczepionka"}

corpus = Corpus(texts, lang="pl")
ssd = SSD(emb, corpus, scores, lexicon)

print("======= PLS =======")
pls = ssd.fit_pls(verbose=True)
pls.stats
pls.words
pls.clusters.pos
pls.snippets(side="pos")
pls.docs.pos(5)

print("======= PLS: rerun split test =======")
pls.test(n_splits=30)

print("======= PLS report =======")
pls.report(top_words=10, clusters=50)
# pls.report(top_words=10, clusters=50).save("report_pls.md")

print("======= Groups (median split) =======")
gr = ssd.fit_groups(median_split=True, verbose=True)
gr.test
gr.pairs
gr.report(top_words=10, clusters=50)
# gr.report(top_words=10, clusters=50).save("report_groups.md")

print("======= PCA+OLS =======")
ols = ssd.fit_ols(verbose=True)
ols.stats
ols.report(top_words=10, clusters=50)
# ols.report(top_words=10, clusters=50).save("report_ols.md")
# ols.plot_sweep(path="sweep_plot.png")

print("======= Suggest lexicon (continuous) =======")
lex = corpus.suggest_lexicon(scores, top_k=15, min_docs=5)
lex.stats
lex.tokens[:10]
lex.suggestions
lex.report()
# lex.report().save("lexicon_suggestions.md")

print("======= Evaluate lexicon (continuous) =======")
eval_lexicon = lex.tokens[:5]
result = corpus.evaluate_lexicon(scores, eval_lexicon)
result.stats
result.summary
result.report()
# result.report().save("lexicon_eval.md")
