"""PCA regression demo — sweep mode, fixed-k mode, and a covariate.

Lexical-norm setup: y = Warriner valence over 13,396 English words.

INTERACTIVE USE
───────────────
Every result / view below has a rich ``__repr__`` (text REPL) and
``_repr_html_`` (Jupyter), so in a notebook or REPL just type the bare
expression and it auto-displays::

    ols
    ols.words
    ols.clusters

This script wraps each in ``show("expr", expr)`` so command-line users
see the same output, prefixed with a ``>>> expr`` label that matches
what you would type interactively. Drop the ``show(...)`` wrapper and
paste the bare expression into Jupyter/REPL to get the same result.
"""

import os

import pandas as pd

from ssdiff import SSD, Corpus, Embeddings


def show(expr, value):
    """Script-mode mirror of REPL auto-display: print ``>>> expr`` then value."""
    print(f"\n>>> {expr}\n{value}")


HERE = os.path.dirname(os.path.abspath(__file__))
EMB = os.path.join(HERE, "data", "glove_300_en_top20k_ratings.bin")
RATINGS = os.path.join(HERE, "data", "Ratings_EN.csv")

emb = Embeddings.load(EMB)
df = pd.read_csv(RATINGS)
corpus = Corpus(df["word"].astype(str).str.strip().tolist(), lang="en")
ssd = SSD(emb, corpus, "valence", use_full_doc=True, data=df)

show("emb", emb)
show("corpus", corpus)
show("ssd", ssd)

print("\n══════ PCA regression, sweep over k ══════")
ols = ssd.fit_regression(verbose=True)
show("ols", ols)
show("ols.stats", ols.stats)
show("ols.test", ols.test)
show("ols.words", ols.words)
show("ols.clusters", ols.clusters)
show("ols.clusters.pos.words", ols.clusters.pos.words)
show("ols.clusters.neg.words", ols.clusters.neg.words)
# ols.plot_sweep("sweep_plot.png")

print("\n══════ PCA regression, fixed k=20 ══════")
ols_k20 = ssd.fit_regression(fixed_k=20, verbose=True)
show("ols_k20.stats", ols_k20.stats)
show("ols_k20.words", ols_k20.words)

print("\n══════ PCA regression controlling for arousal, fixed k=20 ══════")
adj = ssd.fit_regression("~ arousal", fixed_k=20)
show("adj.stats", adj.stats)
print("\n>>> adj.model.summary()")
adj.model.summary()   # prints the glmm summary

print("\n══════ Markdown report ══════")
print(ols.report(clusters={"n": 30, "n_words": 10, "n_snippets": 1}))
# ols.report(clusters={"n": 30, "n_words": 10, "n_snippets": 1}).save("report_ols.md")

print("\n══════ Results page ══════")
ols.save_page("ssd_results_pcaols.html", title="Warriner valence (PCA+OLS)")
