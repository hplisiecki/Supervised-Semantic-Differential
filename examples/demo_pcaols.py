"""PCA + OLS demo — sweep mode and fixed-k mode.

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

from ssdiff import Corpus, Embeddings, SSD


def show(expr, value):
    """Script-mode mirror of REPL auto-display: print ``>>> expr`` then value."""
    print(f"\n>>> {expr}\n{value}")


HERE = os.path.dirname(os.path.abspath(__file__))
EMB = os.path.join(HERE, "data", "glove_300_en_top20k_ratings.ssdembed")
RATINGS = os.path.join(HERE, "data", "Ratings_EN.csv")

emb = Embeddings.load(EMB)
df = pd.read_csv(RATINGS)
corpus = Corpus(df["word"].astype(str).str.strip().tolist(), lang="en")
ssd = SSD(emb, corpus, df["valence"].to_numpy(), use_full_doc=True)

show("emb", emb)
show("corpus", corpus)
show("ssd", ssd)

print("\n══════ PCA + OLS, sweep over k ══════")
ols = ssd.fit_ols(verbose=True)
show("ols", ols)
show("ols.stats", ols.stats)
show("ols.test", ols.test)
show("ols.words", ols.words)
show("ols.clusters", ols.clusters)
show("ols.clusters.pos.words", ols.clusters.pos.words)
show("ols.clusters.neg.words", ols.clusters.neg.words)
# ols.plot_sweep("sweep_plot.png")

print("\n══════ PCA + OLS, fixed k=20 ══════")
ols_k20 = ssd.fit_ols(fixed_k=20, verbose=True)
show("ols_k20.stats", ols_k20.stats)
show("ols_k20.words", ols_k20.words)

print("\n══════ Markdown report ══════")
print(ols.report(clusters={"n": 30, "n_words": 10, "n_snippets": 1}))
# ols.report(clusters={"n": 30, "n_words": 10, "n_snippets": 1}).save("report_ols.md")
