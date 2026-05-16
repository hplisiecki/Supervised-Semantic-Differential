"""PLS demo — supervised semantic dimension from word valence ratings.

Lexical-norm setup: each "doc" is a single word, y = Warriner valence.
Runs on ./data/ — GloVe-300 (top-20k EN) + Warriner ratings (n=13,396).

INTERACTIVE USE
───────────────
Every result / view below has a rich ``__repr__`` (text REPL) and
``_repr_html_`` (Jupyter), so in a notebook or REPL just type the bare
expression and it auto-displays::

    pls
    pls.words
    pls.clusters.pos.words

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

print("\n══════ PLS, k='auto' ══════")
pls = ssd.fit_pls(verbose=True)
show("pls", pls)
show("pls.stats", pls.stats)
show("pls.test", pls.test)
show("pls.words", pls.words)
show("pls.clusters", pls.clusters)
show("pls.clusters.pos.words", pls.clusters.pos.words)
show("pls.clusters.neg.words", pls.clusters.neg.words)
show("pls.snippets", pls.snippets)

print("\n══════ PLS, k=2 (fixed) ══════")
pls_k2 = ssd.fit_pls(k=2, verbose=True)
show("pls_k2.stats", pls_k2.stats)

print("\n══════ Rerun split-half test ══════")
pls.test(n_splits=30)
show("pls.test", pls.test)

print("\n══════ Markdown report ══════")
print(pls.report(clusters={"n": 30, "n_words": 10, "n_snippets": 1}))
# pls.report(clusters={"n": 30, "n_words": 10, "n_snippets": 1}).save("report_pls.md")
