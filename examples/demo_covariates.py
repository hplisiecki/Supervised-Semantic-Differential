"""Covariates demo — PLS and PCA regression controlling for word length and arousal.

Lexical-norm setup: each "doc" is a single word, y = Warriner valence.
Runs on ./data/ — GloVe-300 (top-20k EN) + Warriner ratings (n=13,396).

INTERACTIVE USE
───────────────
Every result / view below has a rich ``__repr__`` (text REPL) and
``_repr_html_`` (Jupyter), so in a notebook or REPL just type the bare
expression and it auto-displays::

    pls_adj
    reg_adj.covariates
    reg_adj.summary()

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
EMB = os.path.join(HERE, "data", "glove_300_en_top20k_ratings.bin")
RATINGS = os.path.join(HERE, "data", "Ratings_EN.csv")

emb = Embeddings.load(EMB)
df = pd.read_csv(RATINGS)
df["word"] = df["word"].astype(str).str.strip()
df["length"] = df["word"].str.len()
corpus = Corpus(df["word"].tolist(), lang="en")
# data= has one row per document, so the formula can name its columns.
ssd = SSD(emb, corpus, "valence", use_full_doc=True, data=df)

show("ssd", ssd)

print("\n══════ PLS without covariates ══════")
pls = ssd.fit_pls(verbose=True)
show("pls.stats", pls.stats)

print("\n══════ PLS controlling for length and arousal ══════")
# y and the document vectors are residualized on the formula; PLS and its
# permutation test run on the residuals.
pls_adj = ssd.fit_pls("~ length + arousal", verbose=True)
show("pls_adj", pls_adj)
show("pls_adj.stats", pls_adj.stats)     # r2 is the R² of the residualized y
show("pls_adj.test", pls_adj.test)       # the test of the text
show("pls_adj.words", pls_adj.words)
print("\n>>> pls_adj.summary()")
pls_adj.summary()   # the text's test and partial R²; no covariate rows

print("\n══════ PCA regression controlling for length and arousal, fixed k=20 ══════")
reg_adj = ssd.fit_regression("~ length + arousal", fixed_k=20)
show("reg_adj.stats", reg_adj.stats)     # F-test of the 20 components together
# Covariate effects with the 20 components held fixed.
show("reg_adj.covariates", reg_adj.covariates)
print("\n>>> reg_adj.summary()")
reg_adj.summary()

print("\n══════ Results pages ══════")
# The Covariates table has its own tab on the PCA regression page.
pls_adj.save_page("ssd_results_covariates_pls.html",
                  title="Warriner valence, length + arousal (PLS)")
reg_adj.save_page("ssd_results_covariates_regression.html",
                  title="Warriner valence, length + arousal (PCA regression)")
