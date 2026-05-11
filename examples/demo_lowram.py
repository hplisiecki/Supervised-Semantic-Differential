"""Low-RAM mode demo — partial-load + attach_corpus.

Note: ``ram_efficient=True`` materialises only the top-50,000 frequency-
ranked rows at load time, then ``attach_corpus()`` adds the corpus-needed
rows from the long tail. The bundled embedding has V=25,729 < 50,000,
so the partial load already covers the whole vocab and ``attach_corpus``
is effectively a no-op here. The API call still demonstrates the
intended workflow for production-scale embeddings (V >> 50k).

INTERACTIVE USE
───────────────
Every result / view below has a rich ``__repr__`` (text REPL) and
``_repr_html_`` (Jupyter), so in a notebook or REPL just type the bare
expression and it auto-displays::

    emb
    pls
    pls.words

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

print("══════ Partial load (ram_efficient=True) ══════")
emb = Embeddings.load(EMB, ram_efficient=True)
show("emb", emb)

df = pd.read_csv(RATINGS)
corpus = Corpus(df["word"].astype(str).str.strip().tolist(), lang="en")
show("corpus", corpus)

print("\n══════ After attach_corpus ══════")
emb.attach_corpus(corpus)
show("emb", emb)

print("\n══════ Fit ══════")
ssd = SSD(emb, corpus, df["valence"].to_numpy(), use_full_doc=True)
show("ssd", ssd)
pls = ssd.fit_pls(verbose=True)
show("pls", pls)
show("pls.stats", pls.stats)
show("pls.words", pls.words)
