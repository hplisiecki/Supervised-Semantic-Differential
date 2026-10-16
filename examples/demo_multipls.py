"""MultiPLS demo — k=3 varimax-rotated supervised dimensions from valence.

INTERACTIVE USE
───────────────
Every result / view below has a rich ``__repr__`` (text REPL) and
``_repr_html_`` (Jupyter), so in a notebook or REPL just type the bare
expression and it auto-displays::

    mp
    mp.words
    mp["dim-1"].words

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
corpus = Corpus(df["word"].astype(str).str.strip().tolist(), lang="en")
ssd = SSD(emb, corpus, df["valence"].to_numpy(), use_full_doc=True)

show("emb", emb)
show("corpus", corpus)
show("ssd", ssd)

print("\n══════ MultiPLS, k=3, varimax rotation ══════")
mp = ssd.fit_multipls(k=3, rotate="varimax", verbose=True)
show("mp", mp)
show("mp.stats", mp.stats)
show("mp.test", mp.test)
show("mp.words", mp.words)

print("\n══════ Per-dim leaves ══════")
for i in range(mp.n_components):
    key = f"dim-{i+1}"
    show(f'mp["{key}"]', mp[key])
    show(f'mp["{key}"].words', mp[key].words)
    show(f'mp["{key}"].clusters', mp[key].clusters)

print("\n══════ Markdown report ══════")
print(mp.report(top_words={"n": 10}))
# mp.report(top_words={"n": 10}).save("report_multipls.md")

print("\n══════ Results page ══════")
mp.save_page("ssd_results_multipls.html", title="Warriner ratings (MultiPLS)")

print("\n══════ Sparse MultiPLS (keep) ══════")
# keep selects K via the sequential p-value chain at a fixed k; each component
# keeps only `keep` non-zero weights. rotate="raw" keeps each axis exactly
# sparse (varimax mixes them). keep="auto" cross-validates the keep-count.
sp = ssd.fit_multipls(k=3, keep=30, rotate="raw", verbose=True)
show("sp", sp)
show("sp.pls_info", sp.pls_info)  # keep=30, pvalue_source="split_nb_seq"
# sp.stats.pvalue is the path-max of the chain (FWER "all k real"), so the
# chain is fixed at fit time and sp.test(...) raises — re-fit to change it.
