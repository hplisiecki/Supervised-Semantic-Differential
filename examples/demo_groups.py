"""Groups demo — median split (2 groups) and tercile split (3 groups).

INTERACTIVE USE
───────────────
Every result / view below has a rich ``__repr__`` (text REPL) and
``_repr_html_`` (Jupyter), so in a notebook or REPL just type the bare
expression and it auto-displays::

    gr2
    gr2.pairs
    gr2[("g1", "g2")].words

This script wraps each in ``show("expr", expr)`` so command-line users
see the same output, prefixed with a ``>>> expr`` label that matches
what you would type interactively. Drop the ``show(...)`` wrapper and
paste the bare expression into Jupyter/REPL to get the same result.

Note: groups are canonically relabelled to ``g1, g2, …`` internally;
``gr.group_labels`` is the canonical → original mapping.
"""

import os

import numpy as np
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
words = df["word"].astype(str).str.strip().tolist()
valence = df["valence"].to_numpy()
corpus = Corpus(words, lang="en")

show("emb", emb)
show("corpus", corpus)

print("\n══════ Median split (low / high valence) ══════")
ssd_med = SSD(emb, corpus, valence, use_full_doc=True)
gr2 = ssd_med.fit_groups(median_split=True, n_perm=1000, verbose=True)
show("gr2", gr2)
show("gr2.test", gr2.test)
show("gr2.pairs", gr2.pairs)
show("gr2.group_labels", gr2.group_labels)
show('gr2[("g1", "g2")].words', gr2[("g1", "g2")].words)
show('gr2[("g1", "g2")].clusters', gr2[("g1", "g2")].clusters)

print("\n══════ Tercile split (3 groups) ══════")
q33, q67 = np.quantile(valence, [1 / 3, 2 / 3])
labels = np.where(valence < q33, "low",
          np.where(valence < q67, "mid", "high")).astype(object)

ssd_3 = SSD(emb, corpus, labels, use_full_doc=True)
gr3 = ssd_3.fit_groups(n_perm=1000, verbose=True)
show("gr3", gr3)
show("gr3.test", gr3.test)
show("gr3.pairs", gr3.pairs)
show("gr3.group_labels", gr3.group_labels)
for p in gr3.pairs:
    show(f'gr3[("{p.g1}", "{p.g2}")].words', gr3[(p.g1, p.g2)].words)

print("\n══════ Markdown report (3-group) ══════")
print(gr3.report(top_words=10, clusters=20))
# gr3.report(top_words=10, clusters=20).save("report_groups.md")
