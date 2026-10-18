"""MultiPLS demo — 3 word-rotated supervised dimensions from valence, arousal and dominance.

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
ssd = SSD(emb, corpus, ["valence", "arousal", "dominance"], use_full_doc=True, data=df)

show("emb", emb)
show("corpus", corpus)
show("ssd", ssd)

print("\n══════ MultiPLS, 3 dimensions, words rotation ══════")
mp = ssd.fit_multipls(ssd_gradients=3, rotate="words", verbose=True)
show("mp", mp)
show("mp.stats", mp.stats)
show("mp.test", mp.test)
show("mp.words", mp.words)

print("\n══════ Per-dim leaves ══════")
for key in mp.keys():
    show(f'mp["{key}"]', mp[key])
    show(f'mp["{key}"].words', mp[key].words)
    show(f'mp["{key}"].outcomes', mp[key].outcomes)
    show(f'mp["{key}"].clusters', mp[key].clusters)

print("\n══════ Markdown report ══════")
print(mp.report(top_words={"n": 10}))
# mp.report(top_words={"n": 10}).save("report_multipls.md")

print("\n══════ Results page ══════")
mp.save_page("ssd_results_multipls.html", title="Warriner ratings (MultiPLS)")
