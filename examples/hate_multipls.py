"""PLS3 of the comments against one score column per target group.

``fit_multipls`` relates the comment vectors to several outcomes at once. Here
there is one outcome column for each of six target groups: it holds the
comment's hate speech score when the comment targets that group, and 0
otherwise. The columns are built below from ``score`` and the 0/1 target flags.

Loads the English wordpack with ``Embeddings.load("en")``. The first call
downloads it (about 380 MB) into ``~/.wordpacks`` (or ``WORDPACK_CACHE``).

Every result and view prints as text in a REPL; ``show("expr", value)`` prints
``>>> expr`` and then the value, as a REPL would.
"""

import os

import pandas as pd

from ssdiff import SSD, Embeddings


def show(expr, value):
    """Script-mode mirror of REPL auto-display: print ``>>> expr`` then value."""
    print(f"\n>>> {expr}\n{value}")


HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "data", "measuring_hate_speech.csv.gz"))

targets = ["women", "black", "muslim", "gay", "immigrant", "jewish"]
for t in targets:
    df[f"score_{t}"] = df["score"] * df[t]
outcomes = [f"score_{t}" for t in targets]
show("df[outcomes].describe().loc[['count', 'mean']]", df[outcomes].describe().loc[["count", "mean"]])

emb = Embeddings.load("en")
# Full-document vectors: the comments are not restricted to a lexicon.
ssd = SSD(emb, df["text"], use_full_doc=True)
show("ssd", ssd)

# outcome is a list of column names of `data`. rotate="words" rotates the
# dimensions (varimax against the leading vocabulary rows) so that each reads
# as a word-space direction. n_perm=200 (default 1000) and n_splits=20
# (default 50) shorten the run.
mp = ssd.fit_multipls(outcomes, data=df, rotate="words", n_perm=200, n_splits=20)
show("mp", mp)
show("mp.stats", mp.stats)
show("mp.test", mp.test)
show("mp.outcome_names", mp.outcome_names)
show("mp.pls_info", mp.pls_info)
for key in mp.keys():
    show(f'mp["{key}"].stats', mp[key].stats)
    show(f'mp["{key}"].outcomes', mp[key].outcomes)
    show(f'mp["{key}"].words', mp[key].words)
show('mp["dim-1"].clusters', mp["dim-1"].clusters)
show('mp["dim-1"].snippets', mp["dim-1"].snippets)
