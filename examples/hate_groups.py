"""Group comparison on the Measuring Hate Speech comments.

``fit_groups`` uses the outcome as a group label and compares the groups in
pairs. The label is the target group of the comment: women, black, muslim or
gay. A comment flagged for exactly one of the four gets that label. A comment
flagged for none or for several gets no label and drops out of the fit.

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

groups = ["women", "black", "muslim", "gay"]
flags = df[groups]
# A missing label drops the document from the fit.
df["target"] = flags.idxmax(axis=1).where(flags.sum(axis=1) == 1)
show('df["target"].value_counts(dropna=False)', df["target"].value_counts(dropna=False))

emb = Embeddings.load("en")
ssd = SSD(emb, df["text"], use_full_doc=True)
show("ssd", ssd)

# n_perm=1000 (default 5000) shortens the run.
gr = ssd.fit_groups("target", data=df, n_perm=1000)
show("gr", gr)
show("gr.test", gr.test)
show("gr.pairs", gr.pairs)
show("gr.group_labels", gr.group_labels)
for p in gr.pairs:
    pair = (p.g1, p.g2)
    show(f"gr[{pair!r}].words", gr[pair].words)
    show(f"gr[{pair!r}].clusters", gr[pair].clusters)
