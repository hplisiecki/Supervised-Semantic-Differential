"""Between-group differences: how do the texts of labelled groups differ?

``fit_groups`` takes one label per document. It compares the mean document
vectors of the groups. The omnibus permutation test asks whether any group
differs from the others; then each pair of groups gets its own permutation
test (Holm-corrected by default) and its own difference direction, with a
word list whose positive end is the pair's first group.

Data: the Measuring Hate Speech comments (Kennedy et al., 2020). Each comment is
one document. Its label is the group the annotators said it targets: women,
black, muslim or gay. A comment flagged for none of the four, or for several,
gets no label and drops out.

The English wordpack is loaded with ``Embeddings.load("en")``. The first call
downloads it (about 380 MB) into ``~/.wordpacks`` (or ``WORDPACK_CACHE``).
"""

import os

import pandas as pd

from ssdiff import SSD, Embeddings

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "data", "measuring_hate_speech.csv.gz"))

groups = ["women", "black", "muslim", "gay"]
flags = df[groups]
# The name of the one flagged column; missing when 0 or several are flagged.
df["target"] = flags.idxmax(axis=1).where(flags.sum(axis=1) == 1)
print(df["target"].value_counts(dropna=False))

emb = Embeddings.load("en")
ssd = SSD(df["text"], embeddings=emb, use_full_doc=True)

# The outcome is the label column of `data`. n_perm=1000 (default 5000)
# shortens the run; it also sets the smallest possible p-value,
# 1/(n_perm + 1) before the Holm correction.
gr = ssd.fit_groups("target", data=df, n_perm=1000)
print(gr)
# The omnibus test, then one row per pair.
print(gr.test)

# The groups are renamed g1, g2, ... in sorted label order; group_labels maps
# each name back to its label. A pair is gr[("g1", "g2")], first group first.
# Other views of a pair: .clusters, .snippets, .stats.
print(gr.group_labels)
print(gr[("g1", "g2")].words)
