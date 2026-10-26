"""Several outcomes: which directions in the text relate to a set of outcomes at once?

``fit_multipls`` fits PLS3 (PLS correlation). It finds dimensions shared by the
text and all outcomes together. Each dimension pairs a direction in the
embedding space (with its word list) with one weight per outcome, which says
how strongly each outcome goes along that dimension. The test asks whether the
first dimension is more than noise.

Data: the Measuring Hate Speech comments (Kennedy et al., 2020). Each comment is
one document. There is one outcome per target group: the comment's hate-speech
score if the annotators said it targets that group, and 0 otherwise. The
script builds these columns from ``score`` and the 0/1 target columns.

The English wordpack is loaded with ``Embeddings.load("en")``. The first call
downloads it (about 380 MB) into ``~/.wordpacks`` (or ``WORDPACK_CACHE``).
"""

import os

import pandas as pd

from ssdiff import SSD, Embeddings

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "data", "measuring_hate_speech.csv.gz"))

targets = ["women", "black", "muslim", "gay", "immigrant", "jewish"]
for t in targets:
    df[f"score_{t}"] = df["score"] * df[t]
outcomes = [f"score_{t}" for t in targets]

emb = Embeddings.load("en")
ssd = SSD(df["text"], embeddings=emb, use_full_doc=True)

# The outcome is a list of column names of `data` (no formula). By default
# there is one dimension per outcome. rotate="words" rotates the dimensions so
# that each reads as its own group of words; the test is on the first
# dimension before rotation, so rotating does not change the p-value.
# n_splits=20 (default 50) shortens the run.
mp = ssd.fit_multipls(outcomes, data=df, rotate="words", n_splits=20)
print(mp)
print(mp.test)

# One leaf per dimension, "dim-1", "dim-2", ... .outcomes gives the outcome
# weights of the dimension, .words its word list. Other views of a leaf:
# .stats, .clusters, .snippets.
for key in mp.keys():
    print(key)
    print(mp[key].outcomes)
    print(mp[key].words)
