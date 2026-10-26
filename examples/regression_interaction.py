"""Interaction: does the text's link to the outcome differ between two kinds of document?

The model is ``score ~ women * ssd()``. As in regression.py, ``ssd()`` stands
for the text. Written with ``*``, the formula has two text terms:

- ``ssd(text)``: the gradient for comments that do not target women
  (``women == 0``);
- ``women:ssd(text)``: how that gradient changes when a comment targets women.

Each text term gets its own gradient, word list and clusters. The gradient for
comments that target women is the sum of the two. ``ssdiff.contrast`` then turns
the difference between the two groups' gradients into a word list of its own.

Data: the Measuring Hate Speech comments (Kennedy et al., 2020). Each comment is
one document, its hate-speech score is the outcome, and ``women`` is 1 when the
annotators said the comment targets women.

The English wordpack is loaded with ``Embeddings.load("en")``. The first call
downloads it (about 380 MB) into ``~/.wordpacks`` (or ``WORDPACK_CACHE``).
"""

import os

import pandas as pd

import ssdiff
from ssdiff import SSD, Embeddings

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "data", "measuring_hate_speech.csv.gz"))

emb = Embeddings.load("en")
ssd = SSD(df["text"], embeddings=emb, use_full_doc=True)

# The text column is named "text", so ssd() is shown as ssd(text).
res = ssd.fit_regression("score ~ women * ssd()", data=df)
res.summary()

# One leaf per text term, keyed by the term. Each leaf has the views of a
# single-term result (.words, .clusters, .snippets, ...).
print(res.keys())

# The test of all text terms together (model with vs. without every ssd()
# term), then one test per text term. The women:ssd(text) row asks whether the
# gradient differs between the two kinds of comment.
print(res.test)

base = res["ssd(text)"]
change = res["women:ssd(text)"]
print(change.words)

# ---- Contrast between the two groups' gradients ----
# A leaf's beta is the gradient of its term alone, so the gradient for
# comments targeting women is base + change. contrast() compares the two
# gradients by direction (each is scaled to length 1 first), so its words are
# the ones that move most between the groups. It has no test of its own.
c = ssdiff.contrast(base.beta + change.beta, base.beta, emb, labels=("targets women", "other"))
print(c.stats)
print(c.words)
