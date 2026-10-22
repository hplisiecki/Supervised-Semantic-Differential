"""Lexicon analysis: how are chosen words used, and how does that use relate to an outcome?

A lexicon is a list of words. With a lexicon, a document's vector is built only
from the words around each lexicon word (a window of a few tokens on each
side), not from the whole document. The gradient then describes the contexts
in which the lexicon words are used: which ways of talking about them go with
a higher outcome. The model is the one of regression.py, ``score ~ ssd()``.

The steps:

1. ``suggest_lexicon`` lists words close in meaning to a few seed words and how
   each relates to the outcome.
2. ``evaluate_lexicon`` checks the chosen list: how many documents contain its
   words, and whether those words go with the outcome.
3. ``with_lexicon`` returns a new ``SSD`` that uses the list, and the fit runs
   on it.

Data: the Measuring Hate Speech comments (Kennedy et al., 2020). Each comment is
one document and its hate-speech score is the outcome. The lexicon names the
groups the comments are about.

The English wordpack is loaded with ``Embeddings.load("en")``. The first call
downloads it (about 380 MB) into ``~/.wordpacks`` (or ``WORDPACK_CACHE``).
"""

import os

import pandas as pd

from ssdiff import SSD, Embeddings

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "data", "measuring_hate_speech.csv.gz"))

emb = Embeddings.load("en")

# Without a lexicon and without use_full_doc, the SSD can only tokenize; that
# is all the lexicon helpers need.
ssd = SSD(emb, df["text"])

# ---- 1. Words near the seeds ----
# For each seed: the top_k closest words that occur in the texts. Every seed
# must be in the embeddings.
seeds = ["woman", "gay", "muslim", "black", "immigrant", "jewish"]
lex = ssd.suggest_lexicon("score", data=df, seeds=seeds, top_k=5)
print(lex.suggestions)

# ---- 2. Check the chosen list ----
# A lexicon matches exact tokens, so every form to count is listed.
lexicon = [
    "woman", "women", "girl", "lesbian", "gay", "gays", "homosexual",
    "muslim", "muslims", "islam", "islamic", "black", "blacks",
    "immigrant", "immigrants", "jewish", "jew", "jews",
]
scored = ssd.evaluate_lexicon("score", lexicon, data=df)
# Per word: how often it occurs and how it relates to the outcome.
print(scored.suggestions)
# For the whole list: docs_any is the number of documents with at least one
# lexicon word. Documents without one have no vector and drop out of the fit.
# ssd.coverage_summary(...) returns the same numbers as a dict.
print(scored.summary)

# ---- 3. Fit on the lexicon contexts ----
# window=3: three tokens on each side of a lexicon word (the default).
ssd_lex = ssd.with_lexicon(lexicon, window=3)
res = ssd_lex.fit_regression("score ~ ssd()", data=df)
res.summary()
print(res.words)
# Snippets show the lexicon words in the comments that lie furthest toward each
# end of the gradient. Other views: .clusters, .stats, .test, .docs.
print(res.snippets)
