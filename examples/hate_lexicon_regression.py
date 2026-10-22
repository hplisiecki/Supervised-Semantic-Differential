"""PCA+OLS with a lexicon on the Measuring Hate Speech comments.

The outcome is the comment's hate speech score. A lexicon restricts the text to
windows around the listed words, so the gradient describes how those words are
used. The lexicon is built with the helpers on ``SSD``: ``suggest_lexicon`` from
seed words, then ``evaluate_lexicon`` and ``coverage_summary``.

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

emb = Embeddings.load("en")

# No lexicon yet: the helpers below only need the tokens.
ssd = SSD(emb, df["text"])
show("ssd", ssd)

print("\n====== Tokens around the seeds ======")
# The table lists each seed, its top_k closest tokens in the text, and the seed
# with an English ending, highest cosine first. Every seed must be in the embeddings.
seeds = ["woman", "gay", "muslim", "black", "immigrant", "jewish"]
lex = ssd.suggest_lexicon("score", data=df, seeds=seeds, top_k=5)
show("lex", lex)
show("lex.suggestions", lex.suggestions)
show("lex.tokens", lex.tokens)

print("\n====== Score the chosen lexicon ======")
# A lexicon matches exact tokens, so every form to count is listed.
lexicon = [
    "woman", "women", "girl", "lesbian", "gay", "gays", "homosexual",
    "muslim", "muslims", "islam", "islamic", "black", "blacks",
    "immigrant", "immigrants", "jewish", "jew", "jews",
]
scored = ssd.evaluate_lexicon("score", lexicon, data=df)
show("scored.suggestions", scored.suggestions)
show("scored.summary", scored.summary)
show('ssd.coverage_summary("score", lexicon, data=df)', ssd.coverage_summary("score", lexicon, data=df))

print("\n====== PCA+OLS on the lexicon contexts ======")
# with_lexicon keeps the tokens and returns a new SSD that uses the lexicon.
# window=3 tokens on each side of a lexicon word (the default).
ssd_lex = ssd.with_lexicon(lexicon, window=3)
show("ssd_lex", ssd_lex)
res = ssd_lex.fit_regression("score", data=df)
show("res", res)
show("res.summary()", res.summary())
show("res.stats", res.stats)
show("res.test", res.test)
show("res.words", res.words)
show("res.clusters", res.clusters)
show("res.clusters.pos.words", res.clusters.pos.words)
show("res.clusters.neg.words", res.clusters.neg.words)
show("res.snippets", res.snippets)
