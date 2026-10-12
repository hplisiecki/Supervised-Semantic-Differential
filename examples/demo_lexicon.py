"""Lexicon demo — find the tokens around seed words, then score a lexicon.

Lexical-norm setup: each "doc" is a single word, y = Warriner valence.
Runs on ./data/ — GloVe-300 (top-20k EN) + Warriner ratings (n=13,396).

INTERACTIVE USE
───────────────
Every result / view below has a rich ``__repr__`` (text REPL) and
``_repr_html_`` (Jupyter), so in a notebook or REPL just type the bare
expression and it auto-displays::

    lex
    lex.suggestions
    lex.suggestions[:]

This script wraps each in ``show("expr", expr)`` so command-line users
see the same output, prefixed with a ``>>> expr`` label that matches
what you would type interactively. Drop the ``show(...)`` wrapper and
paste the bare expression into Jupyter/REPL to get the same result.
"""

import os

import pandas as pd

from ssdiff import Corpus, Embeddings


def show(expr, value):
    """Script-mode mirror of REPL auto-display: print ``>>> expr`` then value."""
    print(f"\n>>> {expr}\n{value}")


HERE = os.path.dirname(os.path.abspath(__file__))
EMB = os.path.join(HERE, "data", "glove_300_en_top20k_ratings.bin")
RATINGS = os.path.join(HERE, "data", "Ratings_EN.csv")

emb = Embeddings.load(EMB)
df = pd.read_csv(RATINGS)
valence = df["valence"].to_numpy()
corpus = Corpus(df["word"].astype(str).str.strip().tolist(), lang="en")

show("emb", emb)
show("corpus", corpus)

print("\n══════ Tokens around the seeds ══════")
# One table for all seeds, highest cosine first. `seed` is the seed a token
# is closest to. Here every doc is one word, so min_docs=1; on real texts the
# default of 5 drops rare tokens.
lex = corpus.suggest_lexicon(
    valence, seeds=["happy", "angry"], embeddings=emb, top_k=15, min_docs=1,
)
show("lex", lex)
show("lex.suggestions", lex.suggestions)          # the first 20 rows
show("lex.suggestions[:]", lex.suggestions[:])    # every row
show('lex.suggestions.to_df(cols="all")', lex.suggestions.to_df(cols="all"))
show("lex.tokens", lex.tokens)

print("\n══════ Score the chosen lexicon ══════")
# A lexicon matches exact tokens, so list every form you want counted. Read
# the table above and write your own list; the first ten tokens stand in here.
lexicon = lex.tokens[:10]
scored = corpus.evaluate_lexicon(valence, lexicon)
show("scored.suggestions", scored.suggestions)
show("scored.summary", scored.summary)

print("\n══════ Markdown report ══════")
print(lex.report(top={"n": 30}))
# lex.report(top={"n": 30}).save("report_lexicon.md")
# lex.suggestions.save("lexicon_suggestions.csv")
