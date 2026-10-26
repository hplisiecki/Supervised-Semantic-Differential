"""Your own embedding file and your own tokens.

A wordpack (``Embeddings.load("en")``) holds word vectors and a tokenizer. Any
other embedding file (word2vec ``.bin``, ``.txt``, gensim ``.kv``) holds only
vectors, so the texts must come in already tokenized, as lists of tokens. Any
tool can make the tokens; here spaCy turns each comment into a list of lemmas.
The analysis is then the one of regression.py, ``score ~ ssd()``.

The embedding is a 10,000-word slice of GloVe 42B 300d, small enough to ship
with the examples. It is here to show the steps; its results mean nothing.

Data: the Measuring Hate Speech comments (Kennedy et al., 2020). Each comment is
one document and its hate-speech score is the outcome.

Neither spaCy nor its English model is a dependency of ssdiff. ``gensim`` is
needed only for the ``.kv`` format::

    pip install spacy gensim
    python -m spacy download en_core_web_sm
"""

import os
import tempfile
import warnings

import pandas as pd
import spacy

from ssdiff import SSD, Embeddings

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = tempfile.mkdtemp(prefix="ssdiff_custom_embedding_")

# Loading a file that is not a wordpack warns that texts must be token lists,
# as they are here.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    emb = Embeddings.load(os.path.join(HERE, "data", "glove_10k.bin"))
print(emb)

# ---- Other file formats ----
# save takes the path without its extension; load reads the format from the
# extension.
emb.save(os.path.join(OUT, "glove_10k"), fmt="kv")
emb.save(os.path.join(OUT, "glove_10k"), fmt="txt")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    print(Embeddings.load(os.path.join(OUT, "glove_10k.kv")))
    print(Embeddings.load(os.path.join(OUT, "glove_10k.txt")))

# ---- Tokens from spaCy ----
df = pd.read_csv(os.path.join(HERE, "data", "measuring_hate_speech.csv.gz"))
# Lemmas need only the tagger and the lemmatizer, so the parser and the entity
# recognizer are left out.
nlp = spacy.load("en_core_web_sm", exclude=["parser", "ner"])
tokens = [
    [t.lemma_.lower() for t in doc if t.is_alpha]
    for doc in nlp.pipe(df["text"], batch_size=256)
]
print(tokens[0])

# ---- Normalise, then fit ----
# SSD needs L2-normalised word vectors, and the rows of this file are raw
# GloVe. normalize() scales each row to length 1, removes the mean vector and
# the top principal component of all words (ABTT; Mu & Viswanath, 2018), and
# scales the rows to length 1 again.
emb.normalize(l2=True, abtt=1)
ssd = SSD(tokens, embeddings=emb, use_full_doc=True)
res = ssd.fit_regression("score ~ ssd()", data=df)
res.summary()
print(res.words)
