"""A custom embedding and spaCy lemmas.

The embedding is a 10,000-word slice of GloVe 42B 300d, trimmed to show the
pipeline. The results of this script mean nothing.

Steps: load a raw embedding file, save it as ``.kv`` and ``.txt`` and load each
one back, lemmatize the comments with spaCy, normalise the embedding, and fit.

A raw embedding file (not a wordpack) cannot tokenize strings, so the texts are
passed as token lists. spaCy does that tokenizing here and nothing else.
Neither spaCy nor its English model is a dependency of ssdiff::

    pip install spacy gensim
    python -m spacy download en_core_web_sm

``gensim`` is needed only for the ``.kv`` format.
"""

import os
import tempfile
import warnings

import pandas as pd
import spacy

from ssdiff import SSD, Embeddings


def show(expr, value):
    """Script-mode mirror of REPL auto-display: print ``>>> expr`` then value."""
    print(f"\n>>> {expr}\n{value}")


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = tempfile.mkdtemp(prefix="ssdiff_custom_embedding_")

# Loading a non-wordpack file warns that texts must be token lists, as they are here.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    emb = Embeddings.load(os.path.join(HERE, "data", "glove_10k.bin"))
show("emb", emb)

print("\n====== Other file formats ======")
# save takes the path without its extension. .kv needs gensim.
emb.save(os.path.join(OUT, "glove_10k"), fmt="kv")
emb.save(os.path.join(OUT, "glove_10k"), fmt="txt")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    from_kv = Embeddings.load(os.path.join(OUT, "glove_10k.kv"))
    from_txt = Embeddings.load(os.path.join(OUT, "glove_10k.txt"))
show("from_kv", from_kv)
show("from_txt", from_txt)

print("\n====== Lemmatize with spaCy ======")
df = pd.read_csv(os.path.join(HERE, "data", "measuring_hate_speech.csv.gz"))
# The parser and the entity recognizer are not needed for lemmas. The remaining
# pipeline is tok2vec, tagger, attribute_ruler, lemmatizer.
nlp = spacy.load("en_core_web_sm", exclude=["parser", "ner"])
tokens = [
    [t.lemma_.lower() for t in doc if t.is_alpha]
    for doc in nlp.pipe(df["text"], batch_size=256)
]
show("tokens[0]", tokens[0])

print("\n====== Normalise, then fit ======")
# SSD raises on embeddings that are not L2-normalised, and the rows of the file
# are raw GloVe. normalize() does L2, then ABTT with 1 component.
emb.normalize(l2=True, abtt=1)
ssd = SSD(emb, tokens, use_full_doc=True)
show("ssd", ssd)

pls = ssd.fit_pls(df["score"].to_numpy(), n_splits=20)
show("pls", pls)
show("pls.stats", pls.stats)
show("pls.words", pls.words)
