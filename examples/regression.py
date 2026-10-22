"""Regression: which direction in the meaning of the text predicts an outcome?

The model is ``outcome ~ ssd()``. ``ssd()`` stands for the text. ssdiff turns
each document into one vector (the mean of its word vectors), reduces the
vectors to K principal components, and regresses the outcome on those
components. K is chosen by a sweep. The fitted coefficients are mapped back to
one direction in the embedding space, the gradient. Words close to the
gradient's positive end go with a higher outcome, words close to its negative
end with a lower one. Covariates go into the formula like in any regression.

Data: Warriner et al. (2013) word norms. Each word is one document, its mean
valence rating is the outcome, and its mean arousal rating is the covariate.

The English wordpack is loaded with ``Embeddings.load("en")``. The first call
downloads it (about 380 MB) into ``~/.wordpacks`` (or ``WORDPACK_CACHE``).
"""

import os
import tempfile

import pandas as pd

from ssdiff import SSD, Embeddings

HERE = os.path.dirname(os.path.abspath(__file__))

# keep_default_na=False: pandas would otherwise read the word "null" as missing.
df = pd.read_csv(os.path.join(HERE, "data", "BRM-emot-submit.csv"), keep_default_na=False, low_memory=False)
df = df.rename(columns={"Word": "word", "V.Mean.Sum": "valence", "A.Mean.Sum": "arousal"})

# A wordpack holds the word vectors and the tokenizer, so texts are passed as
# plain strings.
emb = Embeddings.load("en")

# SSD holds one text and nothing about the outcome; each fit names its outcome.
# use_full_doc=True builds each document's vector from all its words (the
# alternative is a lexicon, see lexicon_analysis.py). A word the tokenizer
# drops has no vector, and its row drops out of every fit.
ssd = SSD(emb, df["word"], use_full_doc=True)
print(ssd)
print("documents with a vector:", ssd.n_kept)

# ---- The text alone ----
# The outcome and covariates are columns of `data`. The p-value tests the
# model with ssd() against the model without it.
res = ssd.fit_regression("valence ~ ssd()", data=df)
res.summary()

# The words closest to each end of the gradient, then the same words grouped
# into clusters of similar meaning. Other views: .stats, .test, .sweep
# (the K sweep), .snippets, .docs.
print(res.words)
print(res.clusters)

# ---- With a covariate ----
# The gradient is now the direction that predicts valence with arousal held
# fixed. The test still compares the model with and without ssd(), so it asks
# what the text adds beyond arousal.
res_cov = ssd.fit_regression("valence ~ arousal + ssd()", data=df)
res_cov.summary()
# The covariates' own coefficients.
print(res_cov.covariates)
print(res_cov.words)

# ---- Saving the results ----
# save_page writes every view into one HTML file that opens from disk.
# Also on every result: report() (all tables as text), predict(texts) for new
# documents, and pickling; a pickled result is restored without its
# embeddings, so call attach(embeddings=emb) after loading it.
out = tempfile.mkdtemp(prefix="ssdiff_regression_")
res_cov.save_page(os.path.join(out, "regression.html"), title="Valence ~ arousal + ssd()")
print("results page:", os.path.join(out, "regression.html"))
