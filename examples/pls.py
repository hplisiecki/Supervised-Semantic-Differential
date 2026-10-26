"""PLS: which direction in the meaning of the text predicts an outcome?

The same question as regression.py, answered with partial least squares
(PLS1). Regression picks its components from the text alone (PCA) and then
regresses the outcome on them. PLS picks components that covary with the
outcome. Both end in one gradient: a direction in the embedding space whose
positive end goes with a higher outcome. PLS has its own significance test
(Lenartowicz & Plisiecki, 2026, arXiv:2609.36307).

Data: Warriner et al. (2013) word norms. Each word is one document, its mean
valence rating is the outcome, and its mean arousal rating is the covariate.

The English wordpack is loaded with ``Embeddings.load("en")``. The first call
downloads it (about 380 MB) into ``~/.wordpacks`` (or ``WORDPACK_CACHE``).
"""

import os

import pandas as pd

from ssdiff import SSD, Embeddings

HERE = os.path.dirname(os.path.abspath(__file__))

# keep_default_na=False: pandas would otherwise read the word "null" as missing.
df = pd.read_csv(os.path.join(HERE, "data", "BRM-emot-submit.csv"), keep_default_na=False, low_memory=False)
df = df.rename(columns={"Word": "word", "V.Mean.Sum": "valence", "A.Mean.Sum": "arousal"})

emb = Embeddings.load("en")
# One text; each fit names its outcome. See regression.py for use_full_doc.
ssd = SSD(df["word"], embeddings=emb, use_full_doc=True)

# ---- The text alone ----
# The outcome is a column name of `data`. k="auto" picks the number of PLS
# components (up to k_max=5). The p-value tests the first component, whatever
# k is picked. n_splits=20 (default 50) only shortens the run.
pls = ssd.fit_pls("valence", data=df, k="auto", n_splits=20)
pls.summary()
# Other views: .stats, .test, .clusters, .snippets, .docs.
print(pls.words)

# ---- With a covariate ----
# A PLS formula lists covariates only: no ssd() term and no interactions.
# The outcome and every document-vector column are first regressed on the
# covariates, and PLS runs on what is left. So the gradient predicts valence
# with arousal held fixed. predict() is not available for such a fit.
pls_cov = ssd.fit_pls("valence ~ arousal", data=df, k="auto", n_splits=20)
pls_cov.summary()
print(pls_cov.words)

# ---- Several gradients ----
# The gradient above is one direction. ssd_gradients=3 also fits three
# components and rotates them (rotate="words", the default) so that each reads
# as its own group of words. They are result["dim-1"] ... result["dim-3"]; the
# headline fit and its p-value are the same as with one gradient.
pls3 = ssd.fit_pls("valence", data=df, ssd_gradients=3, n_splits=20)
for key in pls3.keys():
    print(key)
    print(pls3[key].words)
