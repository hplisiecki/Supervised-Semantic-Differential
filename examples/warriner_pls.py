"""PLS on word valence (Warriner et al., 2013).

Same data as ``warriner_regression.py``: each word is one document and its mean
valence is the outcome, with ``use_full_doc=True`` and no lexicon.

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

# keep_default_na=False: pandas would otherwise read the word "null" as missing.
df = pd.read_csv(os.path.join(HERE, "data", "BRM-emot-submit.csv"), keep_default_na=False, low_memory=False)
df = df.rename(columns={"Word": "word", "V.Mean.Sum": "valence"})

emb = Embeddings.load("en")
ssd = SSD(emb, df["word"], use_full_doc=True)
show("ssd", ssd)

print("\n====== PLS, k='auto' ======")
# k="auto" picks the number of components (up to k_max=5); the p-value is the
# confirmatory test of the first component. n_splits=20 (default 50) shortens the run.
pls = ssd.fit_pls("valence", data=df, k="auto", n_splits=20)
show("pls", pls)
show("pls.summary()", pls.summary())
show("pls.stats", pls.stats)
show("pls.test", pls.test)
show("pls.words", pls.words)
show("pls.clusters", pls.clusters)
show("pls.clusters.pos.words", pls.clusters.pos.words)
show("pls.clusters.neg.words", pls.clusters.neg.words)
show("pls.snippets", pls.snippets)

print("\n====== PLS, k=2 (fixed) ======")
pls_k2 = ssd.fit_pls("valence", data=df, k=2, n_splits=20)
show("pls_k2.stats", pls_k2.stats)

print("\n====== PLS, ssd_gradients=3 (three rotated dimensions) ======")
# The headline fit (k, p-value, r2) is the same as above. The result also holds
# three word-rotated dimensions, from a separate 3-component PLS1 fit:
# result["dim-1"] ... result["dim-3"].
pls_g3 = ssd.fit_pls("valence", data=df, ssd_gradients=3, rotate="words", n_splits=20)
show("pls_g3", pls_g3)
show("pls_g3.pls_info", pls_g3.pls_info)
for key in pls_g3.keys():
    show(f'pls_g3["{key}"].stats', pls_g3[key].stats)
    show(f'pls_g3["{key}"].words', pls_g3[key].words)
