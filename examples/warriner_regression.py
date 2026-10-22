"""PCA+OLS regression on word valence (Warriner et al., 2013).

Each word is one document and its mean valence (``V.Mean.Sum``) is the outcome,
so no lexicon is needed: ``use_full_doc=True`` embeds the whole document.

Loads the English wordpack with ``Embeddings.load("en")``. The first call
downloads it (about 380 MB) into ``~/.wordpacks`` (or ``WORDPACK_CACHE``);
later calls read it from there. Texts are passed as plain strings and the
wordpack tokenizes them.

INTERACTIVE USE
---------------
Every result and view has a text ``__repr__`` and a Jupyter ``_repr_html_``, so
in a REPL or notebook you type the bare expression, for example ``res.words``.
This script wraps each one in ``show("expr", expr)``, which prints ``>>> expr``
and then the value, as a REPL would.
"""

import os
import pickle
import tempfile

import pandas as pd

from ssdiff import SSD, Embeddings


def show(expr, value):
    """Script-mode mirror of REPL auto-display: print ``>>> expr`` then value."""
    print(f"\n>>> {expr}\n{value}")


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = tempfile.mkdtemp(prefix="ssdiff_warriner_regression_")

# keep_default_na=False: pandas would otherwise read the word "null" as missing.
df = pd.read_csv(os.path.join(HERE, "data", "BRM-emot-submit.csv"), keep_default_na=False, low_memory=False)
df = df.rename(columns={"Word": "word", "V.Mean.Sum": "valence"})

emb = Embeddings.load("en")
show("emb", emb)

# A pandas Series gives the text its name ("word"). The wordpack's tokenizer
# can split or drop a word; such a document has no vector and is left out of the fit.
ssd = SSD(emb, df["word"], use_full_doc=True)
show("ssd", ssd)
show("ssd.n_kept", ssd.n_kept)

print("\n====== PCA+OLS: K chosen by the PCA-K sweep ======")
# The outcome is a column of `data`. K is picked by the sweep between k_min and
# k_max; fixed_k=N would skip the sweep.
res = ssd.fit_regression("valence", data=df)
show("res", res)
show("res.summary()", res.summary())
show("res.stats", res.stats)
show("res.test", res.test)
show("res.words", res.words)
show("res.clusters", res.clusters)
show("res.clusters.pos.words", res.clusters.pos.words)
show("res.clusters.neg.words", res.clusters.neg.words)
show("res.snippets", res.snippets)

print("\n====== Tour: what every result has ======")
show("res.report()", res.report())

# save_page writes one self-contained HTML file that opens from disk.
page = os.path.join(OUT, "warriner_regression.html")
res.save_page(page, title="Warriner valence, PCA+OLS")
print(f"\nsaved {page}")

# A view saves itself; the extension picks the format (.csv .json .xlsx .md .txt .html .tex .docx).
words_csv = os.path.join(OUT, "words.csv")
res.words.save(words_csv)
print(f"saved {words_csv}")

# predict takes new documents in the form of the fit's text: strings here.
show('res.predict(["joy", "grief", "table"])', res.predict(["joy", "grief", "table"]))

# A wordpack embedding is not pickled with the result: attach it again after loading.
restored = pickle.loads(pickle.dumps(res))
restored.attach(embeddings=emb)
show("restored.stats", restored.stats)
