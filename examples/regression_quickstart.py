"""Regression in one line: every part of the pipeline at its default.

``ssdiff.fit_regression("score ~ women + ssd(text)", df)`` names the text in the
formula as ``ssd(text)``. No text is passed, so ssdiff builds it from the column
``text`` of ``df`` as ``SSD(df["text"])``: the default English wordpack, and
each document's vector from all its words (there is no lexicon). K is chosen by
the PCA sweep. A text that needs other settings, such as a lexicon or another
wordpack, is built with ``SSD(...)`` and passed by keyword: ``text=SSD(...)``.

The result reads like a statsmodels regression result. ``params``, ``bse`` and
``conf_int()`` hold the covariates. A text term has K component weights and no
single estimate, so it appears only in ``pvalues``, as the Wald test of its
weights.

Data: the Measuring Hate Speech comments (Kennedy et al., 2020). Each comment is
one document, its hate-speech score is the outcome, and ``women`` is 1 when the
annotators said the comment targets women.

The default wordpack is loaded with ``Embeddings.load()``. The first call
downloads it (about 380 MB) into ``~/.wordpacks`` (or ``WORDPACK_CACHE``).
"""

import os

import pandas as pd

import ssdiff

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "data", "measuring_hate_speech.csv.gz"))

res = ssdiff.fit_regression("score ~ women + ssd(text)", df)
res.summary()

# The covariates, as in statsmodels.
print(res.params)
print(res.conf_int())

# The covariates' p-values, then one row for the text term ssd(text).
print(res.pvalues)

# The whole model against score ~ 1 (an F-test, since this model is OLS).
print(res.fvalue, res.f_pvalue, res.rsquared)

# The text's own views: the words at each end of the gradient.
print(res.words)
