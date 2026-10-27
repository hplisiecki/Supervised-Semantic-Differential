"""Saving results: every view and the report, in every format.

Each view of a result has ``.save(path)``, and the extension picks the format:

- ``.docx``: a Word table laid out for a paper's results section (an italic
  caption above the table, rules above and below the header and under the
  last row, no vertical lines, Times New Roman 11 pt).
- ``.xlsx``, ``.csv``, ``.json``: the data, for further analysis.
- ``.html``, ``.md``, ``.tex``, ``.txt``: a table for a web page, a Markdown
  or LaTeX document, or plain text.

``k=`` keeps the first k rows and ``cols=`` picks the columns. ``clusters`` has
a paper layout of its own in ``.docx``: one table per end of the gradient with
No., Size, the 10 closest words and a representative excerpt. Every other view
is saved as it stands: its column names as in code (``cos_beta``, ``y_true``)
and its name as the caption.

``report()`` puts the main tables in one document (``.docx``, ``.html``,
``.md``, ``.tex``, ``.txt``, ``.json``). ``save_page()`` writes an interactive
HTML results page that opens offline.

Data: the Measuring Hate Speech comments (Kennedy et al., 2020), as in
regression_quickstart.py. Each comment is one document, its hate-speech score
is the outcome, and ``women`` is 1 when the annotators said the comment targets
women.

The default wordpack is loaded with ``Embeddings.load()``. The first call
downloads it (about 380 MB) into ``~/.wordpacks`` (or ``WORDPACK_CACHE``).
``.docx`` and ``.xlsx`` need ``pip install ssdiff[results]``. The files are
written to the current folder.
"""

import os

import pandas as pd

import ssdiff

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "data", "measuring_hate_speech.csv.gz"))

res = ssdiff.fit_regression("score ~ women + ssd(text)", df)

# ---- The report: the main tables in one document ----
# Stats, fit info, the clusters of each end, top words, and the most extreme
# and most misdiagnosed documents. clusters={"n_words": 10} lists 10 words per
# cluster, as the .docx clusters table does.
report = res.report(clusters={"n_words": 10})
report.save("report.docx")
report.save("report.html")
report.save("report.md")
report.save("report.tex")

# ---- The results page: everything, to browse ----
res.save_page("results.html")

# ---- Clusters ----
# .docx: the paper layout, both ends in one file. The captions name the
# outcome: "Positive semantic clusters (+β̂ → higher score)".
res.clusters.save("clusters.docx")
# The other formats hold the cluster table itself.
res.clusters.save("clusters.xlsx")
res.clusters.save("clusters.html")
# .pos / .neg save one end; k=5 keeps its first five clusters.
res.clusters.pos.save("clusters_pos.docx", k=5)

# ---- Words ----
# res.words holds every word of both ends. .pos(10) is the 10 words closest to
# the positive end; cols= drops the columns a paper does not need.
res.words.pos(10).save("words_pos.docx", cols=["rank", "word", "cos_beta"])
res.words.neg(10).save("words_neg.docx", cols=["rank", "word", "cos_beta"])
# All words of both ends, as data.
res.words.save("words.xlsx")
res.words.save("words.csv")

# ---- Model fit and tests ----
# One-row views: one row in the table, one column per value.
res.stats.save("stats.docx")
res.stats.save("stats.json")
res.fit_info.save("fit_info.docx")
# One row for all text terms, then one Wald test per text term.
res.test.save("test.docx")
res.test.save("test.html")

# ---- Documents and snippets ----
# docs has one row per kept document: all of them as data, the first 20 for a
# paper.
res.docs.save("docs.xlsx")
res.docs.save("docs.docx", k=20, cols=["doc_id", "y_true", "y_hat", "alignment_score"])
# Snippets: the text around the words at each end.
res.snippets.pos.save("snippets_pos.docx", k=10)
res.snippets.save("snippets.xlsx")
