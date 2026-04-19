# ssdiff — Documentation Index

`ssdiff` is the reference Python implementation of the **Supervised Semantic Differential** (Plisiecki et al., 2025) — a method for finding interpretable semantic dimensions in text data that link to numeric outcomes or categorical groups.

This folder holds four tiers of documentation plus a runnable demo. Start wherever fits your need.

---

## Pick your entry point

### For users — getting things done

| Read this | When you want to… |
|---|---|
| [`api_reference.md`](api_reference.md) | Fit a model. It covers `Embeddings`, `Corpus`, `SSD`, and the three fit methods (`fit_ols`, `fit_pls`, `fit_groups`) with argument tables and a full workflow example. Results are only sketched here. |
| [`demo_new_api.py`](demo_new_api.py) | See the whole pipeline in ~50 lines — load embeddings, build a corpus, fit PLS / PCA+OLS / groups, print stats, export a report. |

### For power users — making results do what you want

| Read this | When you want to… |
|---|---|
| [`results.md`](results.md) | Work with result objects in depth. Every view, every method, every helper — `to_df` / `to_dict` / `to_records` / `save`, rerunning statistical tests, the parameter-keyed cache, `attach()` after un-pickling, building custom reports. |
| [`results_tables.md`](results_tables.md) | Change what's printed or exported column-by-column. Lists every view's columns with the default subset marked, plus how to override defaults globally (`DEFAULT_COLS`) or per-call (`cols="all"`, `cols=[...]`), and how to resize tables (`words.pos(50)`, `save(..., k=N)`). |

### For developers — reading the code

| Read this | When you want to… |
|---|---|
| [`architecture.md`](architecture.md) | Understand the internals. Module layout, the `SSD → backend → Result` pipeline, how views compose on top of `Result` / `View` / `ScalarView` / `TestView`, the parameter-keyed cache, the significance tests, and a full pipeline diagram. |

---

## Also in this folder

- [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) — current rough edges and open items.
- [`sweep_plot.png`](sweep_plot.png) — example PCA+OLS sweep chart.

---

## One-minute orientation

```python
from ssdiff import Embeddings, Corpus, SSD

emb    = Embeddings.load("glove.ssdembed")
corpus = Corpus(texts, lang="en")
ssd    = SSD(emb, corpus, y=scores, lexicon=["happy", "sad", "joy"])

result = ssd.fit_ols()          # PCA + OLS, F-test p-value
# result = ssd.fit_pls()        # PLS with split-half test
# result = ssd.fit_groups()     # categorical outcome, permutation test

print(result.stats)             # headline r², p, n_kept, iqr_effect
print(result.words.pos)         # top 20 β-pos neighbors
result.report().save("report.md")
```

See [`demo_new_api.py`](demo_new_api.py) for a runnable script with a real dataset.

---

## Citing

> Plisiecki, H., Lenartowicz, P., Pokropek, A., Małyska, K., & Flakus, M. (2025).
> *Supervised Semantic Differential*. PsyArXiv. <https://doi.org/10.31234/osf.io/gvrsb_v1>
