# ssdiff — Documentation Index

`ssdiff` is the reference Python implementation of the **Supervised Semantic Differential** (Plisiecki et al., 2025) — a method for finding interpretable semantic dimensions in text data that link to numeric outcomes or categorical groups.

This folder holds three tiers of documentation; runnable examples are in [`../examples/`](../examples/). Start wherever fits your need.

---

## Pick your entry point

### For users — getting things done

| Read this | When you want to… |
|---|---|
| [`embeddings.md`](embeddings.md) | Get embeddings and texts ready. The ready English packs (`Embeddings.load()`), tags and full names, where packs are stored, and loading your own embedding file with token lists for another language. |
| [`api_reference.md`](api_reference.md) | Fit a model. Covers `Embeddings`, `SSD`, the fit methods (`fit_regression`, `fit_pls`, `fit_groups`, `fit_multipls`), formulas with the text term `ssd()`, several texts, `contrast`, and a full workflow example. Results are only sketched here. |
| [`../examples/warriner_regression.py`](../examples/warriner_regression.py) | Runnable example for `fit_regression`, with a tour of what every result has. |
| [`../examples/warriner_pls.py`](../examples/warriner_pls.py) | Runnable example for `fit_pls`. |
| [`../examples/hate_lexicon_regression.py`](../examples/hate_lexicon_regression.py) | Build a lexicon with the lexicon helpers, then `fit_regression`. |
| [`../examples/hate_multipls.py`](../examples/hate_multipls.py) | Runnable example for `fit_multipls`. |
| [`../examples/hate_groups.py`](../examples/hate_groups.py) | Runnable example for `fit_groups`. |
| [`../examples/custom_embedding_spacy.py`](../examples/custom_embedding_spacy.py) | Your own embedding file and token lists made with spaCy. |


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

- [`sweep_plot.png`](sweep_plot.png) — example PCA+OLS sweep chart.

---

## One-minute orientation

```python
from ssdiff import Embeddings, SSD

emb = Embeddings.load()                    # the default English wordpack, downloaded once
ssd = SSD(emb, df["text"], lexicon=["happy", "sad", "joy"])   # one text, no outcome

result = ssd.fit_regression("score", data=df)   # PCA regression, F-test p-value
# result = ssd.fit_regression("score ~ age + sex*ssd()", data=df)  # covariates and interactions
# result = ssd.fit_pls("score", data=df)        # PLS with split-half test
# result = ssd.fit_groups("condition", data=df) # categorical outcome, permutation test
# result = ssd.fit_multipls(["a", "b"], data=df) # several outcomes, PLS3

print(result.stats)             # headline r², p, n_kept, iqr_effect
print(result.words.pos)         # top 20 β-pos neighbors
result.report().save("report.md")
```

The outcome is the first argument of each fit: values, a column name of `data`, or a formula. Several texts in one model use `ssdiff.fit_regression("score ~ ssd(essay) + ssd(diary)", df, essay=..., diary=...)`. Two fitted gradients are compared with `ssdiff.contrast(g1, g2, emb)`.

See [`../examples/warriner_pls.py`](../examples/warriner_pls.py) for a runnable end-to-end script, and [`../examples/hate_multipls.py`](../examples/hate_multipls.py) for `fit_multipls`.

---

## Citing

> Plisiecki, H., Lenartowicz, P., Pokropek, A., Małyska, K., & Flakus, M. (2025).
> *Supervised Semantic Differential*. PsyArXiv. <https://doi.org/10.31234/osf.io/gvrsb_v1>
