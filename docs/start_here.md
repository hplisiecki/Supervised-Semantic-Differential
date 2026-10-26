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
| [`../examples/regression_quickstart.py`](../examples/regression_quickstart.py) | `ssdiff.fit_regression("score ~ women + ssd(text)", df)`: the text built from a column with every default, and the statsmodels names of the result. |
| [`../examples/regression.py`](../examples/regression.py) | `fit_regression`: the text as a predictor, alone and with a covariate. |
| [`../examples/regression_interaction.py`](../examples/regression_interaction.py) | An interaction with the text, `score ~ women * ssd()`, its tests, and `contrast`. |
| [`../examples/pls.py`](../examples/pls.py) | `fit_pls`: alone, with a covariate, and with several rotated gradients. |
| [`../examples/lexicon_analysis.py`](../examples/lexicon_analysis.py) | Build a lexicon with the lexicon helpers, then fit on the lexicon contexts. |
| [`../examples/several_outcomes.py`](../examples/several_outcomes.py) | `fit_multipls`: one text against several outcomes. |
| [`../examples/between_group_differences.py`](../examples/between_group_differences.py) | `fit_groups`: how the texts of labelled groups differ. |
| [`../examples/custom_embedding.py`](../examples/custom_embedding.py) | Your own embedding file and token lists made with spaCy. |


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
ssd = SSD(df["text"], embeddings=emb, lexicon=["happy", "sad", "joy"])   # one text, no outcome

result = ssd.fit_regression("score", data=df)   # PCA regression, F-test p-value
# result = ssd.fit_regression("score ~ age + sex*ssd()", data=df)  # covariates and interactions
# result = ssd.fit_pls("score", data=df)        # PLS with split-half test
# result = ssd.fit_groups("condition", data=df) # categorical outcome, permutation test
# result = ssd.fit_multipls(["a", "b"], data=df) # several outcomes, PLS3

print(result.stats)             # headline r², p, n_kept, iqr_effect
print(result.words.pos)         # top 20 β-pos neighbors
result.report().save("report.md")
```

The outcome is the first argument of each fit: values, a column name of `data`, or a formula. Several texts in one model use `ssdiff.fit_regression("score ~ ssd(essay) + ssd(diary)", df)`; each `ssd(name)` is a text passed as a keyword (`essay=SSD(...)`), else the column `name` of `df` with every default. Two fitted gradients are compared with `ssdiff.contrast(g1, g2, emb)`.

See [`../examples/pls.py`](../examples/pls.py) for a runnable end-to-end script, and [`../examples/several_outcomes.py`](../examples/several_outcomes.py) for `fit_multipls`.

---

## Citing

> Plisiecki, H., Lenartowicz, P., Pokropek, A., Małyska, K., & Flakus, M. (2025).
> *Supervised Semantic Differential*. PsyArXiv. <https://doi.org/10.31234/osf.io/gvrsb_v1>
