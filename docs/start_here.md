# ssdiff — Documentation Index

`ssdiff` is the reference Python implementation of the **Supervised Semantic Differential** (Plisiecki et al., 2025) — a method for finding interpretable semantic dimensions in text data that link to numeric outcomes or categorical groups.

This folder holds four tiers of documentation plus a runnable demo. Start wherever fits your need.

---

## Pick your entry point

### For users — getting things done

| Read this | When you want to… |
|---|---|
| [`api_reference.md`](api_reference.md) | Fit a model. Covers `Embeddings`, `Corpus`, `SSD`, and the fit methods (`fit_ols`, `fit_pls`, `fit_groups`, plus the in-development `fit_multipls`) with argument tables and a full workflow example. Results are only sketched here. |
| [`../examples/demo_api.py`](../examples/demo_api.py) | See the whole pipeline in ~50 lines — load embeddings, build a corpus, fit PLS / PCA+OLS / groups, print stats, export a report. |
| [`../examples/demo_multipls.py`](../examples/demo_multipls.py) | Minimal runnable example for the in-development `fit_multipls` (rotated multi-component PLS). |


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
# result = ssd.fit_multipls(k=2)  # rotated multi-component PLS — in development

print(result.stats)             # headline r², p, n_kept, iqr_effect
print(result.words.pos)         # top 20 β-pos neighbors
result.report().save("report.md")
```

See [`../examples/demo_api.py`](../examples/demo_api.py) for a runnable end-to-end script, and [`../examples/demo_multipls.py`](../examples/demo_multipls.py) for the rotated multi-component variant.

---

## Low-RAM mode

For machines that cannot fit the full embedding matrix in RAM (Colab free
tier, 8 GB laptops, etc.), pass `ram_efficient=True` to
`Embeddings.load`. Only an uncompressed `.ssdembed` file works in this
mode — convert other formats once with the snippet below.

```python
# One-time: convert any format to .ssdembed and pre-normalise.
emb = Embeddings.load("model.bin").normalize(l2=True, abtt=1)
emb.save("model_norm")  # → model_norm.ssdembed

# Each subsequent run:
emb = Embeddings.load("model_norm.ssdembed", ram_efficient=True)
emb.attach_corpus(corpus)
ssd = SSD(emb, corpus, y, lexicon).fit_pls()
```

RAM mode is read-only: `normalize`, `save`, and `SSD.fit_multipls` raise.
For the full PLS / PCA+OLS / group-comparison pipeline this is enough —
`fit_multipls` is the only fit method that needs the full vocabulary as a
rotation target.

---

## Citing

> Plisiecki, H., Lenartowicz, P., Pokropek, A., Małyska, K., & Flakus, M. (2025).
> *Supervised Semantic Differential*. PsyArXiv. <https://doi.org/10.31234/osf.io/gvrsb_v1>
