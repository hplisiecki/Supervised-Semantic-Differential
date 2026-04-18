# ssdiff — Results Guide

Everything you can do with objects returned by:

- `SSD.fit_pls()` → `PLSResult`
- `SSD.fit_ols()` → `PCAOLSResult`
- `SSD.fit_groups()` → `GroupResult`
- `Corpus.suggest_lexicon()` / `.evaluate_lexicon()` → `LexiconResult`

All of them share the same shape: **scalar stats, data views, a textual report, and a unified `save()` on every view/report**.

---

## Quick tour

```python
result = ssd.fit_pls()

# Scalar summaries
result.stats                 # r², p-value, n_kept, …
result.fit_info              # PLS/PCA+OLS fit hyperparameters

# Tabular views — iterable, sliceable, exportable
result.words                 # top words (β-pos + β-neg)
result.words.pos             # top 20 positive-side words
result.words.pos(50)         # top 50 positive-side words
result.words.neg             # top 20 negative-side words
result.clusters.pos          # positive-side clusters (default topn=100)
result.clusters.pos(topn=50) # recompute with topn=50
result.snippets              # representative snippets (default top_per_side=30)
result.snippets(top_per_side=200)
result.docs                  # one row per document (y, ŷ, residual, alignment)

# Per-cluster joins
cl = result.clusters.pos
cl.words(cluster_id=0)       # words inside cluster 0
cl.snippets                  # all positive-side snippets — SidedSnippetsView
cl.snippets(cluster_id=0)    # snippets seeded inside cluster 0

# Ordering docs
result.docs.pos(10)          # 10 docs most aligned with β-pos (highest ŷ)
result.docs.neg(10)          # 10 docs most aligned with β-neg (lowest ŷ)
result.docs.misdiagnosed(5)  # 5 docs with largest |residual|
result.docs.misdiagnosed(5, direction="over")   # model over-predicted
result.docs.misdiagnosed(5, direction="under")  # model under-predicted
result.docs.id(42)           # single doc + its original text (if corpus attached)

# Multi-format narrative report
r = result.report(top_words=5, clusters=30, extreme_docs=5, misdiagnosed=5)
print(r)                     # plain text (also auto-rendered in Jupyter as HTML)
r.save("report.md")          # dispatched on extension:
r.save("report.html")        #   .md .html .docx .tex .txt .json
r.save("report.docx", style="APA")
```

---

## Result types at a glance

| Returned by | Class | Key views |
|---|---|---|
| `SSD.fit_pls()` | `PLSResult` | `stats`, `fit_info`, `test`, `words`, `clusters`, `snippets`, `docs` |
| `SSD.fit_ols()` | `PCAOLSResult` | same + `sweep`, `plot_sweep()` |
| `SSD.fit_groups()` | `GroupResult` | `stats`, `test`, `pairs` (→ `PairView` per contrast) |
| `Corpus.suggest_lexicon()` | `LexiconResult` | `stats`, `suggestions`, `tokens` |
| `Corpus.evaluate_lexicon()` | `LexiconResult` | same + `summary` |

A `PairView` from `gr.pairs["A", "B"]` looks just like a continuous result — it has `.stats`, `.words`, `.clusters`, `.snippets` — so everything you learn here transfers.

---

## Working with views

Every tabular view (`words`, `clusters.pos`, `snippets`, `docs`, `pairs`, `suggestions`, …) supports the same operations:

```python
len(view)              # row count
view[0]                # first row (a frozen dataclass — Word, Cluster, …)
view[:10]              # slice → a view without truncation
list(view)             # materialize to a list of rows
for row in view: ...   # iterate

# Display (`print(view)` in terminal, auto-rendered in Jupyter)
print(view)            # aligned text table with save-hint footer
# to_text() / to_html() are used internally for repr — for file output use save()

# Python values (no file I/O)
view.to_df()           # pandas DataFrame     (needs ssdiff[results])
view.to_dict()         # list[dict]           (always works, no deps)
view.to_records()      # list[tuple]

# File output — one method, extension picks the format
view.save("x.csv")     # .csv .json .xlsx .md .txt .html .tex .docx
view.save("x.xlsx")    # xlsx/docx need ssdiff[results]
view.save("x.md", cols=["word", "cos_beta"])   # optional column subset & order
view.save("x.csv", k=50)                        # optional row cap on size-bearing views
```

Scalar views (`stats`, `fit_info`, `test`, `summary`) behave like dicts too — `stats.r2`, `stats["r2"]`, or `stats.to_dict()`.

### What gets printed in the terminal

- Long views auto-truncate to ~20 rows with a `... N more rows` footer.
- Long text cells (e.g. `snippet.text_window`) are clipped to ~40 chars in the terminal. Exports (`to_df`, `save(...)`, …) keep the full text.
- P-values render as `<.001` or three decimals (`.007`) across all formats — never as `2.36e-07`.
- A short "Save:" hint line is appended so you can copy-paste an export command.

### Changing how many rows you get

Any view with a row-count knob is **callable**. The same shape, the same effect on display AND export — slicing the view first means `save(...)`, `to_df`, etc. return fewer rows too.

| View | Default | Change via | Also applies to export? |
|---|---|---|---|
| `result.words.pos` / `.neg` | 20 | `words.pos(50)` (or `words.pos(None)` for all) | yes |
| `result.docs.pos()` / `.neg()` / `.misdiagnosed()` | 20 | `docs.pos(50)` | yes |
| `result.clusters.pos` / `.neg` | topn=100 | `clusters.pos(topn=50)` (re-clusters) | yes |
| `result.snippets` | top_per_side=30 | `result.snippets(top_per_side=200)` (recomputes) | yes |
| `result.report(...)` | see below | `report(top_words=10, clusters=30, …)` | report only |

```python
# Saving 50 positive words to CSV (not 20, not all):
result.words.pos(50).save("top50_pos.csv")

# Equivalent: use k= on save() to cap rows directly
result.words.pos.save("top50_pos.csv", k=50)

# Saving every available word:
result.words.pos(None).save("all_pos.csv")

# Saving 10 worst-predicted docs as DataFrame:
result.docs.misdiagnosed(10).to_df()
```

**How to discover the knob on any view.** When you `print(view)` in a terminal, the last few lines are a "Save:" hint that shows the resize idiom for that view, with the current value. Example:

```
Count: .pos(k) → first k (current 20, max 100; k=None for all)
```

Cluster and snippet variants cache separately by parameter set, so recomputing with different `topn` / `top_per_side` doesn't overwrite the earlier view:

```python
result.clusters.pos                       # cached (topn=100)
result.clusters.pos(topn=50)              # separate cache entry
result.clusters.pos                       # still the original

result.clear_cache()                      # drop everything
result.clear_cache("clusters")            # drop all cluster variants
```

---

## Domain rows

Iterating a view yields typed, immutable dataclasses (not dicts) — your IDE autocompletes their fields.

```python
Word         side ("pos"|"neg"), rank, word, cos_beta, contrast
Cluster      cluster_id, side, size, coherence, centroid_cos_beta, contrast
ClusterWord  cluster_id, side, word, cos_centroid, cos_beta, contrast
Snippet      snippet_id, side, doc_id, cosine, seed,
             start_token_idx, end_token_idx, start_sent_idx, end_sent_idx,
             text_window, text_surface, text_lemmas,
             cluster_id, contrast, profile_id, post_id
Doc          doc_id, y_true, y_hat, residual, cos_align
Pair         contrast, g1, g2, T, p_raw, p_corrected, cohens_d,
             n_g1, n_g2, contrast_norm
Suggestion   rank, token, freq, cov_all, cov_bal, corr, pvalue, direction
             direction ∈ {"positive", "negative", "none"}
```

**Terminology convention**

- `side` uses `"pos"` / `"neg"` for β-direction (positive pole = higher y_hat).
- `direction` on `misdiagnosed()` uses `"over"` / `"under"` for residual sign — distinct axis from the β direction.

---

## Reports

`result.report(...)` returns a `Report` object with the same text/markdown/html/docx/latex renderers as views.

```python
r = result.report(
    top_words=5,             # top N words per pole (default 5)
    clusters=30,              # include N clusters per side (default: off)
    snippets_per_cluster=3,   # anchor snippets per cluster (default: off)
    extreme_docs=5,           # N most-positive + N most-negative docs
    misdiagnosed=5,           # N most over-predicted + N most under-predicted
)

print(r)                      # text (also auto-renders as HTML in Jupyter)

r.save("report.md")           # extension dispatch:
r.save("report.html")         #   .md  .txt  .html
r.save("report.tex")          #   .tex .json
r.save("report.docx", style="APA")   # style= is only valid for .docx
```

`GroupResult.report()` loops over pairwise contrasts for you.  
`LexiconResult.report(top=20)` renders the suggestions table + optional coverage summary.

---

## Exporting tables

Each view exports itself. There is no bulk `result.save()` — call `save()` on the table you want.

```python
result.stats.save("stats.csv")
result.docs.save("docs.csv")
result.words.save("words.xlsx")
result.clusters.pos.save("clusters_pos.json")
result.snippets.save("snippets.md", cols=["doc_id", "text_window"])
```

Every view accepts the same kwargs:

- `cols=[...]` — select & reorder columns (unknown names warn and are dropped)
- `k=N` — cap rows on size-bearing views (ignored on single-row scalar views)
- `style="APA"` — only valid for `.docx` output

For a narrative bundle, use `result.report().save(...)` — see the [Reports](#reports) section.

---

## Attach / detach

Some views need the source corpus or embeddings to compute:

```python
result.words       # needs embeddings
result.clusters    # needs embeddings
result.snippets    # needs both corpus and embeddings
result.docs.id(42) # corpus attachment unlocks the raw text

result.attach(corpus=my_corpus, embeddings=my_embeddings)
```

Accessing a view without its resource raises a clear `RuntimeError` telling you to call `attach(...)`.

---

## Result-specific extras

### `PLSResult`

```python
result.fit_info              # n_components, pca_k, p_method, n_perm, random_state, …
result.test                  # current p-value test (perm / split / split_cal)
result.test(name="perm", n_perm=5000)    # rerun; updates stats.pvalue too
result.cv_scores             # cv R² by n_components (dict, or None)
result.perm_null             # null-distribution array (or None)
```

### `PCAOLSResult`

```python
result.sweep                 # SweepView: k → (r², r²_adj, p-value)
result.sweep_result          # raw sweep_result object (or None)
result.plot_sweep("sweep.png")      # dual-axis chart; raises if no sweep data
result.test                  # F-test
```

### `GroupResult`

```python
gr.stats                     # G, n_kept, n_perm, correction, random_state, pvalue
gr.test                      # omnibus permutation test — rerun with gr.test(n_perm=...)
gr.pairs                     # list of Pair rows (iterable + exportable)
gr.pairs["A", "B"]           # PairView — reverse order flips T/d signs automatically
gr.pairs["A", "B"].stats     # per-contrast stats
gr.pairs["A", "B"].words     # per-contrast words
gr.pairs["A", "B"].clusters.pos
gr.pairs["A", "B"].snippets
```

### `LexiconResult`

```python
lex.stats                    # var_type, n_docs, n_tokens
lex.suggestions              # token-level rows (cov_bal, corr, pvalue, direction, rank)
lex.tokens                   # list[str] in rank order
lex.summary                  # coverage block — present only after evaluate_lexicon()
```

---

## Optional dependencies

The core library has no hard requirement on pandas / openpyxl / python-docx. Install the extra when you need them:

```bash
pip install ssdiff[results]
```

Without the extra you can still:

- iterate views and inspect rows,
- call `to_dict()`, `to_records()`,
- `save(...)` to `.csv`, `.json`, `.md`, `.tex`, `.txt`, `.html` — no optional deps needed,
- render reports as text / markdown / html / latex / json.

`save('x.xlsx')`, `save('x.docx')`, and `to_df()` need the extra and raise a clear `ImportError` with the install hint when it's missing.

---

## Cheat sheet

| Task | Code |
|---|---|
| Print a summary of the fit | `print(result)` |
| Full narrative report (terminal) | `print(result.report())` |
| Save report as DOCX / Markdown | `result.report().save("r.docx", style="APA")` / `"r.md"` |
| Top 20 positive words as DataFrame | `result.words.pos.to_df()` *(20 is the default)* |
| Top 50 positive words to CSV | `result.words.pos(50).save("w.csv")` |
| Save each table separately | `result.stats.save("stats.csv")`, `result.docs.save("docs.csv")`, … |
| Single table to Excel | `result.words.save("words.xlsx")` |
| 10 worst-predicted docs | `result.docs.misdiagnosed(10)` |
| 10 docs the model over-predicted | `result.docs.misdiagnosed(10, direction="over")` |
| Snippets inside cluster 3 (pos side) | `result.clusters.pos.snippets(cluster_id=3)` |
| Rerun PLS permutation test | `result.test(name="perm", n_perm=5000)` |
| Per-contrast view in group mode | `gr.pairs["A", "B"]` |

---

## See also

- [`api_reference.md`](api_reference.md) — fitting, embeddings, corpus
- [`architecture.md`](architecture.md) — backends and statistical methods
- [`demo_new_api.py`](demo_new_api.py) — runnable continuous example
- [`demo_lexicon_api.py`](demo_lexicon_api.py) — runnable lexicon example
