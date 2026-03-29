# SSDLite — Supervised Semantic Differential

Rewrite and overhaul of the [`ssdiff`](https://github.com/hplisiecki/Supervised-Semantic-Differential) package — an NLP method for finding interpretable semantic dimensions in text linked to numeric outcomes or categorical groups.

## Quick Start

```python
from ssdlite import Embeddings, Corpus, SSD

# Load embeddings and build corpus
emb = Embeddings.load("vectors.ssdembed")
corpus = Corpus(texts, lang="pl")

# Continuous outcome → SSD + PLS
ssd = SSD(emb, corpus, y, lexicon=["word1", "word2", ...])
result = ssd.fit_pls()      # or ssd.fit_ols() for PCA+OLS
```

## Interpreting Results

### Summary & effect sizes

```python
print(result.summary())
# SSD Model Summary (PLS)
# ───────────────────────
# Backend: PLS (3 components)
# Docs:  487 kept / 512 total (25 dropped)
# R² = 0.2341   R²_adj = 0.2293
# ...
# Effect sizes:
#   ‖β‖ (SD(y) per +1.0 cos) = 0.4821
#   Δy per +0.10 cos         = 0.1932
#   IQR(cos) effect on y     = 0.3104
#   Corr(y, ŷ)               = 0.4838

# Individual attributes
result.r2, result.pvalue
result.beta_norm             # ‖β‖ in SD(y) units
result.delta                 # raw y change per +0.10 cosine
result.iqr_effect            # raw y change across IQR of cosine
result.y_corr_pred           # |corr(y, ŷ)|
result.cos_align             # per-doc cosine to β (array)
```

### Top words & clustering

```python
result.top_words(n=20)       # → list[dict] with {side, rank, word, cos}
result.neighbors("pos", n=20)
result.cluster_neighbors("pos", topn=100)
```

### Document-level scores

```python
result.doc_scores()          # → dict with keep_mask, cos_align, score_std, yhat_raw
```

### Extreme documents

```python
# Top/bottom docs by predicted or observed outcome
result.extreme_docs(k=50, by="predicted")   # → list[dict]
result.extreme_docs(k=50, by="observed")

# Text snippets from extreme docs
result.snippets_extreme(corpus.pre_docs, k=50, by="predicted")
```

### Misdiagnosed documents

Documents where the model is most wrong:

```python
result.misdiagnosed(k=20)                  # both over- and under-predicted
result.misdiagnosed(k=20, side="over")     # model over-predicts
result.misdiagnosed(k=20, side="under")    # model under-predicts
# → list[dict] with {idx, y_true, yhat, cos, residual, side}
```

### Split-half significance test (PLS only)

```python
st = result.split_test(n_splits=50, seed=42)          # method="split" (default)
st = result.split_test(n_splits=50, method="split_cal") # permutation-calibrated
st["pvalue"]    # p-value
st["mean_r"]    # mean Pearson r across splits
```

## Return Types

All methods return plain Python types — no pandas dependency:
- `summary()` → `str`
- `top_words()`, `extreme_docs()`, `misdiagnosed()`, `results_table()` → `list[dict]`
- `doc_scores()` → `dict` of arrays

To get a DataFrame: `pd.DataFrame(result.extreme_docs())`.

## Reference

Plisiecki et al. (2025). *Supervised Semantic Differential.* PsyArXiv. [doi:10.31234/osf.io/gvrsb_v1](https://doi.org/10.31234/osf.io/gvrsb_v1)
