# Changelog

## 3.3.0

- `fit_ols()` is renamed `fit_regression()`. The old name still works with the same arguments and warns with `FutureWarning`. Without a formula the results are the same as before.
- `fit_regression(formula, family=..., link=...)` controls for covariates and nesting: `fit_regression("~ age + gender + (1 | ID)")`. K is still chosen by the PCA sweep; the final model at that K is fit by `glmm`, with the K component scores added to the formula. The p-value tests the components together (F-test for OLS and for Gamma and inverse-Gaussian GLMs, likelihood-ratio test otherwise), and `result.model` holds the glmm fit. `predict()` is not available after a fit with a formula or a non-Gaussian family.
- `fit_pls(formula)` controls for fixed covariates: `fit_pls("~ age + gender")` regresses y and every column of the document vectors on the formula by OLS and runs PLS and its test on the residuals (Kennedy permutation). Random effects raise `ValueError`.
- New `result.covariates` on `fit_regression()` results with a formula: the formula's coefficients (estimate, SE, t or z, p, 95% CI) with the components held fixed, also shown on the results page.
- New `result.summary()` on `fit_pls()` and `fit_regression()` results prints and returns a compact summary. For `fit_regression()`: glmm's summary without the text's own `_ssd_*` rows, plus an SSD effects row with the text's p and ΔR² (OLS only); `result.model.summary()` still prints the full glmm summary. For `fit_pls()`: the text's test and its R² (partial R² with a formula).
- `fit_regression()` results' `test` gains the columns `statistic`, `df`, `df_resid` and `r2_delta` (the R² the components add, OLS only).
- `SSD(..., data=df)` takes a table with one row per document; `y` can then be a column name, or a list of column names for `fit_multipls()`.
- `glmm` (>= 0.6.0) is now a dependency.
- `token_stats`, `coverage_summary`, `evaluate_lexicon` and `suggest_lexicon(seeds=...)` take `y=None`: they then return only the statistics that need no outcome, with the outcome fields `None`.
- New `result.save_page(path)` on `fit_pls()`, `fit_regression()`, `fit_groups()` and `fit_multipls()` results writes one HTML file with the result's tables that opens in a browser from disk, with no network. `result.page_data()` returns the same content as a dict.
- New `fit_pls(ssd_gradients=N)` also returns N rotated gradients from one outcome, as leaves `result["dim-1"]` … `result["dim-N"]` with `.stats` (`score_corr`, `cov_share`), `.words`, `.clusters` and `.snippets`; `rotate="words"` (default) or `None`, rotation diagnostics in `result.pls_info`. The standard fit (p-value, `beta`, `r2`, `docs`, `predict()`) does not change. `report()` and `save_page()` show the gradients. `fit_info` gains `ssd_gradients` and `rotate`.
- `fit_multipls(k=...)` is now `fit_multipls(ssd_gradients=...)`, and it defaults to the number of outcomes (with `outcome_groups`, to the number of rotation target rows). The `k` column of `stats` and `pls_info` is now `ssd_gradients`. `rotate="outcomes"` now works with every outcome kept: each dimension then loads on one outcome.
- Sparse PLS is removed: `fit_pls(keep=...)`, the `"split_nb_seq"` test and `fit_info.keep` are gone.

## 3.2.0

- spaCy is no longer installed by default: `pip install ssdiff[spacy]` to keep using `Corpus(texts, lang=...)`. New `.wordpack` embeddings tokenize without spaCy: `Corpus(texts, embeddings=Embeddings.load("en"))`.
- `Embeddings.load` takes a wordpack's tag (`"en"`) or full name (`"en-glove2024-dolma-1"`): the pack is downloaded once into a cache directory and loaded from there. With no argument it loads the tag `"default"`, the same pack as `"en"`. `report()` and `corpus.tokenizer` record the full name.
- The `.ssdembed` format is deprecated: loading one raises an error, and `Embeddings.save` no longer writes it (`fmt` now defaults to `"bin"`).
- `Embeddings.load` no longer takes `ram_efficient`. A `.wordpack` always keeps only its 50,000 most frequent rows in RAM.
- `Embeddings.normalize` no longer takes `re_normalize`: rows are always L2-normalized again after ABTT, and `l2_normalized` then reports it.
- New `result.predict(texts)` on `fit_pls()` and `fit_ols()` results: the predicted outcome for new documents, given as a `Corpus` or, for a fit on a wordpack, as raw texts.
- `fit_pls(test=False)` skips the significance test; `pvalue` is then `NaN`.
- `SSD(sif_a=None)` gives every word weight 1 (plain mean, no SIF weighting).
- `docs.y_hat` now subtracts the training mean of the document vectors before applying `beta`, so `y_hat` and `residual` differ from 3.1.0 by a constant for the same fit.
- `ssd.n_raw` now counts every corpus document and `ssd.n_dropped` includes documents with an invalid outcome; `doc_id` is the document's row in the corpus.
- `Corpus` raises `TypeError` when `texts` is a single string.
