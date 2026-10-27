# Changelog

## 4.0.0

A breaking release. `SSD` now holds one text and nothing about the outcome; each fit names its outcome.

- `SSD(texts, *, embeddings=None, lexicon=None, window=3, sif_a=1e-3, use_full_doc=None)` takes the texts directly, as its only positional argument. `embeddings=None` loads `Embeddings.load()`, the default wordpack. `use_full_doc=None` means full documents without a lexicon and seed contexts with one. Strings are tokenized by the embeddings' wordpack and its stopwords. Any other embeddings take token lists, which are used as given. Strings with embeddings that are not a wordpack raise `TypeError`. The `y=` and `data=` arguments are gone.
- `Corpus` is removed, and so is spaCy: the `[spacy]` extra, spaCy loading and preprocessing, `load_stopwords`, the Polish stopword file and the per-language spaCy table. Wordpacks exist for English only for now, so texts in other languages come in as token lists the caller lemmatized.
- Every fit takes the outcome as its first argument: values, a column name read from `data=`, or, for `fit_regression()` and `fit_pls()`, a formula whose left side names the outcome column. `fit_multipls()` takes 2-D values or a list of column names. `fit_groups()` takes labels or one column name. Rows with a missing outcome or no document vector are dropped at fit time.
- New `ssd.with_lexicon(lexicon, *, window=..., sif_a=..., use_full_doc=...)` returns a new `SSD` that shares the tokens, so one tokenization can be tried with several lexicons.
- The lexicon helpers move from `Corpus` to `SSD`: `suggest_lexicon`, `token_stats`, `coverage_summary` and `evaluate_lexicon`. Their outcome argument works like the fits'.
- New `ssd.n_kept`: the number of documents that have a vector. Each fit reports its own `n`.
- SIF word counts are computed on all documents, not only on those with a valid outcome, so the weights no longer change with missing outcome values.
- `fit_regression()` takes the text as a formula term, `ssd()`: `fit_regression("score ~ age + sex*ssd()", data=df)`. A formula without `ssd(...)` gets `+ ssd()` appended. K is chosen by the PCA sweep without interactions. Each text term gets its own gradient, with `words`, `clusters` and `snippets`; under treatment coding `ssd()` is the gradient at the reference level, and `sexM:ssd()` the difference from it.
- With more than one text term, `fit_regression()` returns a `RegressionResult` keyed by term (`res["sexM:ssd(essay)"]`). With one text term it returns a `PCAOLSResult`, as before.
- `fit_regression()` results' `test` gains a `label` column and one row per term label. The first row is the test of all text terms, a refit compared with `glmm.compare`. Each term row is one Wald test of that term's coefficients (`Fit.contrast_test`), so terms contained in an interaction are tested too.
- New `ssdiff.fit_regression(formula, data, **texts)` fits one or several texts in one regression: `ssdiff.fit_regression("score ~ ssd(essay) + sex*ssd(diary)", data=df, essay=essay, diary=diary)`. An `ssd(name)` with no keyword is built from the column `name` of `data` with every default, so `ssdiff.fit_regression("score ~ ssd(essay)", df)` is a whole fit; a keyword wins over a column. Each text has its own embeddings, lexicon, PCA basis and K. A row stays only if it has a vector in every text. The result is keyed by text, and `test` adds one row per text.
- New `fit_regression(ssd_gradients=N)`: with N > 1 the `PCAOLSResult` also holds N rotated gradients, `result["dim-1"]` … `result["dim-N"]`, as `fit_pls(ssd_gradients=N)` does. They come from the regression's own model: its K components are split into N consecutive blocks, each block's part of the model's beta is one gradient, and the N gradients are rotated by words. The standard fit does not change. It needs the text as one plain `ssd()` term. `fit_info.ssd_gradients` is now 1, not `None`, for every `fit_regression()` result.
- New `ssdiff.contrast(g1, g2, embeddings, *, labels=("g1", "g2"))` returns a `ContrastResult`: the normalized difference of the two unit gradients, with `words`, `clusters` and `stats` (`cos_g1_g2`, `diff_norm`). It has no test, no snippets and no documents.
- `summary()` shows the formula as written, `ssd()` terms included. Its SSD effects block is replaced by a text-term table, one Wald row per text term (df, statistic, p), and, with several text terms, a joint-tests block with the rows of `test` that compare fits.
- `fit_regression()` results (`PCAOLSResult` and `RegressionResult`, at every level) gain statsmodels names: `params`, `bse`, `tvalues`, `conf_int()`, `cov_params()` (covariates only), `pvalues` (covariates, then one Wald row per text term), `fvalue` / `f_pvalue` (OLS) and `llr` / `llr_pvalue` (other models) for the whole model against `outcome ~ 1`, `df_model`, `df_resid`, `nobs`, `rsquared`, `rsquared_adj`, `aic`, `bic`, `llf`, `resid`, `fittedvalues` and `random_effects`. They need pandas.
- Results carry `text` (the `SSD` they were fit on) in place of `corpus`, and no longer carry `lang`. A pickled result leaves out its text's embeddings and document vectors.
- `result.predict(texts)` and `result.attach(texts=..., embeddings=...)` take texts in the form the fit used: strings for a fit on strings, token lists for a fit on token lists. `attach(corpus=...)` is gone.
- Results from token-list texts have no snippets.
- `.docx` tables take the layout of a paper's results section: an italic caption above the table, rules above and below the header and under the last row, no vertical lines, Times New Roman 11 pt. This holds for views, multi-key views and `report()`. `clusters.save("x.docx")` writes one table per side with the report's cluster columns (No., Size, the 10 closest words, a representative excerpt), captioned with the pole: "Positive semantic clusters (+β̂ → higher score)", or "toward A" for a group pair. New `fit_info.outcome` holds the outcome's column name (`"y"` when given as values).
- A cluster table in `report()` drops its excerpt column when no cluster has a snippet, as with token-list texts.
- The neighbour filter of embeddings other than a wordpack skips a word that contains a digit or starts with an uppercase letter, in every script. Uppercase Greek words are now skipped too.
- The results page moves to report format 2. It shows each text the result uses, one unit per text term, the per-term and per-text tests, and the contrast page. Documents from token lists show their tokens.
- `glmm` >= 0.6.2 is required.

## 3.3.0

- `Corpus(lang=..., model=...)` is renamed `Corpus(spacy_lang=..., spacy_model=...)`: both only choose the spaCy pipeline. The old names raise `TypeError`.
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
- `fit_pls()` and `fit_multipls()` (and their `result.test(...)` reruns) test with plskit's default `test_method="auto"` instead of always `split_exact`: plskit runs `split_exact` or `split_nb`, chosen from X alone, and `result.test.name` says which ran. With n below 250 it is `split_exact`, as before.
- Sparse PLS is removed: `fit_pls(keep=...)`, the `"split_nb_seq"` test and `fit_info.keep` are gone.
- k-means, the `fit_groups()` permutation test and the `median_split` tie-break draw from a new random generator shared with the R package (ssdiff), so clusters, the chosen K of a `fit_regression()` sweep, `fit_groups()` p-values and `median_split` groups differ from earlier versions for the same seed.
- Profile mode is removed: in `Corpus(texts)` without `pretokenized=True`, an element of `texts` that is a list raises `TypeError`. For several posts per participant, join each participant's posts into one text, or use one row per post with a participant random effect in `fit_regression()`. The snippets table loses `post_id`.

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
