# SSDLite vs Official `ssdiff` — Validation Report

Comparison across 7 Polish-language datasets (Kalibra corpus, N ≈ 636–655 each),
GloVe 800d embeddings (L2 + ABTT m=1), context window ±3, SIF a = 0.001.

---

## 1. Algorithmic Equivalence

SSDLite's PCA+OLS backend reproduces the official `ssdiff` results **exactly** —
identical K, R², adj R², p-values, and top words across all datasets.

| Dataset      | K   | R²     | adj R² | p-value   |
|--------------|-----|--------|--------|-----------|
| imigrant     | 46  | 0.1407 | 0.0434 | 0.03484   |
| klimat       | 38  | 0.1031 | 0.0384 | 0.01541   |
| naukowcy     | 60  | 0.2014 | 0.1120 | < 0.001   |
| polityka     | 84  | 0.2610 | 0.1355 | < 0.001   |
| szczepienie  | 72  | 0.3113 | 0.2167 | < 0.001   |
| zaufanie     | 42  | 0.1209 | 0.0541 | 0.00180   |
| zdrowie      | 72  | 0.1567 | 0.0272 | 0.12859   |

Both PCA+OLS columns (SSDLite and official) match in every cell.
Top-word lists are identical; cluster structure is identical.

**Conclusion:** SSDLite is a faithful reimplementation — zero numerical divergence.

---

## 2. Speed

### Full pipeline (NKJP 300d embeddings, szczepienie)

Each fit operation averaged over 10 runs. SSDLite times averaged across 2 full
benchmark passes.

| Operation               | SSDLite       | Official       | Speedup |
|-------------------------|---------------|----------------|---------|
| Preprocess texts        | 5.3 s         | 6.8 s          | 1.3×    |
| Load .txt embeddings    | 63.3 s        | 285.1 s        | **4.5×**  |
| Load .txt (parallel)    | 21.3 s        | —              | **13.4×** |
| Load native format      | 2.0 s (.ssdembed) | 1.7 s (.kv) | —      |
| **Normalization**       | **3.5 s**     | **150.9 s**    | **43×** |
| PCA sweep (per run)     | 11.9 s        | 21.4 s         | 1.8×    |
| SSD fit K=20 (per run)  | 0.10 s        | 0.15 s         | —       |

| Resource (Peak RSS)     | SSDLite       | Official       | Ratio    |
|-------------------------|---------------|----------------|----------|
| **Normalization**       | **2,414 MB**  | **17,217 MB**  | **7.1×** |
| PCA sweep               | 2,366 MB      | 2,694 MB       | ~1×      |
| SSD fit                  | 2,350 MB      | 2,380 MB       | ~1×      |

The normalization step dominates: `ssdiff` loads the full covariance matrix
for SVD-based ABTT, peaking at 17 GB — enough to swap on most machines.
SSDLite's in-place implementation stays under 2.5 GB.

---

## 3. Python Version Support

| Python version | Official `ssdiff` | SSDLite |
|----------------|-------------------|---------|
| 3.9            | ✗ (fails) | ✗ |
| 3.10           | ✓ | ✓ |
| 3.11           | ✓ | ✓ |
| 3.12           | ✓ | ✓ |
| 3.13           | ✗ | ✓ |

The official `ssdiff` claims Python 3.9+ compatibility but fails on 3.9;
it works only on 3.10–3.12. SSDLite supports 3.10–3.13.

---

## Summary

| Claim                        | Evidence                                                        |
|------------------------------|-----------------------------------------------------------------|
| Drop-in replacement          | PCA+OLS reproduces official R², p, K, words exactly (7 datasets)|
| Faster normalization         | 43× faster (3.5 s vs 151 s) — required step for new embeddings |
| Faster embedding loading     | 4.5× serial, 13× parallel for .txt format                      |
| Faster PCA sweep             | 1.8× per run (12 s vs 21 s)                                    |
| 7× lower peak memory         | 2.4 GB vs 17.2 GB during normalization — won't swap on 16 GB   |
| Stable on limited hardware   | Single-core profile; no thread-pool overhead or swap risk       |
| Fewer dependencies           | No pandas, optional sklearn                                     |
| Wider Python support         | 3.10–3.13 vs 3.10–3.12 (official claims 3.9 but fails, lacks 3.13) |
