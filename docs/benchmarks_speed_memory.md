# SSDLite vs Official `ssdiff` — Validation Report

Comparison across 7 Polish-language datasets (Kalibra corpus, N ≈ 636–655 each),
GloVe 800d embeddings (L2 + ABTT m=1), context window ±3, SIF a = 0.001.

---

## 1. Speed

### Full pipeline (NKJP 300d embeddings, szczepienie)

Each fit operation averaged over 10 runs.

| Operation               | SSDLite        | Official       | Speedup     |
|-------------------------|----------------|----------------|-------------|
| Preprocess texts        | 6.4 s          | 6.8 s          | ~1×         |
| Load .txt embeddings    | 73.9 s         | 285.1 s        | **3.9×**    |
| Load .txt (parallel)    | 21.4 s         | —              | **13.3×**   |
| Load native format      | 1.8 s (.ssdembed) | 1.7 s (.kv) | ~1×         |
| **Normalization**       | **3.3 s**      | **150.9 s**    | **45×**     |
| PCA sweep (per run)     | 1.9 s          | 21.4 s         | **11.2×**   |
| SSD fit OLS (per run)   | 0.10 s         | 0.15 s         | ~1×         |
| SSD fit PLS (per run)   | 0.10 s         | —              | —           |

| Resource (Peak RSS)     | SSDLite        | Official       | Ratio       |
|-------------------------|----------------|----------------|-------------|
| **Normalization**       | **2,415 MB**   | **17,217 MB**  | **7.1×**    |
| PCA sweep               | 2,281 MB       | 2,694 MB       | ~1×         |
| SSD fit                 | 2,367 MB       | 2,694 MB       | ~1×         |

The normalization step dominates: `ssdiff` loads the full covariance matrix
for SVD-based ABTT, peaking at 17 GB — enough to swap on most machines.
SSDLite's in-place implementation stays under 2.5 GB.

---

## 2. Python Version Support

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
| Faster normalization         | 45× faster (3.3 s vs 151 s) — required step for new embeddings |
| Faster embedding loading     | 3.9× serial, 13× parallel for .txt format                      |
| Faster PCA sweep             | 11× per run (1.9 s vs 21.4 s)                                  |
| 7× lower peak memory         | 2.4 GB vs 17.2 GB during normalization — won't swap on 16 GB   |
| Stable on limited hardware   | Single-core profile; no thread-pool overhead or swap risk       |
| Fewer dependencies           | No pandas, no scikit-learn, no scipy, no requests; gensim optional |
| Wider Python support         | 3.10–3.13 vs 3.10–3.12 (official claims 3.9 but fails, lacks 3.13) |
