# Construct Validity — PLS vs PCA+OLS Comparison

Dataset: Imbir (2016) — 4,905 Polish words, 800d GloVe embeddings (L2 + ABTT normalized).

| Dimension | K (PLS1) | R² | r | K (PLS auto) | R² | r | K (PCA+OLS) | R² | r |
|---|--:|-----:|--:|--:|-----:|--:|--:|-----:|--:|
| Valence | 1 | 0.63 | 0.79 | 3 | 0.72 | 0.85 | 119 | 0.70 | 0.83 |
| Arousal | 1 | 0.49 | 0.70 | 3 | 0.63 | 0.79 | 15 | 0.48 | 0.69 |
| Dominance | 1 | 0.50 | 0.71 | 3 | 0.64 | 0.80 | 119 | 0.62 | 0.79 |
| Origin | 1 | 0.49 | 0.70 | 3 | 0.63 | 0.79 | 19 | 0.46 | 0.68 |
| Significance | 1 | 0.58 | 0.76 | 2 | 0.66 | 0.81 | 19 | 0.59 | 0.77 |
| Concreteness | 1 | 0.66 | 0.81 | 3 | 0.74 | 0.86 | 47 | 0.68 | 0.83 |
| Imageability | 1 | 0.54 | 0.73 | 3 | 0.66 | 0.82 | 57 | 0.59 | 0.77 |
| Age of Acq. | 1 | 0.49 | 0.70 | 3 | 0.66 | 0.81 | 119 | 0.61 | 0.78 |

PCA+OLS R² back-computed from Adj R² in manuscript Table 9 (Plisiecki et al., 2025).
PLS R² back-computed from Adj R² (difference < 0.001 for K ≤ 3).
