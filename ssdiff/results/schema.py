"""Frozen domain dataclasses — pure data, no methods, no back-references.

Joins between entities are expressed through composite keys (`cluster_id`,
`contrast`, …) and performed by views, never by attribute access on rows.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Word:
    side: str            # "pos" | "neg"
    rank: int            # 1-based
    word: str
    cos_beta: float
    contrast: str | None = None


@dataclass(frozen=True, slots=True)
class Cluster:
    cluster_id: int       # 0-based, per side
    side: str
    size: int
    coherence: float
    centroid_cos_beta: float
    contrast: str | None = None


@dataclass(frozen=True, slots=True)
class ClusterWord:
    cluster_id: int
    side: str
    word: str
    cos_centroid: float
    cos_beta: float
    contrast: str | None = None


@dataclass(frozen=True, slots=True)
class Snippet:
    snippet_id: int
    side: str
    doc_id: int
    cosine: float
    seed: str
    start_token_idx: int
    end_token_idx: int
    start_sent_idx: int
    end_sent_idx: int
    text_window: str
    text_surface: str
    text_lemmas: str
    cluster_id: int | None = None
    contrast: str | None = None
    post_id: int | None = None


@dataclass(frozen=True, slots=True)
class Doc:
    doc_id: int
    y_true: float
    y_hat: float
    residual: float
    cos_align: float


@dataclass(frozen=True, slots=True)
class Pair:
    contrast: str          # "g1_vs_g2"
    g1: str
    g2: str
    T: float
    p_raw: float
    p_corrected: float
    cohens_d: float
    n_g1: int
    n_g2: int
    contrast_norm: float


@dataclass(frozen=True, slots=True)
class Suggestion:
    token: str
    freq: int
    cov_all: float
    cov_bal: float
    corr: float
    pvalue: float
    direction: str          # "positive", "negative", "none"
    rank: float             # balanced coverage × (1 − |corr|/corr_cap)


@dataclass(frozen=True, slots=True)
class Stats:
    """Continuous-result stats. One row per result.

    ``r2_adj`` is only populated for PCA+OLS — for PLS it stays ``None``
    because adjusted R² is an OLS statistic and isn't meaningful under
    cross-validated PLS.
    """
    backend: str            # "PLS" | "PCA+OLS"
    r2: float
    pvalue: float
    n_raw: int
    n_kept: int
    n_dropped: int
    y_mean: float
    y_std: float
    beta_norm: float
    delta: float
    iqr_effect: float
    y_corr_pred: float
    r2_adj: float | None = None


@dataclass(frozen=True, slots=True)
class FitInfo:
    """Backend-specific fit diagnostics (hyperparams + scalar fit state).

    Not to be confused with `Stats`, which holds *model-quality* scalars that
    are comparable across backends. `FitInfo` holds backend-specific *fit
    configuration* (how the model was fit, not how well).

    All fields are Optional because different backends populate different
    subsets. A renderer that encounters None for an irrelevant field skips it.
    """
    n_components: int | None = None
    # PLS-specific
    pca_k: int | None = None
    p_method: str | None = None      # "perm" | "split" | "split_cal" | None
    n_perm: int | None = None
    n_splits: int | None = None
    split_ratio: float | None = None
    split_mean_r: float | None = None
    random_state: int | None = None
    # PCA+OLS specific
    k_min: int | None = None
    k_max: int | None = None
    k_step: int | None = None
    best_k: int | None = None


@dataclass(frozen=True, slots=True)
class Summary:
    """LexiconResult summary (present only after evaluate_lexicon).

    Raw floats here; formatting is applied at render time (D11).
    """
    docs_any: int
    cov_all: float
    q1: float
    q4: float
    corr_any: float
    hits_mean: float
    hits_median: float
    types_mean: float
    types_median: float
    group_cov: dict[str, float] | None = None
