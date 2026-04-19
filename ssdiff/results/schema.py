"""Frozen domain dataclasses — pure data, no methods, no back-references.

Joins between entities are expressed through composite keys (`cluster_id`,
`contrast`, …) and performed by views, never by attribute access on rows.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Word:
    """Single nearest-neighbor word ranked by cosine similarity to β.

    Fields
    ------
    side : str
        ``"pos"`` for words near +β; ``"neg"`` for words near −β.
    rank : int
        1-based rank within the side (1 = closest to the gradient pole).
    word : str
        Lemma / surface form from the embedding vocabulary.
    cos_beta : float
        Cosine similarity between the word vector and the semantic gradient.
    contrast : str or None
        Group contrast key (e.g. ``"A_vs_B"``); ``None`` for continuous results.
    """
    side: str            # "pos" | "neg"
    rank: int            # 1-based
    word: str
    cos_beta: float
    contrast: str | None = None


@dataclass(frozen=True, slots=True)
class Cluster:
    """Summary row for one word cluster on a β pole.

    Fields
    ------
    cluster_id : int
        0-based index, unique within the (side, contrast) combination.
    side : str
        ``"pos"`` or ``"neg"``.
    size : int
        Number of words assigned to this cluster.
    coherence : float
        Mean pairwise cosine similarity among cluster member vectors.
    centroid_cos_beta : float
        Cosine similarity between the cluster centroid and the semantic gradient.
    contrast : str or None
        Group contrast key; ``None`` for continuous results.
    """
    cluster_id: int       # 0-based, per side
    side: str
    size: int
    coherence: float
    centroid_cos_beta: float
    contrast: str | None = None


@dataclass(frozen=True, slots=True)
class ClusterWord:
    """One word member of a cluster, with both cluster-local and global scores.

    Fields
    ------
    cluster_id : int
        Cluster this word belongs to (aligns with ``Cluster.cluster_id``).
    side : str
        ``"pos"`` or ``"neg"``.
    word : str
        Token string.
    cos_centroid : float
        Cosine similarity between this word vector and the cluster centroid.
    cos_beta : float
        Cosine similarity between this word vector and the semantic gradient.
    contrast : str or None
        Group contrast key; ``None`` for continuous results.
    """
    cluster_id: int
    side: str
    word: str
    cos_centroid: float
    cos_beta: float
    contrast: str | None = None


@dataclass(frozen=True, slots=True)
class Snippet:
    """A text window extracted near a seed word aligned with one β pole.

    Fields
    ------
    snippet_id : int
        Sequential ID (unique per result, across both sides).
    side : str
        ``"pos"`` or ``"neg"``.
    doc_id : int
        Index into the parent corpus (aligns with ``Doc.doc_id``).
    cosine : float
        Cosine similarity between the snippet's SIF-weighted vector and the
        semantic gradient.
    seed : str
        The lexicon token that anchored this snippet.
    start_token_idx, end_token_idx : int
        Token-level span within the document (inclusive start, exclusive end).
    start_sent_idx, end_sent_idx : int
        Sentence-level span within the document.
    text_window : str
        Surface text of the extracted window.
    text_surface : str
        Full surface text of the enclosing sentence(s).
    text_lemmas : str
        Lemmatized version of the enclosing sentence(s).
    cluster_id : int or None
        Cluster this snippet's seed belongs to, if clusters were computed.
    contrast : str or None
        Group contrast key; ``None`` for continuous results.
    post_id : int or None
        Within-document post index for multi-post documents (e.g. forum data).
    """
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
    """Per-document stats row.

    Fields
    ------
    doc_id : int
    y_true : float
    y_hat : float
    residual : float
    alignment_score : float
        **SSD alignment score** for this document
        (``s_i = d_i · gradient``, i.e. cosine similarity between the
        personal concept vector and the semantic gradient).
    """
    doc_id: int
    y_true: float
    y_hat: float
    residual: float
    alignment_score: float


@dataclass(frozen=True, slots=True)
class Pair:
    """Statistics for one pairwise group contrast.

    Fields
    ------
    contrast : str
        Human-readable label in the form ``"g1_vs_g2"``.
    g1, g2 : str
        Group labels in canonical order (g1 is the first group alphabetically
        after sorting by ``str``).
    T : float
        Observed cosine distance between the two group centroids (1 − cosine).
    p_raw : float
        Raw permutation p-value before multiple-comparison correction.
    p_corrected : float
        Corrected p-value (method stored in ``GroupResult.correction``).
    cohens_d : float
        Pooled Cohen's d computed from projections onto the contrast vector.
    n_g1, n_g2 : int
        Number of documents in each group after filtering.
    contrast_norm : float
        L2 norm of the raw (un-normalized) contrast vector ``c_g1 − c_g2``.
    """
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
    """One candidate lexicon token ranked by coverage and correlation.

    Fields
    ------
    token : str
        Lemma / surface form from the corpus vocabulary.
    freq : int
        Number of documents containing this token.
    cov_all : float
        Fraction of all documents containing this token.
    cov_bal : float
        Balanced coverage — fraction after equalizing group sizes.
    corr : float
        Point-biserial or Spearman correlation with the outcome variable.
    pvalue : float
        P-value for the correlation (uncorrected).
    direction : str
        ``"positive"``, ``"negative"``, or ``"none"`` — sign of ``corr``.
    rank : float
        Combined ranking score: balanced coverage × (1 − |corr| / corr_cap).
        Lower is better.
    """
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

    Fields
    ------
    backend : str
        ``"PLS"`` or ``"PCA+OLS"``.
    r2 : float
        In-sample R² (variance explained by β).
    pvalue : float
        P-value from the active statistical test.
    n_raw : int
        Total documents before preprocessing.
    n_kept : int
        Documents retained after filtering (missing embeddings, short texts).
    n_dropped : int
        Documents dropped during preprocessing.
    y_mean, y_std : float
        Mean and standard deviation of the outcome on its original scale.
    beta_norm : float
        L2 norm of β. Effect-size summary: outcome change per unit gradient move.
    delta : float
        ``0.1 × beta_norm × y_std`` — effect at the 10th-percentile step.
    iqr_effect : float
        ``(Q3 − Q1) alignment score × beta_norm × y_std`` — IQR-based effect.
    y_corr_pred : float
        Absolute Pearson correlation between y and ŷ on the standardized scale.
    r2_adj : float or None
        Adjusted R² (PCA+OLS only).
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
