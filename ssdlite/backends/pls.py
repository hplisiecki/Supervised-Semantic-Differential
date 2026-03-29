"""Pure-numpy NIPALS PLS1 + cross-validated component selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ssdlite.utils.math import standardize


@dataclass(frozen=True)
class PLSCVResult:
    """Result from PLS component selection via cross-validation.

    Attributes
    ----------
    best_n_components : int
        Selected number of components.
    cv_scores : dict
        Mapping n_components -> mean CV R².
    cv_scores_se : dict
        Mapping n_components -> standard error of CV R².
    best_cv_r2 : float
        CV R² at the selected number of components.
    """
    best_n_components: int
    cv_scores: dict
    cv_scores_se: dict
    best_cv_r2: float


def pls1_fit(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """NIPALS PLS1 for a single outcome variable.

    Parameters
    ----------
    X : (n, D) centered/standardized.
    y : (n,) centered/standardized.
    n_components : latent components to extract.

    Returns
    -------
    T : (n, n_components)   X-scores
    P : (D, n_components)   X-loadings
    W : (D, n_components)   X-weights (unit-normed)
    Q : (n_components,)     y-loadings
    coef : (D,)             regression coefficients
    """
    n, D = X.shape
    n_components = min(n_components, n - 1, D)

    T = np.zeros((n, n_components), dtype=np.float64)
    P = np.zeros((D, n_components), dtype=np.float64)
    W = np.zeros((D, n_components), dtype=np.float64)
    Q = np.zeros(n_components, dtype=np.float64)

    Xk = X.copy()
    yk = y.copy()

    for a in range(n_components):
        w = Xk.T @ yk
        w_norm = float(np.linalg.norm(w))
        if w_norm < 1e-14:
            T, P, W, Q = T[:, :a], P[:, :a], W[:, :a], Q[:a]
            break
        w = w / w_norm

        t = Xk @ w
        tt = float(t @ t)
        if tt < 1e-14:
            T, P, W, Q = T[:, :a], P[:, :a], W[:, :a], Q[:a]
            break

        p = Xk.T @ t / tt
        q = float(yk @ t / tt)

        T[:, a] = t
        P[:, a] = p
        W[:, a] = w
        Q[a] = q

        Xk = Xk - np.outer(t, p)
        yk = yk - q * t

    coef = _pls1_coef_at_k(W, P, Q, W.shape[1])
    return T, P, W, Q, coef


def _pls1_coef_at_k(W, P, Q, k):
    """Regression coefficient using first k PLS components."""
    Wk, Pk, Qk = W[:, :k], P[:, :k], Q[:k]
    return Wk @ np.linalg.solve(Pk.T @ Wk, Qk)


def pls1_cv_select(
    X: np.ndarray,
    y: np.ndarray,
    max_components: int = 15,
    *,
    n_folds: int = 10,
    seed: int | None = None,
    use_1se_rule: bool = True,
    verbose: bool = False,
    pca_k: int | None = None,
) -> PLSCVResult:
    """K-fold CV to select optimal n_components for PLS1.

    Parameters
    ----------
    X : ndarray of shape (n, D)
        Raw (unstandardized) feature matrix.
    y : ndarray of shape (n,)
        Raw outcome variable.
    max_components : int, default 15
        Maximum number of components to evaluate.
    n_folds : int, default 10
        Number of CV folds.
    seed : int or None
        Random seed for fold assignment.
    use_1se_rule : bool, default True
        If True, select the smallest k within 1 SE of the best
        (parsimonious model selection).
    verbose : bool, default False
        Print progress.
    pca_k : int or None, default None
        If set, apply PCA dimensionality reduction (to *pca_k* components)
        inside each CV fold before fitting PLS.  This ensures the CV
        selects components in the same feature space as the final fit.

    Returns
    -------
    PLSCVResult
    """
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    max_comp = max(min(max_components, n // n_folds - 2), 1)

    indices = np.arange(n)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    r2_matrix = np.full((n_folds, max_comp), np.nan, dtype=np.float64)

    for fi in range(n_folds):
        val_idx = folds[fi]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fi])

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        Xs_tr, x_mean, x_scale = standardize(X_train)
        Xs_val = (X_val - x_mean) / x_scale

        # Optional PCA reduction inside the fold
        if pca_k is not None:
            from ssdlite.utils.math import pca_fit_transform
            k = min(pca_k, Xs_tr.shape[0] - 1, Xs_tr.shape[1])
            Xs_tr, comps, _ = pca_fit_transform(Xs_tr, k)
            Xs_val = Xs_val @ comps.T

        ys_tr_2d, y_mean, y_scale = standardize(y_train.reshape(-1, 1))
        ys_tr = ys_tr_2d.ravel()

        _, P, W, Q, _ = pls1_fit(Xs_tr, ys_tr, max_comp)
        actual_comp = W.shape[1]

        y_scale_val = float(y_scale[0]) if y_scale[0] > 0 else 1.0
        ys_val = (y_val - float(y_mean[0])) / y_scale_val

        ss_tot = float(np.sum((ys_val - np.mean(ys_val)) ** 2))

        for k in range(1, actual_comp + 1):
            coef_k = _pls1_coef_at_k(W, P, Q, k)
            ys_pred = Xs_val @ coef_k
            ss_res = float(np.sum((ys_val - ys_pred) ** 2))
            r2_matrix[fi, k - 1] = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    cv_scores, cv_scores_se = {}, {}
    for k in range(1, max_comp + 1):
        vals = r2_matrix[:, k - 1]
        finite = vals[np.isfinite(vals)]
        if len(finite) > 0:
            cv_scores[k] = float(np.mean(finite))
            cv_scores_se[k] = float(np.std(finite, ddof=1) / np.sqrt(len(finite)))
        else:
            cv_scores[k] = float("nan")
            cv_scores_se[k] = float("nan")

    valid = {k: v for k, v in cv_scores.items() if np.isfinite(v)}
    if not valid:
        best_k, best_r2 = 1, float("nan")
    elif use_1se_rule:
        k_star = max(valid, key=valid.get)
        threshold = valid[k_star] - cv_scores_se.get(k_star, 0.0)
        best_k = min(k for k, v in valid.items() if v >= threshold)
        best_r2 = valid[best_k]
    else:
        best_k = max(valid, key=valid.get)
        best_r2 = valid[best_k]

    return PLSCVResult(
        best_n_components=best_k,
        cv_scores=cv_scores,
        cv_scores_se=cv_scores_se,
        best_cv_r2=best_r2,
    )


def _pls1_cv_r2(X, y, n_components, n_folds, fold_indices, pca_k=None) -> float:
    """Cross-validated R² for PLS1 with fixed fold splits (used by permutation test)."""
    ss_res_total = 0.0
    ss_tot_total = 0.0

    for fi in range(n_folds):
        val_idx = fold_indices[fi]
        train_idx = np.concatenate([fold_indices[j] for j in range(n_folds) if j != fi])

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        Xs_tr, x_mean, x_scale = standardize(X_tr)
        Xs_val = (X_val - x_mean) / x_scale

        if pca_k is not None:
            from ssdlite.utils.math import pca_fit_transform
            k = min(pca_k, Xs_tr.shape[0] - 1, Xs_tr.shape[1])
            Xs_tr, comps, _ = pca_fit_transform(Xs_tr, k)
            Xs_val = Xs_val @ comps.T

        ys_tr_2d, y_mean, y_scale = standardize(y_tr.reshape(-1, 1))
        ys_tr = ys_tr_2d.ravel()

        _, _, _, _, coef = pls1_fit(Xs_tr, ys_tr, n_components)

        y_scale_val = float(y_scale[0]) if y_scale[0] > 0 else 1.0
        ys_val = (y_val - float(y_mean[0])) / y_scale_val

        ys_pred = Xs_val @ coef
        ss_res_total += float(np.sum((ys_val - ys_pred) ** 2))
        ss_tot_total += float(np.sum((ys_val - np.mean(ys_val)) ** 2))

    return 1.0 - (ss_res_total / ss_tot_total) if ss_tot_total > 0 else 0.0


def pls1_permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int,
    *,
    n_perm: int = 1000,
    n_folds: int = 5,
    seed: int | None = None,
    verbose: bool = False,
    pca_k: int | None = None,
) -> tuple[float, float, np.ndarray]:
    """Permutation test for PLS1 significance using cross-validated R².

    Parameters
    ----------
    X : ndarray of shape (n, D)
        Raw feature matrix.
    y : ndarray of shape (n,)
        Raw outcome variable.
    n_components : int
        Number of PLS components.
    n_perm : int, default 1000
        Number of permutations.
    n_folds : int, default 5
        CV folds for R² computation.
    seed : int or None
        Random seed.
    verbose : bool, default False
        Print progress.
    pca_k : int or None
        Optional PCA preprocessing before PLS.

    Returns
    -------
    p_perm : float
        Permutation p-value, computed with the (b+1)/(m+1) formula
        (Phipson & Smyth, 2010).
    cv_r2_obs : float
        Observed cross-validated R².
    cv_r2_null : ndarray of shape (n_perm,)
        Null distribution of cross-validated R².
    """
    rng = np.random.default_rng(seed)

    n = X.shape[0]
    indices = np.arange(n)
    rng.shuffle(indices)
    fold_indices = list(np.array_split(indices, n_folds))

    cv_r2_obs = _pls1_cv_r2(X, y, n_components, n_folds, fold_indices, pca_k=pca_k)

    cv_r2_null = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        y_perm = rng.permutation(y)
        cv_r2_null[i] = _pls1_cv_r2(X, y_perm, n_components, n_folds, fold_indices, pca_k=pca_k)
        if verbose and (i + 1) % 100 == 0:
            print(f"  [perm] {i + 1}/{n_perm}")

    p_perm = float((np.sum(cv_r2_null >= cv_r2_obs) + 1) / (n_perm + 1))
    return p_perm, cv_r2_obs, cv_r2_null


def _split_half_correlations(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int,
    n_splits: int,
    split_ratio: float,
    rng: np.random.Generator,
    pca_k: int | None = None,
) -> np.ndarray:
    """Run split-half PLS and return per-split Pearson r values.

    Internal helper shared by all split-test aggregation methods.
    """
    from ssdlite.utils.math import pca_fit_transform

    n = X.shape[0]
    n_train = max(int(n * split_ratio), n_components + 2)
    n_train = min(n_train, n - 3)

    r_splits = np.empty(n_splits, dtype=np.float64)

    for i in range(n_splits):
        perm = rng.permutation(n)
        train_idx, test_idx = perm[:n_train], perm[n_train:]

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        Xs_tr, x_mean, x_scale = standardize(X_tr)
        Xs_te = (X_te - x_mean) / x_scale

        if pca_k is not None:
            k = min(pca_k, Xs_tr.shape[0] - 1, Xs_tr.shape[1])
            Xs_tr, comps, _ = pca_fit_transform(Xs_tr, k)
            Xs_te = Xs_te @ comps.T

        ys_tr_2d, _, _ = standardize(y_tr.reshape(-1, 1))
        ys_tr = ys_tr_2d.ravel()

        _, _, _, _, coef = pls1_fit(Xs_tr, ys_tr, n_components)

        scores_te = (Xs_te @ coef).ravel()
        scores_c = scores_te - scores_te.mean()
        y_te_c = y_te - y_te.mean()
        ss_s = float(scores_c @ scores_c)
        ss_y = float(y_te_c @ y_te_c)

        if ss_s < 1e-15 or ss_y < 1e-15:
            r_splits[i] = 0.0
            continue

        r = float(scores_c @ y_te_c) / np.sqrt(ss_s * ss_y)
        r_splits[i] = max(-1.0, min(1.0, r))

    return r_splits


def pls1_split_test(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int,
    *,
    n_splits: int = 50,
    split_ratio: float = 0.5,
    seed: int | None = None,
    pca_k: int | None = None,
) -> tuple[float, float]:
    """Split-half significance test for PLS1 with overlap-corrected t-test.

    Aggregates per-split Pearson r values via Fisher z-transform and
    applies a variance correction to account for overlap between splits.

    The corrected standard error is:
        se = std(z, ddof=1) * sqrt(1/n_splits + n_test/n_train)

    The n_test/n_train term prevents the SE from vanishing as n_splits
    grows, honestly reflecting that correlated splits add limited new
    information.

    References: Lenartowicz P. (2026). New tests for PLS (In preparation).

    Returns
    -------
    p_split : corrected one-sided p-value.
    mean_r : mean Pearson r across splits (back-transformed from z).
    """
    from ssdlite.utils.math import t_sf

    rng = np.random.default_rng(seed)
    r_splits = _split_half_correlations(
        X, y, n_components, n_splits, split_ratio, rng, pca_k,
    )

    # Fisher z-transform (clamp to avoid inf)
    r_clamped = np.clip(r_splits, -0.9999, 0.9999)
    z_splits = np.arctanh(r_clamped)

    z_mean = float(np.mean(z_splits))
    z_std = float(np.std(z_splits, ddof=1))

    n = X.shape[0]
    n_train = max(int(n * split_ratio), n_components + 2)
    n_train = min(n_train, n - 3)
    n_test = n - n_train

    # Overlap-corrected SE
    se = z_std * np.sqrt(1.0 / n_splits + n_test / n_train)

    if se < 1e-15:
        p_split = 1.0 if z_mean <= 0 else 0.0
    else:
        t_stat = z_mean / se
        p_split = t_sf(t_stat, float(n_splits - 1))

    mean_r = float(np.tanh(z_mean))
    return p_split, mean_r


def pls1_split_test_calibrated(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int,
    *,
    n_splits: int = 50,
    split_ratio: float = 0.5,
    n_perm: int = 200,
    seed: int | None = None,
    pca_k: int | None = None,
    verbose: bool = False,
    early_stop_alpha: float = 0.05,
) -> tuple[float, float]:
    """Permutation-calibrated split-half test for PLS1.

    Runs the full split-half procedure on the observed data and on
    permuted y to build an exact null distribution of the mean split-half
    correlation.  Guarantees correct FPR control regardless of
    dependence between splits.

    Early stopping: after a minimum of 50 permutations, checks every 25
    whether a 99 % Wald CI for the p-value excludes *early_stop_alpha*.
    If so, the result is already conclusive and remaining permutations
    are skipped.

    References: Lenartowicz P. (2026). New tests for PLS (In preparation).
    Phipson & Smyth (2010) for the (b+1)/(m+1) formula.

    Returns
    -------
    p_cal : permutation-calibrated p-value.
    mean_r_obs : mean Pearson r on observed data.
    """
    rng = np.random.default_rng(seed)

    # Observed statistic
    r_obs = _split_half_correlations(
        X, y, n_components, n_splits, split_ratio, rng, pca_k,
    )
    mean_r_obs = float(np.mean(r_obs))

    # Null distribution with early stopping
    _MIN_PERM = min(50, n_perm)
    _CHECK_EVERY = 25
    _Z99 = 2.576  # z for 99 % two-sided CI

    exceedances = 0
    m = 0  # permutations completed

    for m in range(1, n_perm + 1):
        y_perm = rng.permutation(y)
        r_null = _split_half_correlations(
            X, y_perm, n_components, n_splits, split_ratio, rng, pca_k,
        )
        if float(np.mean(r_null)) >= mean_r_obs:
            exceedances += 1

        # Early stopping: 99 % Wald CI excludes alpha → conclusive
        if m >= _MIN_PERM and m % _CHECK_EVERY == 0:
            p_est = (exceedances + 1) / (m + 1)
            se = np.sqrt(p_est * (1.0 - p_est) / m)
            if se > 0 and (p_est - _Z99 * se > early_stop_alpha
                           or p_est + _Z99 * se < early_stop_alpha):
                break

        if verbose and m % 50 == 0:
            print(f"  [cal-perm] {m}/{n_perm}  (exc={exceedances})")

    # Phipson & Smyth (2010)
    p_cal = float((exceedances + 1) / (m + 1))
    return p_cal, mean_r_obs
