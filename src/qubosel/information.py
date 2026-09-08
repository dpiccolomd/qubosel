"""Discretisation and plug-in information measures.

All estimators work on integer *codes* (one column per variable). Entropies
are in nats unless ``base`` is given (``base=2`` for bits).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


# --------------------------------------------------------------------------- binning
def default_n_bins(n_samples: int) -> int:
    """Default bin budget: ``max(2, floor(sqrt(n)/2))`` (31 -> 2, 100 -> 5)."""
    return max(2, math.floor(math.sqrt(n_samples) / 2.0))


def _factorize(col: np.ndarray) -> np.ndarray:
    _, codes = np.unique(col, return_inverse=True)
    return codes.astype(np.int64)


def _quantile_codes(col: np.ndarray, n_bins: int) -> np.ndarray:
    qs = np.quantile(col, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(qs)[1:-1]  # interior edges only
    return np.searchsorted(edges, col, side="right").astype(np.int64)


def _equal_width_codes(col: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-width bins on ``[min, max]``; the maximum falls in the last bin.

    Matches ``numpy.histogramdd(col, bins=n_bins)`` counting semantics.
    """
    lo, hi = float(col.min()), float(col.max())
    if hi == lo:
        return np.zeros(col.shape[0], dtype=np.int64)
    edges = np.linspace(lo, hi, n_bins + 1)
    codes = np.searchsorted(edges, col, side="right") - 1
    codes[col >= hi] = n_bins - 1
    return codes.astype(np.int64)


def discretize(
    X: ArrayLike,
    n_bins: int | None = None,
    method: str = "quantile",
) -> NDArray[np.int64]:
    """Turn each column of ``X`` into integer codes.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    n_bins : int, optional
        Bin budget per column. Default: :func:`default_n_bins` for ``"quantile"``,
        10 for ``"legacy"``. The effective number of bins is ``min(n_unique, n_bins)``.
    method : {"quantile", "legacy"}
        ``"quantile"`` (default) uses quantile edges; columns with
        ``n_unique <= n_bins`` are factorised as-is (binary/one-hot columns are
        left intact). ``"legacy"`` is the equal-width histogram binning of the D-Wave
        example lineage: min-max scaling followed by equal-width bins of
        ``min(n_unique, 10)`` per column (``numpy.histogramdd`` semantics).
    """
    Xa = np.asarray(X, dtype=np.float64)
    if Xa.ndim != 2:
        raise ValueError("X must be 2-D")
    if np.isnan(Xa).any():
        raise ValueError("X contains NaN; impute before discretising")
    d = Xa.shape[1]
    n = Xa.shape[0]
    if method == "quantile":
        budget = default_n_bins(n) if n_bins is None else int(n_bins)
    elif method == "legacy":
        budget = 10 if n_bins is None else int(n_bins)
    else:
        raise ValueError(f"unknown method {method!r}")
    if budget < 1:
        raise ValueError("n_bins must be >= 1")
    out = np.empty((n, d), dtype=np.int64)
    for j in range(d):
        col = Xa[:, j]
        n_unique = np.unique(col).shape[0]
        if method == "quantile":
            if n_unique <= budget:
                out[:, j] = _factorize(col)
            else:
                out[:, j] = _quantile_codes(col, budget)
        else:
            lo, hi = col.min(), col.max()
            scaled = (col - lo) / (hi - lo) if hi > lo else np.zeros_like(col)
            out[:, j] = _equal_width_codes(scaled, min(n_unique, budget))
    return out


# --------------------------------------------------------------------------- entropy
def _joint_codes(*cols: np.ndarray) -> np.ndarray:
    stacked = np.stack([np.asarray(c).ravel() for c in cols], axis=1)
    _, inv = np.unique(stacked, axis=0, return_inverse=True)
    return inv.ravel()


def _counts(codes: np.ndarray) -> np.ndarray:
    return (
        np.bincount(_joint_codes(codes)) if codes.ndim == 1 else np.bincount(_joint_codes(*codes.T))
    )


def entropy(*cols: ArrayLike, base: float | None = None, correction: str | None = None) -> float:
    """Plug-in (joint) entropy of one or more code vectors.

    ``correction="miller_madow"`` adds ``(m - 1) / (2 N)`` nats, ``m`` being the
    number of occupied cells.
    """
    arrs = [np.asarray(c).ravel() for c in cols]
    n = arrs[0].shape[0]
    if n == 0:
        return 0.0
    counts = np.bincount(_joint_codes(*arrs))
    counts = counts[counts > 0]
    p = counts / n
    h = float(-(p * np.log(p)).sum())
    if correction == "miller_madow":
        h += (counts.shape[0] - 1) / (2.0 * n)
    elif correction in ("permutation", "expected"):
        raise ValueError(f"correction={correction!r} applies to MI matrices, not to entropies")
    elif correction is not None:
        raise ValueError(f"unknown correction {correction!r}")
    if base is not None:
        h /= math.log(base)
    return max(h, 0.0)


def mutual_info(
    a: ArrayLike, b: ArrayLike, base: float | None = None, correction: str | None = None
) -> float:
    """``I(a; b) = H(a) + H(b) - H(a, b)`` (clipped at 0)."""
    kw = {"base": base, "correction": correction}
    return max(entropy(a, **kw) + entropy(b, **kw) - entropy(a, b, **kw), 0.0)


def conditional_mi(
    a: ArrayLike,
    b: ArrayLike,
    z: ArrayLike,
    base: float | None = None,
    correction: str | None = None,
) -> float:
    """``I(a; b | z) = sum_z p(z) I(a; b | Z = z)``."""
    a_arr, b_arr, z_arr = (np.asarray(v).ravel() for v in (a, b, z))
    n = a_arr.shape[0]
    zc = _joint_codes(z_arr)
    total = 0.0
    for val in np.unique(zc):
        mask = zc == val
        w = mask.sum() / n
        total += w * mutual_info(a_arr[mask], b_arr[mask], base=base, correction=correction)
    return max(total, 0.0)


def null_mean_mi(
    a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n_null: int, base: float | None
) -> float:
    """Mean plug-in MI between ``a`` and ``n_null`` permutations of ``b`` (the estimator's bias)."""
    return float(
        np.mean([mutual_info(a, rng.permutation(b), base=base) for _ in range(int(n_null))])
    )


def expected_mi(a: np.ndarray, b: np.ndarray, base: float | None = None) -> float:
    """Exact expectation of plug-in MI under the permutation null (fixed marginals).

    The null model permutes ``b`` against ``a`` over all ``n!`` permutations;
    the contingency cells are then hypergeometric and the expectation has a
    closed form (Vinh, Epps & Bailey 2010; used by Mandros, Boley & Vreeken
    2017/2020 as the "reliable" correction). Computed with scikit-learn's
    ``expected_mutual_information``.
    """
    from sklearn.metrics.cluster import contingency_matrix, expected_mutual_information

    a_arr, b_arr = np.asarray(a).ravel(), np.asarray(b).ravel()
    C = contingency_matrix(a_arr, b_arr)
    emi = float(expected_mutual_information(C, int(a_arr.shape[0])))
    if base is not None:
        emi /= math.log(base)
    return max(emi, 0.0)


def expected_conditional_mi(
    a: np.ndarray, b: np.ndarray, z: np.ndarray, base: float | None = None
) -> float:
    """Exact null expectation of ``I(a; b | z)`` under permutation of ``b`` within each ``z``."""
    a_arr, b_arr, z_arr = (np.asarray(v).ravel() for v in (a, b, z))
    n = a_arr.shape[0]
    zc = _joint_codes(z_arr)
    total = 0.0
    for val in np.unique(zc):
        mask = zc == val
        total += mask.sum() / n * expected_mi(a_arr[mask], b_arr[mask], base=base)
    return total


def _split_correction(correction: str | None) -> tuple[str | None, str | None]:
    """Return ``(entropy_correction, null_mode)`` with ``null_mode`` in {None, "mc", "exact"}."""
    if correction == "permutation":
        return None, "mc"
    if correction == "expected":
        return None, "exact"
    return correction, None


def mi_matrix(
    codes: ArrayLike,
    y: ArrayLike,
    base: float | None = None,
    correction: str | None = None,
    n_null: int = 100,
    random_state: int | np.random.Generator | None = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Importance vector ``I_i = I(x_i; y)`` and redundancy matrix ``R_ij = I(x_i; x_j)``.

    ``R`` is symmetric with a zero diagonal. ``correction="expected"`` subtracts
    from every term its exact expectation under the permutation null (the
    "reliable mutual information" of Mandros, Boley & Vreeken 2017/2020, via
    the closed form of Vinh, Epps & Bailey 2010) and clips at zero;
    ``correction="permutation"`` is the Monte-Carlo version over ``n_null``
    permutations. With few samples the plug-in estimator's positive bias grows
    with the number of distinct values of a variable, and without this
    subtraction the QUBO ranks columns by cardinality rather than by association.
    """
    C = np.asarray(codes)
    yv = np.asarray(y).ravel()
    d = C.shape[1]
    corr, permute = _split_correction(correction)
    rng = (
        np.random.default_rng(random_state)
        if not isinstance(random_state, np.random.Generator)
        else random_state
    )
    kw = {"base": base, "correction": corr}
    imp = np.array([mutual_info(C[:, i], yv, **kw) for i in range(d)])
    red = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            red[i, j] = red[j, i] = mutual_info(C[:, i], C[:, j], **kw)
    if permute == "mc":
        for i in range(d):
            imp[i] = max(imp[i] - null_mean_mi(C[:, i], yv, rng, n_null, base), 0.0)
        for i in range(d):
            for j in range(i + 1, d):
                red[i, j] = red[j, i] = max(
                    red[i, j] - null_mean_mi(C[:, i], C[:, j], rng, n_null, base), 0.0
                )
    elif permute == "exact":
        for i in range(d):
            imp[i] = max(imp[i] - expected_mi(C[:, i], yv, base), 0.0)
        for i in range(d):
            for j in range(i + 1, d):
                red[i, j] = red[j, i] = max(red[i, j] - expected_mi(C[:, i], C[:, j], base), 0.0)
    return imp, red


def conditional_redundancy_matrix(
    codes: ArrayLike,
    y: ArrayLike,
    base: float | None = None,
    correction: str | None = None,
    n_null: int = 100,
    random_state: int | np.random.Generator | None = 0,
) -> NDArray[np.float64]:
    """Class-conditional redundancy ``D[i, j] = I(x_i; x_j | y)`` (symmetric, zero diagonal)."""
    C = np.asarray(codes)
    yv = np.asarray(y).ravel()
    d = C.shape[1]
    corr, permute = _split_correction(correction)
    rng = (
        np.random.default_rng(random_state)
        if not isinstance(random_state, np.random.Generator)
        else random_state
    )
    out = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            v = conditional_mi(C[:, i], C[:, j], yv, base=base, correction=corr)
            if permute == "mc":
                null = np.mean(
                    [
                        conditional_mi(C[:, i], rng.permutation(C[:, j]), yv, base=base)
                        for _ in range(int(n_null))
                    ]
                )
                v = max(v - float(null), 0.0)
            elif permute == "exact":
                v = max(v - expected_conditional_mi(C[:, i], C[:, j], yv, base=base), 0.0)
            out[i, j] = out[j, i] = v
    return out


def cmi_matrix(
    codes: ArrayLike,
    y: ArrayLike,
    base: float | None = None,
    correction: str | None = None,
    n_null: int = 100,
    random_state: int | np.random.Generator | None = 0,
) -> NDArray[np.float64]:
    """Conditional-MI matrix ``C[i, j] = I(x_j; y | x_i)`` (zero diagonal)."""
    C = np.asarray(codes)
    yv = np.asarray(y).ravel()
    d = C.shape[1]
    corr, permute = _split_correction(correction)
    rng = (
        np.random.default_rng(random_state)
        if not isinstance(random_state, np.random.Generator)
        else random_state
    )
    out = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if i != j:
                v = conditional_mi(C[:, j], yv, C[:, i], base=base, correction=corr)
                if permute == "mc":
                    null = np.mean(
                        [
                            conditional_mi(C[:, j], rng.permutation(yv), C[:, i], base=base)
                            for _ in range(int(n_null))
                        ]
                    )
                    v = max(v - float(null), 0.0)
                elif permute == "exact":
                    v = max(v - expected_conditional_mi(C[:, j], yv, C[:, i], base=base), 0.0)
                out[i, j] = v
    return out


# --------------------------------------------------------- discretisation-free
def rank_relevance(X: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
    """``2 |AUC_i - 0.5|`` per feature for a binary outcome (no discretisation).

    The AUC of a single feature as a score is a rank statistic (Mann-Whitney),
    so this relevance has no cardinality bias and no bins.
    """
    from sklearn.metrics import roc_auc_score

    Xa = np.asarray(X, dtype=np.float64)
    yv = np.asarray(y).ravel()
    classes = np.unique(yv)
    if classes.shape[0] != 2:
        raise ValueError("rank_relevance requires a binary outcome")
    yb = (yv == classes[1]).astype(int)
    out = np.zeros(Xa.shape[1])
    for i in range(Xa.shape[1]):
        col = Xa[:, i]
        if np.unique(col).shape[0] > 1:
            out[i] = 2.0 * abs(roc_auc_score(yb, col) - 0.5)
    return out


def spearman_redundancy(X: ArrayLike) -> NDArray[np.float64]:
    """Squared Spearman rank correlation between features (symmetric, zero diagonal)."""
    from scipy.stats import rankdata

    Xa = np.asarray(X, dtype=np.float64)
    R = np.column_stack([rankdata(Xa[:, j]) for j in range(Xa.shape[1])])
    sd = R.std(axis=0)
    R = (R - R.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    C = (R.T @ R) / Xa.shape[0]
    C[:, sd == 0] = 0.0
    C[sd == 0, :] = 0.0
    out = C**2
    np.fill_diagonal(out, 0.0)
    return out


def ksg_mi_matrix(
    X: ArrayLike,
    y: ArrayLike,
    discrete: ArrayLike | None = None,
    n_neighbors: int = 3,
    n_repeats: int = 3,
    random_state: int | None = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Importance and redundancy with scikit-learn's kNN (Kraskov) MI estimator.

    ``discrete`` marks columns treated as discrete (default: at most two unique
    values). Estimates are averaged over ``n_repeats`` seeds because the
    estimator jitters continuous columns. Values are in nats, clipped at 0.
    """
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    Xa = np.asarray(X, dtype=np.float64)
    yv = np.asarray(y).ravel()
    _, y_codes = np.unique(yv, return_inverse=True)
    d = Xa.shape[1]
    disc = (
        np.array([np.unique(Xa[:, j]).shape[0] <= 2 for j in range(d)])
        if discrete is None
        else np.asarray(discrete, dtype=bool)
    )
    seeds = range(int(random_state or 0), int(random_state or 0) + int(n_repeats))
    imp = np.zeros(d)
    red = np.zeros((d, d))
    for seed in seeds:
        imp += mutual_info_classif(
            Xa, y_codes.ravel(), discrete_features=disc, n_neighbors=n_neighbors, random_state=seed
        )
        for i in range(d):
            for j in range(i + 1, d):
                if disc[i] and disc[j]:
                    v = mutual_info_classif(
                        Xa[:, [i]],
                        Xa[:, j].astype(int),
                        discrete_features=[True],
                        random_state=seed,
                    )[0]
                elif disc[j]:
                    v = mutual_info_classif(
                        Xa[:, [i]],
                        Xa[:, j].astype(int),
                        discrete_features=[False],
                        n_neighbors=n_neighbors,
                        random_state=seed,
                    )[0]
                elif disc[i]:
                    v = mutual_info_classif(
                        Xa[:, [j]],
                        Xa[:, i].astype(int),
                        discrete_features=[False],
                        n_neighbors=n_neighbors,
                        random_state=seed,
                    )[0]
                else:
                    v = mutual_info_regression(
                        Xa[:, [i]], Xa[:, j], n_neighbors=n_neighbors, random_state=seed
                    )[0]
                red[i, j] += v
    imp = np.maximum(imp / len(seeds), 0.0)
    red = red / len(seeds)
    red = np.maximum(red + red.T, 0.0)
    np.fill_diagonal(red, 0.0)
    return imp, red
