"""Synthetic clinical-like dataset generator."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def make_clinical_synthetic(
    n_samples: int = 60,
    n_features: int = 14,
    n_informative: int = 3,
    n_redundant_pairs: int = 2,
    n_binary: int = 2,
    onehot_groups: Sequence[Sequence[int]] = ((3,), (4,)),
    class_weight: float = 0.35,
    signal: float = 1.5,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Small tabular dataset mimicking a clinical outcome study.

    Layout of the columns (in order): ``n_informative`` continuous informative
    features, one redundant near-copy for each of the first ``n_redundant_pairs``
    informative features, ``n_binary`` binary noise features, one-hot groups of
    the sizes listed in ``onehot_groups`` (each group is a categorical variable
    with that many levels, noise), and continuous noise features to reach
    ``n_features``. The binary outcome depends only on the informative block.

    Returns ``(X, y, truth)`` where ``truth`` holds the informative and
    redundant column indices.
    """
    rng = np.random.default_rng(random_state)
    n_onehot = sum(sum(g) for g in onehot_groups)
    n_fixed = n_informative + n_redundant_pairs + n_binary + n_onehot
    if n_features < n_fixed:
        raise ValueError(f"n_features must be >= {n_fixed} for this layout")
    if n_redundant_pairs > n_informative:
        raise ValueError("n_redundant_pairs must be <= n_informative")

    cols: dict[str, np.ndarray] = {}
    inf = rng.normal(size=(n_samples, n_informative))
    w = rng.uniform(0.8, 1.2, size=n_informative) * signal
    logit = inf @ w
    logit = logit - np.quantile(logit, 1.0 - class_weight)
    y = (logit + rng.logistic(scale=0.5, size=n_samples) > 0).astype(int)
    if np.unique(y).shape[0] < 2:  # pragma: no cover - degenerate draw
        y[rng.integers(0, n_samples)] ^= 1
    for i in range(n_informative):
        cols[f"inf_{i}"] = inf[:, i]
    for i in range(n_redundant_pairs):
        cols[f"red_{i}"] = inf[:, i] + rng.normal(scale=0.15, size=n_samples)
    for i in range(n_binary):
        cols[f"bin_{i}"] = rng.integers(0, 2, size=n_samples).astype(float)
    for g, sizes in enumerate(onehot_groups):
        for size in sizes:
            levels = rng.integers(0, size, size=n_samples)
            for lvl in range(size):
                cols[f"cat{g}_{lvl}"] = (levels == lvl).astype(float)
    i = 0
    while len(cols) < n_features:
        cols[f"noise_{i}"] = rng.normal(size=n_samples)
        i += 1
    X = pd.DataFrame(cols)
    names = list(X.columns)
    truth = {
        "informative": [names.index(f"inf_{i}") for i in range(n_informative)],
        "redundant": [names.index(f"red_{i}") for i in range(n_redundant_pairs)],
        "weights": w,
    }
    return X, y, truth


CLINICAL_COHORT_NAMES = [
    "Sex",
    "Age",
    "Duration",
    "CountA",
    "CountB",
    "CountTotal",
    "Ordinal",
    "Binary_A",
    "Binary_B",
    "Binary_C",
    "Binary_D",
    "Binary_E",
    "Binary_F",
    "Binary_G",
]


def make_clinical_cohort(
    n_samples: int = 31,
    truth: Sequence[str] = ("Age", "CountTotal", "Ordinal"),
    effect: float = 0.9,
    intercept: float = -0.6,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Synthetic cohort with the structure of a small small clinical cohort.

    Fourteen columns: sex (binary), age and disease duration (continuous,
    correlated), two zero-inflated event counts and their sum, an ordinal score
    (1-6) and seven binary indicators (comorbidities, categories). The binary
    outcome is a logistic function of the
    ``truth`` columns (standardised, coefficient ``effect`` with a fixed sign
    per variable). Draws with fewer than three cases in a class are redrawn.

    Returns ``(X, y, truth_info)`` with the indices of the true predictors.
    """
    rng = np.random.default_rng(random_state)
    names = CLINICAL_COHORT_NAMES
    unknown = [t for t in truth if t not in names]
    if unknown:
        raise ValueError(f"unknown truth columns {unknown}; choose from {names}")
    sign = {
        "Age": +1,
        "Duration": -1,
        "CountA": -1,
        "CountB": -1,
        "CountTotal": -1,
        "Ordinal": -1,
    }
    for _ in range(100):
        N = n_samples
        age = rng.normal(35, 12, N).clip(18, 70)
        dur = (0.6 * (age - 35) + rng.normal(0, 9, N) + 20).clip(1, 60)
        tfs = rng.poisson(4, N) * 12 * rng.binomial(1, 0.8, N)
        tgs = rng.poisson(1.2, N) * 12 * rng.binomial(1, 0.5, N)
        ts = tfs + tgs
        asm = rng.integers(1, 7, N).astype(float)
        sex = rng.binomial(1, 0.5, N)
        dummies = np.column_stack(
            [rng.binomial(1, p, N) for p in (0.15, 0.35, 0.25, 0.12, 0.08, 0.40, 0.25)]
        )
        X = pd.DataFrame(
            np.column_stack([sex, age, dur, tfs, tgs, ts, asm, dummies]).astype(float),
            columns=names,
        )

        def z(v: np.ndarray) -> np.ndarray:
            return (v - v.mean()) / (v.std() + 1e-9)

        logit = intercept + sum(sign.get(t, -1) * effect * z(X[t].to_numpy()) for t in truth)
        y = (rng.random(N) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
        if y.sum() >= 3 and (1 - y).sum() >= 3:
            break
    else:  # pragma: no cover - only with pathological parameters
        raise RuntimeError("could not draw a cohort with both classes")
    info = {"truth": [names.index(t) for t in truth], "truth_names": list(truth)}
    return X, y, info
