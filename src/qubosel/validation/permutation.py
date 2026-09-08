"""Permutation test for a cross-validated score (selection inside the null)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.model_selection import cross_val_score


@dataclass
class PermutationResult:
    score: float
    null_scores: NDArray[np.float64]
    p_value: float
    n_permutations: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _cv_score(estimator, X, y, cv, scoring) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(np.mean(cross_val_score(clone(estimator), X, y, cv=cv, scoring=scoring)))


def permutation_test(
    estimator,
    X,
    y,
    cv=5,
    n_permutations: int = 1000,
    scoring: str = "roc_auc",
    random_state: int | None = None,
    n_jobs: int | None = None,
) -> PermutationResult:
    """Empirical p-value of the CV score against label permutations.

    Permutation ``i`` uses ``default_rng(seed + i).permutation(y)`` and a fresh
    clone of ``estimator`` (so feature selection inside a Pipeline is part of
    the null). ``p = (1 + #[null >= score]) / (n_permutations + 1)``.
    """
    Xa, ya = np.asarray(X), np.asarray(y)
    seed = 0 if random_state is None else int(random_state)
    score = _cv_score(estimator, Xa, ya, cv, scoring)
    if n_permutations > 0:
        null = Parallel(n_jobs=n_jobs)(
            delayed(_cv_score)(
                estimator, Xa, np.random.default_rng(seed + i).permutation(ya), cv, scoring
            )
            for i in range(int(n_permutations))
        )
        null_arr = np.asarray(null, dtype=np.float64)
    else:
        null_arr = np.zeros(0)
    p = (1.0 + float(np.sum(null_arr >= score))) / (n_permutations + 1.0)
    return PermutationResult(
        score, null_arr, p, int(n_permutations), {"scoring": scoring, "seed": seed}
    )
