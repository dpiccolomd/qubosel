"""Bootstrap selection stability."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import clone


def nogueira_stability(Z: NDArray) -> float:
    """Nogueira, Sechidis & Brown (2018) stability of a selection matrix ``Z`` (M x d).

    ``Phi = 1 - mean_f s_f^2 / (kbar/d (1 - kbar/d))`` with unbiased
    ``s_f^2 = M/(M-1) p_f (1 - p_f)``. Returns NaN when ``kbar`` is 0 or ``d``
    or when fewer than two subsets are given.
    """
    Za = np.asarray(Z, dtype=np.float64)
    M, d = Za.shape
    if M < 2:
        return float("nan")
    p = Za.mean(axis=0)
    kbar = Za.sum(axis=1).mean()
    denom = (kbar / d) * (1.0 - kbar / d)
    if denom <= 0.0:
        return float("nan")
    s2 = M / (M - 1.0) * p * (1.0 - p)
    return float(1.0 - s2.mean() / denom)


def mean_pairwise_jaccard(Z: NDArray) -> float:
    Za = np.asarray(Z, dtype=bool)
    M = Za.shape[0]
    if M < 2:
        return float("nan")
    vals = []
    for i, j in combinations(range(M), 2):
        inter = np.logical_and(Za[i], Za[j]).sum()
        union = np.logical_or(Za[i], Za[j]).sum()
        vals.append(1.0 if union == 0 else inter / union)
    return float(np.mean(vals))


@dataclass
class StabilityResult:
    frequencies: NDArray[np.float64]
    nogueira: float
    jaccard_mean: float
    subsets: NDArray[np.int8]
    n_resamples: int
    names: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_frame(self):
        import pandas as pd

        names = self.names or [f"x{i}" for i in range(self.frequencies.shape[0])]
        return (
            pd.DataFrame({"feature": names, "frequency": self.frequencies})
            .sort_values("frequency", ascending=False)
            .reset_index(drop=True)
        )


def _resample(rng: np.random.Generator, y: np.ndarray, max_tries: int = 100) -> np.ndarray:
    n = y.shape[0]
    for _ in range(max_tries):
        idx = rng.integers(0, n, size=n)
        if np.unique(y[idx]).shape[0] >= 2:
            return idx
    raise RuntimeError("could not draw a resample with both classes")


def bootstrap_stability(
    selector,
    X,
    y,
    n_resamples: int = 200,
    random_state: int | None = None,
    n_jobs: int | None = None,
    names: list[str] | None = None,
) -> StabilityResult:
    """Refit a cloned ``selector`` on bootstrap resamples and summarise its selections.

    The selector must expose ``get_support()`` after ``fit``. Single-class
    resamples are redrawn (up to 100 tries).
    """
    from joblib import Parallel, delayed

    Xa = np.asarray(X)
    ya = np.asarray(y)
    if names is None and hasattr(X, "columns"):
        names = [str(c) for c in X.columns]
    rng = np.random.default_rng(random_state)
    draws = [_resample(rng, ya) for _ in range(int(n_resamples))]

    def one(idx):
        est = clone(selector)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(Xa[idx], ya[idx])
        return np.asarray(est.get_support(), dtype=np.int8)

    subsets = (
        np.vstack(Parallel(n_jobs=n_jobs)(delayed(one)(idx) for idx in draws))
        if draws
        else np.zeros((0, Xa.shape[1]), dtype=np.int8)
    )
    freq = subsets.mean(axis=0) if len(draws) else np.zeros(Xa.shape[1])
    return StabilityResult(
        freq,
        nogueira_stability(subsets),
        mean_pairwise_jaccard(subsets),
        subsets,
        len(draws),
        names,
    )
