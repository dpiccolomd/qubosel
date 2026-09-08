"""Selection under permuted labels: does the selector depend on the outcome at all?

A feature-selection procedure whose optimum does not change when the outcome
is shuffled is not measuring association with the outcome. This control refits
the selector on ``y`` permuted ``n_permutations`` times and reports, for every
feature, how often it is selected under the null, next to the observed
selection. Small-sample plug-in mutual information is the typical culprit: its
positive bias grows with the number of distinct values of a variable, so the
QUBO ends up ranking columns by cardinality.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from sklearn.base import clone


@dataclass
class NullSelectionResult:
    observed: NDArray[np.bool_]
    null_frequency: NDArray[np.float64]
    null_matrix: NDArray[np.int8]
    p_value: NDArray[np.float64]
    n_permutations: int
    same_as_observed: float
    names: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_frame(self):
        import pandas as pd

        names = self.names or [f"x{i}" for i in range(self.observed.shape[0])]
        return (
            pd.DataFrame(
                {
                    "feature": names,
                    "selected": self.observed,
                    "null_frequency": self.null_frequency,
                    "p_value": self.p_value,
                }
            )
            .sort_values(["selected", "null_frequency"], ascending=[False, False])
            .reset_index(drop=True)
        )


def _fit_support(selector, X, y) -> np.ndarray:
    est = clone(selector)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(X, y)
    return np.asarray(est.get_support(), dtype=np.int8)


def permuted_label_selection(
    selector,
    X,
    y,
    n_permutations: int = 200,
    random_state: int | None = None,
    n_jobs: int | None = None,
    names: list[str] | None = None,
) -> NullSelectionResult:
    """Refit ``selector`` on label permutations and compare with the observed selection.

    Returns per-feature null selection frequencies, a one-sided p-value for
    each *selected* feature (``(1 + #[null selects it]) / (n + 1)``; NaN for
    unselected features) and the fraction of permutations whose selected set is
    identical to the observed one. Permutation ``i`` uses
    ``default_rng(seed + i).permutation(y)``.
    """
    Xa, ya = np.asarray(X), np.asarray(y)
    if names is None and hasattr(X, "columns"):
        names = [str(c) for c in X.columns]
    seed = 0 if random_state is None else int(random_state)
    observed = _fit_support(selector, Xa, ya).astype(bool)
    if n_permutations > 0:
        rows = Parallel(n_jobs=n_jobs)(
            delayed(_fit_support)(selector, Xa, np.random.default_rng(seed + i).permutation(ya))
            for i in range(int(n_permutations))
        )
        null = np.vstack(rows)
    else:
        null = np.zeros((0, Xa.shape[1]), dtype=np.int8)
    freq = null.mean(axis=0) if n_permutations > 0 else np.zeros(Xa.shape[1])
    counts = null.sum(axis=0)
    p = np.where(observed, (1.0 + counts) / (n_permutations + 1.0), np.nan)
    same = (
        float(np.mean(np.all(null == observed.astype(np.int8), axis=1)))
        if n_permutations
        else float("nan")
    )
    return NullSelectionResult(
        observed, freq, null, p, int(n_permutations), same, names, {"seed": seed}
    )
