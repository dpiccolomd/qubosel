"""Ground-truth simulations: at which sample size does an objective recover known predictors?

The experiment draws synthetic cohorts with known true predictors
(:func:`qubosel.datasets.make_clinical_cohort`), fits every candidate selector
with an *exact* solver, and reports the exact-recovery rate and the mean
Jaccard index with the true set as a function of ``n``. Solver noise is
excluded on purpose: this isolates the objective and its estimator.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif

from qubosel.datasets import make_clinical_cohort
from qubosel.selector import QUBOFeatureSelector

Scenario = Sequence[str]


def standard_objectives(k: int, include_ksg: bool = True) -> dict[str, Any]:
    """Candidate selectors compared in the recovery experiment (exact solver, exact ``k``).

    All QUBO variants use the Mücke structure at ``alpha = 0.5`` with an exact
    cardinality penalty and exhaustive enumeration, so they differ only in the
    relevance/redundancy estimator:

    * ``mucke_hist10``: plug-in MI, equal-width 10 bins (cardinality-biased);
    * ``mucke_mm_quantile4``: plug-in MI with Miller-Madow correction, 4 quantile bins;
    * ``mucke_permutation``: plug-in MI minus its Monte-Carlo permutation mean (quantile bins);
    * ``mucke_expected``: plug-in MI minus its exact permutation expectation
      (reliable MI, Mandros et al.; quantile bins);
    * ``mucke_ksg``: kNN (Kraskov) MI estimator, no discretisation;
    * ``rank``: ``2|AUC - 0.5|`` relevance and Spearman² redundancy;
    * ``anova_topk``: univariate F-test baseline (no redundancy term).
    """
    common = {
        "k": k,
        "solvers": "brute_force",
        "cardinality": "penalty",
        "lam": "auto",
        "alpha": 0.5,
    }
    objectives: dict[str, Any] = {
        "mucke_hist10": QUBOFeatureSelector(binning="legacy", n_bins=10, **common),
        "mucke_mm_quantile4": QUBOFeatureSelector(
            binning="quantile", n_bins=4, mi_correction="miller_madow", **common
        ),
        "mucke_permutation": QUBOFeatureSelector(
            binning="quantile", n_bins=4, mi_correction="permutation", n_null=50, **common
        ),
        "mucke_expected": QUBOFeatureSelector(
            binning="quantile", n_bins=4, mi_correction="expected", **common
        ),
        "rank": QUBOFeatureSelector(formulation="rank", **common),
        "anova_topk": SelectKBest(f_classif, k=k),
    }
    if include_ksg:
        objectives["mucke_ksg"] = QUBOFeatureSelector(estimator="ksg", ksg_repeats=3, **common)
    return objectives


def _one_replicate(
    objectives: Mapping[str, Any],
    n: int,
    truth: Sequence[str],
    seed: int,
    cohort: Callable[..., tuple],
) -> dict[str, tuple[float, float]]:
    X, y, info = cohort(n_samples=n, truth=truth, random_state=seed)
    T = set(info["truth"])
    out = {}
    for name, sel in objectives.items():
        est = clone(sel)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(X, y)
        S = set(np.flatnonzero(est.get_support()))
        out[name] = (float(S == T), len(S & T) / len(S | T))
    return out


def recovery_curve(
    objectives: Mapping[str, Any],
    n_list: Sequence[int] = (31, 60, 100, 200, 300, 500),
    scenarios: Mapping[str, Scenario] | None = None,
    n_replicates: int = 60,
    random_state: int = 0,
    n_jobs: int | None = None,
    cohort: Callable[..., tuple] = make_clinical_cohort,
    verbose: bool = False,
) -> pd.DataFrame:
    """Exact-recovery rate of the true predictor set for every objective and ``n``.

    Returns a long DataFrame with columns ``scenario, n, method, exact_recovery,
    jaccard, se`` (``se`` is the binomial standard error of the recovery rate).
    Replicate ``r`` at size ``n`` uses seed ``random_state + 1000 n + r`` so
    every objective sees identical cohorts.
    """
    if scenarios is None:
        scenarios = {
            "continuous/ordinal truth": ("Age", "CountTotal", "Ordinal"),
            "one binary truth": ("Age", "Binary_A", "Ordinal"),
        }
    rows = []
    for sc_name, truth in scenarios.items():
        for n in n_list:
            reps = Parallel(n_jobs=n_jobs)(
                delayed(_one_replicate)(objectives, n, truth, random_state + 1000 * n + r, cohort)
                for r in range(int(n_replicates))
            )
            for m in objectives:
                hits = np.array([r[m][0] for r in reps])
                rows.append(
                    {
                        "scenario": sc_name,
                        "n": n,
                        "method": m,
                        "exact_recovery": float(hits.mean()),
                        "jaccard": float(np.mean([r[m][1] for r in reps])),
                        "se": float(np.sqrt(hits.mean() * (1 - hits.mean()) / len(hits))),
                    }
                )
            if verbose:
                print(
                    f"{sc_name} n={n}: "
                    + " | ".join(f"{m} {np.mean([r[m][0] for r in reps]):.2f}" for m in objectives)
                )
    return pd.DataFrame(rows)


def recovery_table(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Pivot ``recovery_curve`` output to an ``n x method`` table of recovery rates."""
    return df[df.scenario == scenario].pivot(index="n", columns="method", values="exact_recovery")
