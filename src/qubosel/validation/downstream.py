"""Fold-enclosed downstream evaluation of candidate feature sets."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score


def bootstrap_auc_ci(
    y,
    prob,
    n_boot: int = 2000,
    random_state: int | None = 0,
    alpha: float = 0.05,
    max_tries: int = 100,
) -> tuple[float, float]:
    """Percentile bootstrap CI of the AUC of held-out probabilities (redraw if single-class)."""
    ya, pa = np.asarray(y), np.asarray(prob)
    rng = np.random.default_rng(random_state)
    n = ya.shape[0]
    aucs = np.empty(n_boot)
    for b in range(n_boot):
        for _ in range(max_tries):
            idx = rng.integers(0, n, size=n)
            if np.unique(ya[idx]).shape[0] == 2:
                break
        else:
            raise RuntimeError("could not draw a two-class resample")
        aucs[b] = roc_auc_score(ya[idx], pa[idx])
    lo, hi = np.quantile(aucs, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def evaluate_downstream(
    feature_sets: Mapping[str, Sequence[int] | Sequence[str] | np.ndarray],
    X,
    y,
    estimator,
    cv=None,
    n_boot: int = 2000,
    random_state: int | None = 42,
):
    """Cross-validate ``estimator`` on each feature subset.

    Returns a DataFrame with fold-mean AUC/accuracy and pooled sensitivity,
    specificity, Brier score and a bootstrap AUC CI of the pooled held-out
    probabilities. Feature sets may be index arrays, boolean masks or column
    names (if ``X`` is a DataFrame).
    """
    import pandas as pd

    ya = np.asarray(y)
    if cv is None:
        cv = StratifiedKFold(5, shuffle=True, random_state=random_state)
    rows = []
    for name, feats in feature_sets.items():
        feats = np.asarray(list(feats))
        if feats.dtype == bool:
            cols = np.flatnonzero(feats)
            Xs = np.asarray(X)[:, cols]
        elif feats.dtype.kind in "iu":
            Xs = np.asarray(X)[:, feats]
        else:
            Xs = np.asarray(X[list(feats)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auc = cross_val_score(clone(estimator), Xs, ya, cv=cv, scoring="roc_auc")
            acc = cross_val_score(clone(estimator), Xs, ya, cv=cv, scoring="accuracy")
            prob = cross_val_predict(clone(estimator), Xs, ya, cv=cv, method="predict_proba")[:, 1]
        pred = (prob >= 0.5).astype(int)
        tp = int(np.sum((pred == 1) & (ya == 1)))
        tn = int(np.sum((pred == 0) & (ya == 0)))
        fp = int(np.sum((pred == 1) & (ya == 0)))
        fn = int(np.sum((pred == 0) & (ya == 1)))
        lo, hi = (
            bootstrap_auc_ci(ya, prob, n_boot=n_boot, random_state=random_state)
            if n_boot > 0
            else (float("nan"), float("nan"))
        )
        rows.append(
            {
                "feature_set": name,
                "n_features": Xs.shape[1],
                "auc_mean": float(np.mean(auc)),
                "auc_std": float(np.std(auc)),
                "acc_mean": float(np.mean(acc)),
                "acc_std": float(np.std(acc)),
                "pooled_auc": float(roc_auc_score(ya, prob)),
                "auc_ci_low": lo,
                "auc_ci_high": hi,
                "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
                "specificity": tn / (tn + fp) if tn + fp else float("nan"),
                "brier": float(brier_score_loss(ya, prob)),
                "accuracy_pooled": float(accuracy_score(ya, pred)),
            }
        )
    return pd.DataFrame(rows)
