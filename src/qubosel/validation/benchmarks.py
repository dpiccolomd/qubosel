"""Classical feature-selection baselines at the same cardinality."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, SelectorMixin, f_classif
from sklearn.linear_model import LassoCV
from sklearn.utils.validation import check_is_fitted, validate_data


class LassoTopK(SelectorMixin, BaseEstimator):
    """Keep the ``k`` features with the largest |coef| of a ``LassoCV`` fit."""

    def __init__(
        self, k: int = 3, cv: int = 5, max_iter: int = 10000, random_state: int | None = None
    ):
        self.k = k
        self.cv = cv
        self.max_iter = max_iter
        self.random_state = random_state

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        return tags

    def fit(self, X, y):
        X, y = validate_data(self, X, y, dtype=np.float64)
        cv = max(2, min(int(self.cv), X.shape[0]))
        model = LassoCV(cv=cv, max_iter=self.max_iter, random_state=self.random_state)
        model.fit(X, y)
        self.coef_ = model.coef_
        k = min(int(self.k), X.shape[1])
        order = np.lexsort((np.arange(X.shape[1]), -np.abs(self.coef_)))
        mask = np.zeros(X.shape[1], dtype=bool)
        mask[order[:k]] = True
        self.support_ = mask
        return self

    def _get_support_mask(self):
        check_is_fitted(self, "support_")
        return self.support_


def classical_selectors(
    k: int, random_state: int | None = 42, n_estimators: int = 100
) -> dict[str, BaseEstimator]:
    """``anova`` (SelectKBest f_classif), ``lasso_topk`` and ``rfe_rf`` at cardinality ``k``."""
    return {
        "anova": SelectKBest(score_func=f_classif, k=k),
        "lasso_topk": LassoTopK(k=k, random_state=random_state),
        "rfe_rf": RFE(
            RandomForestClassifier(n_estimators=n_estimators, random_state=random_state),
            n_features_to_select=k,
            step=1,
        ),
    }


def compare_selectors(selectors: dict[str, BaseEstimator], X, y, names=None):
    """Fit every selector on ``(X, y)`` and return a feature x method 0/1 table."""
    import pandas as pd

    Xa, ya = np.asarray(X), np.asarray(y)
    if names is None:
        names = (
            [str(c) for c in X.columns]
            if hasattr(X, "columns")
            else [f"x{i}" for i in range(Xa.shape[1])]
        )
    cols = {}
    for name, sel in selectors.items():
        est = clone(sel).fit(Xa, ya)
        cols[name] = np.asarray(est.get_support(), dtype=int)
    return pd.DataFrame(cols, index=names)
