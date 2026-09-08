import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from qubosel import QUBOFeatureSelector
from qubosel.datasets import make_clinical_synthetic
from qubosel.solvers import BruteForceSolver
from qubosel.validation import (
    LassoTopK,
    bootstrap_auc_ci,
    bootstrap_stability,
    classical_selectors,
    compare_selectors,
    evaluate_downstream,
    nogueira_stability,
    permutation_test,
)
from qubosel.validation.stability import mean_pairwise_jaccard


class StubClassifier(ClassifierMixin, BaseEstimator):
    """Scores 1 when the labels are sorted (original) and 0 otherwise."""

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.good_ = bool(np.all(np.diff(y) >= 0))
        return self

    def predict_proba(self, X):
        p = np.linspace(0, 1, X.shape[0]) if self.good_ else np.linspace(1, 0, X.shape[0])
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def test_permutation_p_value_by_hand():
    X = np.zeros((20, 2))
    y = np.repeat([0, 1], 10)
    from sklearn.model_selection import KFold

    cv = KFold(2, shuffle=False)  # keeps y sorted inside every fold on the original labels
    res = permutation_test(StubClassifier(), X, y, cv=cv, n_permutations=19, random_state=0)
    n_ge = int(np.sum(res.null_scores >= res.score))
    assert res.p_value == pytest.approx((1 + n_ge) / 20)
    assert res.n_permutations == 19 and res.null_scores.shape == (19,)
    zero = permutation_test(StubClassifier(), X, y, cv=cv, n_permutations=0)
    assert zero.p_value == 1.0 and zero.null_scores.shape == (0,)


def test_permutation_with_selector_inside(rng):
    X, y, _ = make_clinical_synthetic(n_samples=60, random_state=1)
    pipe = Pipeline(
        [
            ("sel", QUBOFeatureSelector(k=2, solvers="brute_force")),
            ("clf", LogisticRegression(max_iter=300)),
        ]
    )
    res = permutation_test(pipe, X, y, cv=3, n_permutations=5, random_state=0, n_jobs=1)
    assert 0 < res.p_value <= 1 and res.score > 0.5


def test_nogueira_and_jaccard():
    Z = np.tile(np.array([[1, 1, 0, 0, 0]]), (10, 1))
    assert nogueira_stability(Z) == pytest.approx(1.0)
    assert mean_pairwise_jaccard(Z) == pytest.approx(1.0)
    assert np.isnan(nogueira_stability(np.zeros((5, 4))))
    assert np.isnan(nogueira_stability(np.ones((1, 4))))
    assert np.isnan(mean_pairwise_jaccard(np.ones((1, 4))))
    rng = np.random.default_rng(0)
    Z = np.zeros((200, 10), dtype=int)
    for i in range(200):
        Z[i, rng.choice(10, 3, replace=False)] = 1
    phi = nogueira_stability(Z)
    assert -0.1 <= phi <= 0.1
    assert 0 <= mean_pairwise_jaccard(Z) <= 1


def test_bootstrap_stability():
    X, y, _truth = make_clinical_synthetic(n_samples=80, random_state=2)
    sel = QUBOFeatureSelector(k=3, solvers="brute_force")
    res = bootstrap_stability(sel, X, y, n_resamples=8, random_state=0)
    assert res.subsets.shape == (8, 14) and res.frequencies.shape == (14,)
    assert res.names == list(X.columns)
    assert res.to_frame().iloc[0]["frequency"] >= res.to_frame().iloc[-1]["frequency"]
    assert 0 <= res.jaccard_mean <= 1


def test_classical_selectors_and_compare():
    X, y, _ = make_clinical_synthetic(n_samples=80, random_state=2)
    sels = classical_selectors(3, n_estimators=20)
    table = compare_selectors(sels, X, y)
    assert table.shape == (14, 3) and (table.sum(axis=0) == 3).all()
    lt = LassoTopK(k=2).fit(np.asarray(X), y)
    assert lt.get_support().sum() == 2
    table2 = compare_selectors({"lasso": lt}, np.asarray(X), y)
    assert table2.index[0] == "x0"


def test_evaluate_downstream_and_auc_ci():
    X, y, truth = make_clinical_synthetic(n_samples=80, random_state=2)
    sets = {"truth": truth["informative"], "mask": np.arange(14) < 3, "names": list(X.columns[:2])}
    df = evaluate_downstream(sets, X, y, LogisticRegression(max_iter=300), n_boot=50)
    assert df.shape[0] == 3 and set(["auc_mean", "auc_ci_low", "brier"]) <= set(df.columns)
    assert (df["auc_ci_low"] <= df["pooled_auc"]).all()
    assert (df["pooled_auc"] <= df["auc_ci_high"]).all()
    assert df["n_features"].tolist() == [3, 3, 2]
    df0 = evaluate_downstream({"a": [0]}, np.asarray(X), y, LogisticRegression(), n_boot=0)
    assert np.isnan(df0["auc_ci_low"].iloc[0])
    lo, hi = bootstrap_auc_ci(y, np.asarray(X)[:, 0], n_boot=100)
    assert lo <= hi


@pytest.mark.slow
def test_synthetic_recovery():
    hits = 0
    seeds = range(10)
    for s in seeds:
        X, y, truth = make_clinical_synthetic(n_samples=200, signal=2.5, random_state=s)
        sel = QUBOFeatureSelector(k=3, solvers=BruteForceSolver(), random_state=0).fit(X, y)
        picked = set(np.flatnonzero(sel.support_))
        # a redundant copy counts as recovering its informative source
        recovered = {
            i
            for i in truth["informative"]
            if i in picked or (i < len(truth["redundant"]) and truth["redundant"][i] in picked)
        }
        hits += len(recovered) == 3
    assert hits >= 0.8 * len(seeds)


def test_make_clinical_synthetic_errors():
    with pytest.raises(ValueError):
        make_clinical_synthetic(n_features=5)
    with pytest.raises(ValueError):
        make_clinical_synthetic(n_redundant_pairs=5, n_features=30)
    X, _y, _truth = make_clinical_synthetic(n_features=20, random_state=0)
    assert X.shape == (60, 20) and isinstance(X, pd.DataFrame)


def test_permuted_label_selection_detects_label_independence():
    from qubosel.validation import permuted_label_selection

    X, y, _truth = make_clinical_synthetic(n_samples=200, signal=2.5, random_state=0)
    sel = QUBOFeatureSelector(k=3, solvers="brute_force")
    res = permuted_label_selection(sel, X, y, n_permutations=20, random_state=0, n_jobs=1)
    assert res.null_matrix.shape == (20, 14) and res.observed.sum() == 3
    assert res.names == list(X.columns)
    df = res.to_frame()
    assert set(df.columns) == {"feature", "selected", "null_frequency", "p_value"}
    assert np.isnan(res.p_value[~res.observed]).all()
    assert np.all((res.p_value[res.observed] > 0) & (res.p_value[res.observed] <= 1))
    assert 0.0 <= res.same_as_observed <= 1.0
    # an informative selection should not be reproduced by most label shuffles
    assert res.same_as_observed < 0.5

    class ConstantSelector(QUBOFeatureSelector):
        """Selects the first k columns whatever the labels."""

        def fit(self, X, y):
            X, y = np.asarray(X), np.asarray(y)
            self.n_features_in_ = X.shape[1]
            self.support_ = np.arange(X.shape[1]) < self.k
            return self

    const = permuted_label_selection(ConstantSelector(k=2), np.asarray(X), y, n_permutations=10)
    assert const.same_as_observed == 1.0
    assert np.allclose(const.null_frequency[:2], 1.0) and np.allclose(const.p_value[:2], 1.0)
    zero = permuted_label_selection(sel, np.asarray(X), y, n_permutations=0)
    assert np.isnan(zero.same_as_observed) and zero.null_matrix.shape == (0, 14)
