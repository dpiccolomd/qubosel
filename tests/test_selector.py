import warnings

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import parametrize_with_checks

from qubosel import QUBOFeatureSelector
from qubosel.datasets import make_clinical_synthetic
from qubosel.solvers import SimulatedAnnealingSolver

EXPECTED_FAILED: dict[str, str] = {}


@parametrize_with_checks(
    [QUBOFeatureSelector(k=2, solvers="brute_force")],
    expected_failed_checks=lambda est: EXPECTED_FAILED,
)
def test_sklearn_compatible(estimator, check):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check(estimator)


@pytest.fixture(scope="module")
def data():
    X, y, truth = make_clinical_synthetic(n_samples=80, random_state=3)
    return X, y, truth


def test_pandas_feature_names(data):
    X, y, _ = data
    sel = QUBOFeatureSelector(k=3, random_state=0).fit(X, y)
    names = list(sel.get_feature_names_out())
    assert len(names) == 3 and set(names) <= set(X.columns)
    assert sel.transform(X).shape == (80, 3)
    assert sel.n_features_in_ == 14
    assert sel.alpha_status_[3] == "exact"
    assert sel.qubos_[3].names == list(X.columns)


def test_clone_and_repeatability(data):
    X, y, _ = data
    a = QUBOFeatureSelector(
        k=3, solvers=[SimulatedAnnealingSolver(num_reads=5), "sb"], n_repeats=2, random_state=1
    )
    b = clone(a)
    assert a.get_params() == b.get_params()
    a.fit(X, y)
    b.fit(X, y)
    assert np.array_equal(a.support_, b.support_)
    assert np.array_equal(a.votes_, b.votes_)


def test_n_jobs_reproducible(data):
    X, y, _ = data
    kw = dict(
        ks=[2, 3], solvers=("sa", {"num_reads": 3, "num_sweeps": 50}), n_repeats=2, random_state=4
    )
    a = QUBOFeatureSelector(n_jobs=1, **kw).fit(X, y)
    b = QUBOFeatureSelector(n_jobs=2, **kw).fit(X, y)
    assert np.array_equal(a.votes_, b.votes_)
    assert [r.seed for r in a.records_] == [r.seed for r in b.records_]


def test_pipeline_cross_val(data):
    X, y, _ = data
    pipe = Pipeline(
        [
            ("sel", QUBOFeatureSelector(k=3, random_state=0)),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )
    scores = cross_val_score(pipe, X, y, cv=3, scoring="roc_auc")
    assert scores.shape == (3,) and np.all(scores > 0.5)


def test_penalty_and_none_cardinality(data):
    X, y, _ = data
    p = QUBOFeatureSelector(k=4, cardinality="penalty", lam="auto", solvers="brute_force").fit(X, y)
    assert p.support_.sum() == 4
    p2 = QUBOFeatureSelector(
        k=4,
        cardinality="penalty",
        lam="legacy",
        normalize=True,
        formulation="cmi",
        binning="legacy",
        base=2,
        solvers="brute_force",
    ).fit(X, y)
    assert p2.support_.sum() == 4 and "normalize_scale" in p2.qubos_[4].metadata
    u = QUBOFeatureSelector(cardinality=None, alpha=0.9, solvers="brute_force").fit(X, y)
    assert u.support_.sum() >= 1


def test_default_k_and_solver_auto(data):
    X, y, _ = data
    sel = QUBOFeatureSelector(random_state=0).fit(X, y)
    assert sel.ks_ == [4]  # round(sqrt(14))
    assert sel.solvers_[0].name == "brute_force"
    big = QUBOFeatureSelector(brute_force_max_n=5, alpha_oracle="solver", random_state=0).fit(X, y)
    assert big.solvers_[0].name == "sa"


def test_warnings_and_errors(data):
    X, y, _ = data
    Xa = np.asarray(X)
    with pytest.warns(UserWarning, match="clamping"):
        QUBOFeatureSelector(k=20, solvers="brute_force").fit(Xa[:, :5], y)
    Xc = Xa.copy()
    Xc[:, 0] = 1.0
    with pytest.warns(UserWarning, match="constant"):
        QUBOFeatureSelector(k=2, solvers="brute_force").fit(Xc, y)
    with pytest.raises(ValueError):
        QUBOFeatureSelector(k=0).fit(Xa, y)
    with pytest.raises(ValueError):
        QUBOFeatureSelector(formulation="x").fit(Xa, y)
    with pytest.raises(ValueError):
        QUBOFeatureSelector(cardinality="x").fit(Xa, y)
    with pytest.raises(ValueError):
        QUBOFeatureSelector(formulation="cmi", cardinality="alpha").fit(Xa, y)
    with pytest.raises(ValueError):
        QUBOFeatureSelector(alpha_oracle="x").fit(Xa, y)
    with pytest.raises(ValueError):
        QUBOFeatureSelector(solvers=[]).fit(Xa, y)
    with pytest.raises(ValueError):
        QUBOFeatureSelector().fit(Xa[:1], y[:1])


def test_no_consensus_fallback(data):
    X, y, _ = data
    with pytest.warns(UserWarning, match="min_votes"):
        sel = QUBOFeatureSelector(k=3, solvers="brute_force", min_votes=5).fit(X, y)
    assert sel.support_.sum() == 3


def test_callable_formulation(data):
    from qubosel import build_qubo

    X, y, _ = data

    def custom(Xa, ya, names):
        return build_qubo(Xa, ya, formulation="cmi", binning="legacy", base=2, names=names)

    sel = QUBOFeatureSelector(
        k=3,
        formulation=custom,
        cardinality="penalty",
        lam="legacy",
        normalize=True,
        solvers="brute_force",
        oversize_policy="keep",
    ).fit(X, y)
    assert sel.support_.sum() == 3 and sel.base_qubo_.names == list(X.columns)
    with pytest.raises(ValueError):
        QUBOFeatureSelector(formulation=custom, cardinality="alpha").fit(np.asarray(X), y)


def test_selector_with_permutation_corrected_mi(data):
    X, y, _ = data
    sel = QUBOFeatureSelector(
        k=3, solvers="brute_force", mi_correction="permutation", n_null=20, random_state=0
    ).fit(X, y)
    assert sel.support_.sum() == 3
    assert sel.base_qubo_.metadata["mi_correction"] == "permutation"
    again = QUBOFeatureSelector(
        k=3, solvers="brute_force", mi_correction="permutation", n_null=20, random_state=0
    ).fit(X, y)
    assert np.array_equal(sel.support_, again.support_)


def test_rank_and_ksg_selectors(data):
    X, y, _truth = data
    r = QUBOFeatureSelector(k=3, formulation="rank", solvers="brute_force").fit(X, y)
    assert r.support_.sum() == 3 and r.alpha_status_[3] == "exact"
    k = QUBOFeatureSelector(
        k=3, estimator="ksg", ksg_repeats=1, solvers="brute_force", random_state=0
    ).fit(X, y)
    assert k.support_.sum() == 3


def test_selector_with_expected_mi(data):
    X, y, _ = data
    sel = QUBOFeatureSelector(k=3, solvers="brute_force", mi_correction="expected").fit(X, y)
    assert sel.support_.sum() == 3 and sel.base_qubo_.metadata["mi_correction"] == "expected"
