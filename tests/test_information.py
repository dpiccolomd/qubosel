import numpy as np
import pytest
from sklearn.metrics import mutual_info_score

from qubosel.information import (
    cmi_matrix,
    conditional_mi,
    default_n_bins,
    discretize,
    entropy,
    mi_matrix,
    mutual_info,
)


def test_default_n_bins():
    assert default_n_bins(31) == 2
    assert default_n_bins(100) == 5
    assert default_n_bins(4) == 2


def test_mi_self_is_entropy(rng):
    a = rng.integers(0, 4, size=200)
    assert mutual_info(a, a) == pytest.approx(entropy(a))


def test_mi_matches_sklearn(rng):
    a = rng.integers(0, 3, size=300)
    b = (a + rng.integers(0, 2, size=300)) % 3
    assert mutual_info(a, b) == pytest.approx(mutual_info_score(a, b), abs=1e-12)
    assert mutual_info(a, b, base=2) == pytest.approx(mutual_info_score(a, b) / np.log(2))


def test_chain_rule(rng):
    x = rng.integers(0, 3, size=400)
    z = rng.integers(0, 2, size=400)
    y = (x + z + rng.integers(0, 2, size=400)) % 3
    joint = np.stack([y, z], axis=1)
    _, yz = np.unique(joint, axis=0, return_inverse=True)
    lhs = conditional_mi(x, y, z)
    rhs = mutual_info(x, yz.ravel()) - mutual_info(x, z)
    assert lhs == pytest.approx(rhs, abs=1e-12)


def test_legacy_binning_matches_histogramdd(rng):
    X = rng.normal(size=(50, 3))
    X[:, 1] = rng.integers(0, 4, size=50)  # 4 unique values -> 4 bins
    codes = discretize(X, method="legacy")
    for j in range(3):
        col = X[:, j]
        nb = min(np.unique(col).shape[0], 10)
        scaled = (col - col.min()) / (col.max() - col.min())
        hist, _ = np.histogramdd(scaled[:, None], bins=[nb])
        counts = np.bincount(codes[:, j], minlength=nb)
        assert np.array_equal(hist.astype(int), counts)


def test_quantile_binning_keeps_binary_and_onehot(rng):
    X = np.column_stack([rng.integers(0, 2, 40), rng.normal(size=40)])
    codes = discretize(X)
    assert np.array_equal(codes[:, 0], X[:, 0].astype(int))
    assert codes[:, 1].max() <= default_n_bins(40) - 1
    codes5 = discretize(X, n_bins=5)
    assert codes5[:, 1].max() == 4


def test_discretize_errors():
    with pytest.raises(ValueError):
        discretize(np.zeros(3))
    with pytest.raises(ValueError):
        discretize(np.array([[np.nan, 1.0]]))
    with pytest.raises(ValueError):
        discretize(np.zeros((3, 1)), method="foo")
    with pytest.raises(ValueError):
        discretize(np.zeros((3, 1)), n_bins=0)
    with pytest.raises(ValueError):
        entropy([0, 1], correction="bogus")
    assert entropy(np.zeros(0)) == 0.0
    assert discretize(np.ones((5, 1)), method="legacy").max() == 0


def test_miller_madow_reduces_bias():
    p = np.array([0.4, 0.3, 0.2, 0.1])
    true_h = -(p * np.log(p)).sum()
    rng = np.random.default_rng(0)
    plug, mm = [], []
    for _ in range(200):
        s = rng.choice(4, size=30, p=p)
        plug.append(entropy(s))
        mm.append(entropy(s, correction="miller_madow"))
    assert abs(np.mean(mm) - true_h) < abs(np.mean(plug) - true_h)
    assert np.mean(plug) < true_h


def test_mi_and_cmi_matrices(rng):
    X = rng.integers(0, 3, size=(100, 4))
    y = rng.integers(0, 2, size=100)
    imp, red = mi_matrix(X, y)
    assert imp.shape == (4,) and red.shape == (4, 4)
    assert np.allclose(red, red.T) and np.all(np.diag(red) == 0)
    assert red[0, 1] == pytest.approx(mutual_info(X[:, 0], X[:, 1]))
    C = cmi_matrix(X, y)
    assert np.all(np.diag(C) == 0)
    assert C[0, 1] == pytest.approx(conditional_mi(X[:, 1], y, X[:, 0]))


def test_permutation_correction_removes_cardinality_bias():
    from qubosel.information import mi_matrix

    rng = np.random.default_rng(0)
    n = 31
    y = rng.integers(0, 2, size=n)
    X = np.column_stack([rng.integers(0, 10, n), rng.integers(0, 2, n), rng.integers(0, 4, n)])
    imp_plug, red_plug = mi_matrix(X, y)
    imp_corr, red_corr = mi_matrix(X, y, correction="permutation", n_null=200, random_state=1)
    # plug-in importance of an *independent* 10-level column is inflated by its cardinality
    assert imp_plug[0] > imp_plug[1] and imp_plug[0] > 0.05
    assert np.all(imp_corr <= imp_plug + 1e-12) and np.all(imp_corr >= 0)
    assert imp_corr[0] < 0.5 * imp_plug[0]
    assert np.all(red_corr <= red_plug + 1e-12)
    # deterministic given the seed
    again = mi_matrix(X, y, correction="permutation", n_null=200, random_state=1)
    assert np.allclose(again[0], imp_corr) and np.allclose(again[1], red_corr)
    # a real association survives the correction
    y2 = (X[:, 0] >= 5).astype(int)
    imp2, _ = mi_matrix(X, y2, correction="permutation", n_null=100, random_state=0)
    assert imp2[0] > 0.3
    with pytest.raises(ValueError):
        entropy(y, correction="permutation")


def test_permutation_correction_conditional_matrices():
    from qubosel.information import cmi_matrix, conditional_redundancy_matrix

    rng = np.random.default_rng(3)
    X = rng.integers(0, 5, size=(40, 3))
    y = rng.integers(0, 2, size=40)
    C = cmi_matrix(X, y, correction="permutation", n_null=30, random_state=0)
    D = conditional_redundancy_matrix(X, y, correction="permutation", n_null=30, random_state=0)
    assert C.shape == (3, 3) and np.all(np.diag(C) == 0) and np.all(C >= 0)
    assert np.allclose(D, D.T) and np.all(D >= 0)
    assert np.all(C <= cmi_matrix(X, y) + 1e-12)
    assert np.all(D <= conditional_redundancy_matrix(X, y) + 1e-12)


def test_miller_madow_mi_is_subtractive_and_reduces_bias():
    rng = np.random.default_rng(1)
    plug, mm = [], []
    for _ in range(200):
        a = rng.integers(0, 6, size=30)
        b = rng.integers(0, 2, size=30)  # independent of a
        plug.append(mutual_info(a, b))
        mm.append(mutual_info(a, b, correction="miller_madow"))
    assert np.mean(mm) < np.mean(plug)
    assert abs(np.mean(mm)) < 0.5 * np.mean(plug)


def test_rank_relevance_and_spearman_redundancy():
    from qubosel.information import rank_relevance, spearman_redundancy

    rng = np.random.default_rng(0)
    n = 200
    x0 = rng.normal(size=n)
    y = (x0 + rng.normal(scale=0.5, size=n) > 0).astype(int)
    X = np.column_stack([x0, rng.normal(size=n), np.ones(n), -x0 + rng.normal(scale=0.1, size=n)])
    rel = rank_relevance(X, y)
    assert rel[0] > 0.6 and rel[1] < 0.2 and rel[2] == 0.0
    red = spearman_redundancy(X)
    assert red.shape == (4, 4) and np.allclose(red, red.T) and np.all(np.diag(red) == 0)
    assert red[0, 3] > 0.9 and red[0, 1] < 0.1 and red[0, 2] == 0.0
    with pytest.raises(ValueError):
        rank_relevance(X, np.arange(n) % 3)


def test_ksg_mi_matrix():
    from qubosel.information import ksg_mi_matrix

    rng = np.random.default_rng(0)
    n = 150
    x0 = rng.normal(size=n)
    y = (x0 + rng.normal(scale=0.5, size=n) > 0).astype(int)
    X = np.column_stack(
        [x0, rng.normal(size=n), rng.integers(0, 2, n), x0 + rng.normal(scale=0.2, size=n)]
    )
    imp, red = ksg_mi_matrix(X, y, n_repeats=2, random_state=0)
    assert imp.shape == (4,) and red.shape == (4, 4)
    assert imp[0] > imp[1] and np.all(imp >= 0)
    assert red[0, 3] > red[0, 1] and np.allclose(red, red.T) and np.all(np.diag(red) == 0)


def test_expected_mi_matches_permutation_mean():
    from qubosel.information import (
        cmi_matrix,
        conditional_redundancy_matrix,
        expected_conditional_mi,
        expected_mi,
        mi_matrix,
        null_mean_mi,
    )

    rng = np.random.default_rng(0)
    a = rng.integers(0, 6, size=40)
    b = rng.integers(0, 2, size=40)
    emi = expected_mi(a, b)
    mc = null_mean_mi(a, b, np.random.default_rng(1), 3000, None)
    assert emi == pytest.approx(mc, abs=0.01)
    assert expected_mi(a, b, base=2) == pytest.approx(emi / np.log(2))
    z = rng.integers(0, 2, size=40)
    ecmi = expected_conditional_mi(a, b, z)
    from qubosel.information import conditional_mi

    g = np.random.default_rng(2)
    mc_c = []
    for _ in range(2000):
        bp = b.copy()
        for val in (0, 1):
            idx = np.flatnonzero(z == val)
            bp[idx] = g.permutation(b[idx])
        mc_c.append(conditional_mi(a, bp, z))
    assert ecmi == pytest.approx(float(np.mean(mc_c)), abs=0.01)
    X = np.column_stack([a, rng.integers(0, 10, 40), b])
    imp_e, red_e = mi_matrix(X, b, correction="expected")
    imp_p, red_p = mi_matrix(X, b, correction="permutation", n_null=300, random_state=0)
    assert np.allclose(imp_e, imp_p, atol=0.03) and np.allclose(red_e, red_p, atol=0.03)
    assert np.array_equal(mi_matrix(X, b, correction="expected")[0], imp_e)  # deterministic
    assert np.all(cmi_matrix(X, b, correction="expected") >= 0)
    D = conditional_redundancy_matrix(X, b, correction="expected")
    assert np.allclose(D, D.T) and np.all(D >= 0)
    with pytest.raises(ValueError):
        entropy(b, correction="expected")
