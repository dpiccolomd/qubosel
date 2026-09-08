import itertools

import numpy as np
import pytest

from qubosel.formulations import (
    add_cardinality_penalty,
    alpha_search,
    auto_penalty,
    build_cmi_qubo,
    build_mucke_qubo,
    build_qubo,
    normalize,
)
from qubosel.solvers.brute_force import BruteForceSolver

from .conftest import random_qubo

bf = BruteForceSolver()


def _random_ir(n, rng):
    imp = rng.uniform(0.05, 1.0, size=n)
    A = rng.uniform(0.0, 0.5, size=(n, n))
    red = 0.5 * (A + A.T)
    np.fill_diagonal(red, 0.0)
    return imp, red


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_prop1_monotone_cardinality(seed):
    rng = np.random.default_rng(seed)
    imp, red = _random_ir(10, rng)
    sizes = [bf.solve(build_mucke_qubo(imp, red, a)).n_selected for a in np.linspace(0, 1, 41)]
    assert all(b >= a for a, b in itertools.pairwise(sizes))
    assert sizes[0] == 0 and sizes[-1] == 10


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_alpha_search_hits_every_k(seed):
    rng = np.random.default_rng(seed)
    n = 8
    imp, red = _random_ir(n, rng)
    for k in range(1, n):
        res = alpha_search(imp, red, k, lambda q: bf.solve(q).x)
        assert res.status == "exact" and res.n_selected == k
        assert int(res.x.sum()) == k and 0 < res.alpha < 1
        assert res.trace[-1] == (res.alpha, k)


def test_alpha_search_inexact_and_errors():
    imp = np.array([1.0, 0.0, 0.0])  # only one feature can ever be selected
    red = np.zeros((3, 3))
    res = alpha_search(imp, red, 2, lambda q: bf.solve(q).x, max_iter=10)
    assert res.status == "inexact" and res.n_selected == 1
    with pytest.raises(ValueError):
        alpha_search(imp, red, 5, lambda q: bf.solve(q).x)


def test_mu_rule_excludes_zero_importance(rng):
    imp = np.array([0.0, 0.5, 0.4, 0.3])
    A = rng.uniform(0, 0.1, size=(4, 4))
    red = 0.5 * (A + A.T)
    np.fill_diagonal(red, 0.0)
    q = build_mucke_qubo(imp, red, alpha=1.0)  # alpha=1: no redundancy term
    assert q.Q[0, 0] == q.metadata["mu"] > 0
    assert q.metadata["n_zero_importance"] == 1
    assert bf.solve(q).x[0] == 0
    with pytest.raises(ValueError):
        build_mucke_qubo(imp, red, alpha=2.0)
    with pytest.raises(ValueError):
        build_mucke_qubo(imp, np.zeros((3, 3)))


def test_penalty_energy_identity(rng):
    q = random_qubo(6, rng)
    k = 2
    pen = add_cardinality_penalty(q, k, lam=3.0)
    for _ in range(20):
        x = rng.integers(0, 2, 6)
        assert pen.energy(x) == pytest.approx(q.energy(x) + 3.0 * (x.sum() - k) ** 2)
    assert pen.metadata["lam"] == 3.0
    assert add_cardinality_penalty(q, k, "legacy").metadata["lam"] == 20.0
    with pytest.raises(ValueError):
        add_cardinality_penalty(q, k, "bogus")
    with pytest.raises(ValueError):
        add_cardinality_penalty(q, -1)


@pytest.mark.parametrize("seed", range(5))
def test_auto_penalty_exact_k(seed):
    rng = np.random.default_rng(seed)
    q = random_qubo(9, rng, scale=3.0)
    for k in (1, 3, 5, 8):
        pen = add_cardinality_penalty(q, k, lam="auto")
        assert bf.solve(pen).n_selected == k
    assert auto_penalty(q) > 1.0


def test_legacy_penalty_is_soft():
    """Documented counterexample: lam = 10k does not enforce k when gains are large."""
    n, k = 6, 2
    Q = -100.0 * (np.ones((n, n)) - np.eye(n))
    from qubosel.qubo import QUBO

    q = QUBO(Q)
    pen = add_cardinality_penalty(q, k, lam="legacy")
    assert bf.solve(pen).n_selected == n != k
    assert bf.solve(normalize(pen)).n_selected == n
    assert bf.solve(add_cardinality_penalty(q, k, lam="auto")).n_selected == k


def test_normalize(rng):
    q = random_qubo(5, rng, scale=4.0)
    q.offset = 2.0
    nq = normalize(q)
    assert nq.max_abs() == pytest.approx(1.0)
    s = nq.metadata["normalize_scale"]
    assert nq.offset == pytest.approx(2.0 / s)
    from qubosel.qubo import QUBO

    z = QUBO(np.zeros((3, 3)))
    assert np.array_equal(normalize(z).Q, z.Q)


def test_build_qubo_variants(rng):
    X = rng.normal(size=(50, 5))
    y = (X[:, 0] + rng.normal(scale=0.5, size=50) > 0).astype(int)
    qm = build_qubo(X, y, formulation="mucke", alpha=0.7, names=list("abcde"))
    assert qm.names == list("abcde") and qm.metadata["formulation"] == "mucke"
    qc = build_qubo(X, y, formulation="cmi")
    assert qc.metadata["formulation"] == "cmi"
    assert qc.Q[0, 0] == pytest.approx(-qc.metadata["I"][0])
    with pytest.raises(ValueError):
        build_qubo(X, y, formulation="bogus")
    with pytest.raises(ValueError):
        build_qubo(X, y[:-1])
    with pytest.raises(ValueError):
        build_cmi_qubo(np.ones(3), np.zeros((2, 2)))


def test_build_qubo_rank_and_ksg(rng):
    X = rng.normal(size=(80, 5))
    y = (X[:, 0] + rng.normal(scale=0.5, size=80) > 0).astype(int)
    qr = build_qubo(X, y, formulation="rank", alpha=0.6)
    assert qr.metadata["estimator"] == "rank" and qr.metadata["I"][0] == qr.metadata["I"].max()
    qk = build_qubo(X, y, estimator="ksg", ksg_repeats=1)
    assert qk.metadata["estimator"] == "ksg" and qk.metadata["I"].shape == (5,)
    with pytest.raises(ValueError):
        build_qubo(X, y, formulation="cmi", estimator="ksg")
    with pytest.raises(ValueError):
        build_qubo(X, y, estimator="bogus")
