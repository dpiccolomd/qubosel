import itertools

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from qubosel.qubo import QUBO, Ising
from qubosel.solvers.brute_force import BruteForceSolver

from .conftest import random_qubo


def _sym_matrix(n):
    return arrays(np.float64, (n, n), elements=st.floats(-5, 5, allow_nan=False)).map(
        lambda A: 0.5 * (A + A.T)
    )


@settings(max_examples=60, deadline=None)
@given(
    st.integers(1, 12).flatmap(
        lambda n: st.tuples(
            _sym_matrix(n), st.lists(st.integers(0, 1), min_size=n, max_size=n), st.floats(-3, 3)
        )
    )
)
def test_qubo_ising_identity(data):
    Q, x, off = data
    q = QUBO(Q, offset=off)
    x = np.array(x)
    assert q.energy(x) == pytest.approx(q.to_ising().energy(1 - 2 * x), abs=1e-9)


def test_upper_roundtrip(rng):
    q = random_qubo(6, rng)
    U = q.to_upper()
    assert np.allclose(np.tril(U, -1), 0)
    q2 = QUBO.from_upper(U)
    assert np.allclose(q.Q, q2.Q)
    # dictionary uses the upper representation
    d = q.to_dict()
    assert d[(0, 1)] == pytest.approx(2 * q.Q[0, 1])
    assert d[(2, 2)] == pytest.approx(q.Q[2, 2])


def test_from_upper_folds_lower_triangle():
    full = np.array([[1.0, 2.0], [3.0, 4.0]])
    q = QUBO.from_upper(full)
    assert q.Q[0, 1] == pytest.approx(2.5)


def test_energies_batch(rng):
    q = random_qubo(7, rng)
    X = rng.integers(0, 2, size=(20, 7))
    assert np.allclose(q.energies(X), [q.energy(x) for x in X])


def test_marginal_gain(rng):
    q = random_qubo(9, rng)
    x = rng.integers(0, 2, size=9)
    for i in range(9):
        x1, x0 = x.copy(), x.copy()
        x1[i], x0[i] = 1, 0
        assert q.marginal_gain(x, i) == pytest.approx(q.energy(x1) - q.energy(x0))


def test_validation_errors():
    with pytest.raises(ValueError):
        QUBO(np.array([[1.0, 2.0], [0.0, 1.0]]))
    with pytest.raises(ValueError):
        QUBO(np.zeros((2, 3)))
    q = QUBO(np.eye(2))
    with pytest.raises(ValueError):
        q.energy([0, 2])
    with pytest.raises(ValueError):
        q.energy([0, 1, 1])
    with pytest.raises(ValueError):
        Ising(np.zeros(2), np.array([[0, 1.0], [2.0, 0]]))
    with pytest.raises(ValueError):
        Ising(np.zeros(2), np.zeros((2, 2))).energy([0, 1])
    with pytest.raises(ValueError):
        QUBO(np.eye(2), names=["a"])


def test_ising_local_fields(rng):
    q = random_qubo(5, rng)
    ising = q.to_ising()
    s = rng.choice([-1.0, 1.0], size=5)
    f = ising.local_fields(s)
    for i in range(5):
        s2 = s.copy()
        s2[i] *= -1
        assert ising.energy(s2) - ising.energy(s) == pytest.approx(-2 * s[i] * f[i])


@pytest.mark.parametrize("n", [1, 4, 9, 13])
def test_brute_force_matches_itertools(n, rng):
    q = random_qubo(n, rng)
    best = min(itertools.product([0, 1], repeat=n), key=lambda t: q.energy(np.array(t)[::-1]))
    res = BruteForceSolver(chunk_size=64).solve(q)
    assert res.energy == pytest.approx(q.energy(np.array(best)[::-1]))
    assert res.energy == pytest.approx(q.energy(res.x))
    assert res.n_selected == int(res.x.sum())


def test_brute_force_tie_break_deterministic():
    q = QUBO(np.zeros((5, 5)))  # every assignment has energy 0
    res = BruteForceSolver().solve(q)
    assert res.x.tolist() == [0] * 5 and res.metadata["index"] == 0
    q = QUBO(np.diag([-1.0, 0.0, 0.0]))
    res = BruteForceSolver().solve(q)
    assert res.x.tolist() == [1, 0, 0]


def test_brute_force_limits(rng):
    with pytest.raises(ValueError):
        BruteForceSolver(max_n=3).solve(random_qubo(4, rng))
    with pytest.raises(ValueError):
        BruteForceSolver(max_n=3).all_energies(random_qubo(4, rng))
    q = random_qubo(5, rng)
    E = BruteForceSolver().all_energies(q)
    assert E.shape == (32,)
    assert E.min() == pytest.approx(BruteForceSolver().solve(q).energy)


def test_empty_qubo():
    res = BruteForceSolver().solve(QUBO(np.zeros((0, 0)), offset=2.0))
    assert res.n_selected == 0 and res.energy == 2.0
