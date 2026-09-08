import numpy as np
import pytest

pytest.importorskip("pennylane")
pytestmark = pytest.mark.qaoa

from qubosel.qubo import QUBO  # noqa: E402
from qubosel.solvers import get_solver  # noqa: E402
from qubosel.solvers.brute_force import BruteForceSolver  # noqa: E402
from qubosel.solvers.qaoa import QAOASolver, basis_state_energy  # noqa: E402


def _random_qubo(n, seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    return QUBO(0.5 * (A + A.T), offset=0.3)


def test_basis_state_expectation_equals_qubo_energy():
    q = _random_qubo(5, 0)
    rng = np.random.default_rng(1)
    for _ in range(6):
        x = rng.integers(0, 2, 5)
        assert basis_state_energy(q, x) == pytest.approx(q.energy(x), abs=1e-8)


def test_diagonal_qubo_p1_finds_optimum():
    q = QUBO(np.diag([-1.0, 0.5, -0.8, 0.2, -0.3, 0.9]))
    res = QAOASolver(p=1, shots=500, max_iter=60).solve(q, seed=0)
    assert res.energy == pytest.approx(BruteForceSolver().solve(q).energy)
    assert res.metadata["approximation_ratio"] == pytest.approx(1.0)
    assert "pennylane_version" in res.metadata


def test_p2_quality_on_n8():
    q = _random_qubo(8, 3)
    res = QAOASolver(p=2, shots=1000, max_iter=80).solve(q, seed=0)
    all_e = BruteForceSolver().all_energies(q)
    assert res.energy <= np.quantile(all_e, 0.2)
    assert res.metadata["approximation_ratio"] >= 0.7
    assert res.energy == pytest.approx(q.energy(res.x))


def test_seed_determinism_and_readouts():
    q = _random_qubo(5, 4)
    s = QAOASolver(p=1, shots=200, max_iter=30)
    a, b = s.solve(q, seed=11), s.solve(q, seed=11)
    assert np.array_equal(a.x, b.x) and np.array_equal(a.metadata["params"], b.metadata["params"])
    mp = QAOASolver(p=1, shots=200, max_iter=30, readout="most_probable").solve(q, seed=11)
    assert np.array_equal(mp.x, a.metadata["most_probable"])
    with pytest.raises(ValueError):
        QAOASolver(readout="x", max_iter=5, shots=10).solve(q, seed=0)


def test_registry_and_device_factory():
    import pennylane as qp

    calls = []

    def factory(wires, shots, seed):
        calls.append((wires, shots))
        return qp.device("default.qubit", wires=wires, shots=shots, seed=seed)

    s = get_solver("qaoa", p=1, device=factory, shots=100, max_iter=10, n_restarts=2)
    assert isinstance(s, QAOASolver)
    res = s.solve(_random_qubo(4, 5), seed=0)
    assert calls == [(4, None), (4, 100)] and res.x.shape == (4,)
    assert res.metadata["n_function_evals"] > 0


def test_shots_optimization_path():
    res = QAOASolver(p=1, analytic_optimization=False, shots=100, max_iter=10).solve(
        _random_qubo(4, 6), seed=0
    )
    assert res.x.shape == (4,)
