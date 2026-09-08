import numpy as np
import pytest

from qubosel.qubo import QUBO
from qubosel.solvers import (
    SimulatedAnnealingSolver,
    SimulatedBifurcationSolver,
    available_solvers,
    get_solver,
)
from qubosel.solvers.annealing import neal_beta_range
from qubosel.solvers.brute_force import BruteForceSolver

from .conftest import random_qubo


def _instances(seed=0, sizes=(8, 12, 16), per_size=10):
    rng = np.random.default_rng(seed)
    out = []
    for n in sizes:
        for _ in range(per_size):
            q = random_qubo(n, rng)
            out.append((q, BruteForceSolver().solve(q).energy))
    return out


def _hit_rate(solver, instances, tol=0.02):
    hits, gaps = 0, []
    for i, (q, opt) in enumerate(instances):
        res = solver.solve(q, seed=i)
        assert res.energy == pytest.approx(q.energy(res.x))
        if np.isclose(res.energy, opt, atol=1e-9):
            hits += 1
        else:
            gaps.append((res.energy - opt) / abs(opt))
    return hits / len(instances), gaps


@pytest.mark.slow
def test_sa_hit_rate():
    rate, gaps = _hit_rate(SimulatedAnnealingSolver(num_reads=50, num_sweeps=1000), _instances())
    assert rate >= 0.95, (rate, gaps)


@pytest.mark.slow
@pytest.mark.parametrize("variant", ["ballistic", "discrete"])
def test_sb_hit_rate(variant):
    solver = SimulatedBifurcationSolver(variant=variant, n_agents=128, n_steps=2000)
    rate, gaps = _hit_rate(solver, _instances())
    assert rate >= 0.90, (rate, gaps)
    assert all(g <= 0.02 for g in gaps), gaps


@pytest.mark.parametrize("name", ["sa", "sb"])
def test_seed_determinism(name, rng):
    q = random_qubo(10, rng)
    s = get_solver(name)
    a, b = s.solve(q, seed=7), s.solve(q, seed=7)
    assert np.array_equal(a.x, b.x) and a.energy == b.energy
    assert a.metadata["solver"] == name and a.metadata["seed"] == 7


def test_sb_ancilla_vs_field_agree(rng):
    for _ in range(5):
        q = random_qubo(8, rng)
        opt = BruteForceSolver().solve(q).energy
        a = SimulatedBifurcationSolver(linear_mode="ancilla", n_steps=1000).solve(q, seed=1)
        f = SimulatedBifurcationSolver(linear_mode="field", n_steps=1000).solve(q, seed=1)
        assert a.energy == pytest.approx(opt, rel=0.05)
        assert f.energy == pytest.approx(opt, rel=0.05)


def test_sb_readout_zero_and_no_tracking():
    q = QUBO(np.zeros((4, 4)))  # J = 0 -> rms guard, x stays at 0 in places
    res = SimulatedBifurcationSolver(track_best=False, n_steps=50).solve(q, seed=0)
    assert res.energy == 0.0 and set(res.x.tolist()) <= {0, 1}
    s = SimulatedBifurcationSolver()
    S = s._readout(np.array([[0.0, -0.5], [1.0, 1.0]]), use_ancilla=False)
    assert S[0, 0] == 1.0 and S[0, 1] == -1.0
    S = s._readout(np.array([[0.5], [-1.0]]), use_ancilla=True)  # global flip
    assert S.shape == (1, 1) and S[0, 0] == -1.0


def test_sb_fixed_xi0_and_errors(rng):
    q = random_qubo(6, rng)
    res = SimulatedBifurcationSolver(xi0=0.3, n_steps=200).solve(q, seed=0)
    assert res.metadata["xi0"] == 0.3
    with pytest.raises(ValueError):
        SimulatedBifurcationSolver(variant="x").solve(q)
    with pytest.raises(ValueError):
        SimulatedBifurcationSolver(linear_mode="x").solve(q)
    with pytest.raises(ValueError):
        SimulatedBifurcationSolver(backend="x").solve(q)


@pytest.mark.parametrize("seed", range(5))
def test_sa_swap_moves_on_penalty_qubo(seed):
    """Flip-only SA freezes the cardinality on legacy penalty QUBOs; swap moves fix it."""
    from qubosel.datasets import make_clinical_synthetic
    from qubosel.formulations import add_cardinality_penalty, build_qubo, normalize

    X, y, _ = make_clinical_synthetic(n_samples=60, random_state=seed)
    base = build_qubo(X.to_numpy(), y, formulation="cmi", binning="legacy", base=2)
    q = normalize(add_cardinality_penalty(base, 4, lam="legacy"))
    opt = BruteForceSolver().solve(q)
    res = SimulatedAnnealingSolver(num_reads=50, swap_moves=True).solve(q, seed=seed)
    assert res.n_selected == 4
    assert res.energy == pytest.approx(q.energy(res.x))
    assert res.energy == pytest.approx(opt.energy, abs=1e-12)
    again = SimulatedAnnealingSolver(num_reads=50, swap_moves=True).solve(q, seed=seed)
    assert np.array_equal(again.x, res.x)
    assert res.metadata["beta_range"][1] > 1e3  # data-driven cold beta


def test_sa_swap_moves_consistent_energy(rng):
    q = random_qubo(9, rng)
    res = SimulatedAnnealingSolver(num_reads=5, num_sweeps=100, swap_moves=True).solve(q, seed=3)
    assert res.energy == pytest.approx(q.energy(res.x))
    one = SimulatedAnnealingSolver(num_reads=2, num_sweeps=10, swap_moves=True).solve(
        QUBO(np.array([[-1.0]])), seed=0
    )
    assert one.x.tolist() == [1]


def test_sa_beta_range_and_errors(rng):
    q = random_qubo(6, rng)
    res = SimulatedAnnealingSolver(beta_range=(0.1, 5.0), num_reads=4, num_sweeps=50).solve(q, 0)
    assert res.metadata["beta_range"] == (0.1, 5.0)
    assert res.metadata["read_energies"].shape == (4,)
    with pytest.raises(ValueError):
        SimulatedAnnealingSolver(backend="x").solve(q)
    assert neal_beta_range(np.zeros(3), np.zeros((3, 3))) == (0.1, 1.0)
    hot, cold = neal_beta_range(np.array([1.0, 0.0]), np.zeros((2, 2)))
    assert hot < cold


def test_registry(rng):
    assert set(available_solvers()) == {"sa", "sb", "brute_force", "ksubset", "qaoa"}
    s = get_solver(("sa", {"num_reads": 3}))
    assert s.num_reads == 3
    assert get_solver(s) is s
    assert get_solver("sa", num_reads=2) == SimulatedAnnealingSolver(num_reads=2)
    assert get_solver("sa") != get_solver("sb")
    assert "num_reads=50" in repr(get_solver("sa"))
    with pytest.raises(ValueError):
        get_solver("nope")
    with pytest.raises(ValueError):
        get_solver(s, num_reads=4)
    with pytest.raises(TypeError):
        get_solver(3)  # type: ignore[arg-type]


def test_dwave_backend(rng):
    pytest.importorskip("dwave.samplers")
    q = random_qubo(8, rng)
    res = SimulatedAnnealingSolver(backend="dwave", num_reads=20).solve(q, seed=1)
    assert res.energy == pytest.approx(BruteForceSolver().solve(q).energy, rel=0.05)


def test_torch_backend(rng):
    pytest.importorskip("simulated_bifurcation")
    q = random_qubo(8, rng)
    res = SimulatedBifurcationSolver(backend="torch", n_agents=64, n_steps=2000).solve(q, seed=1)
    assert res.energy == pytest.approx(BruteForceSolver().solve(q).energy, rel=0.05)


def test_ksubset_matches_brute_force_with_exact_penalty(rng):
    from qubosel.formulations import add_cardinality_penalty
    from qubosel.solvers import KSubsetSolver

    for k in (1, 2, 3):
        q = random_qubo(10, rng)
        ks = KSubsetSolver(k=k, chunk_size=7).solve(q)
        bf = BruteForceSolver().solve(add_cardinality_penalty(q, k, lam="auto"))
        assert ks.n_selected == k and ks.energy == pytest.approx(q.energy(ks.x))
        assert ks.energy == pytest.approx(q.energy(bf.x))
    with pytest.raises(ValueError):
        KSubsetSolver(k=11).solve(q)
    assert get_solver("ksubset", k=2).k == 2
