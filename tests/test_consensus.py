import numpy as np
import pytest

from qubosel.consensus import resolve_min_votes, run_grid, trim_to_k
from qubosel.formulations import add_cardinality_penalty
from qubosel.qubo import QUBO
from qubosel.solvers import BruteForceSolver, SimulatedAnnealingSolver
from qubosel.solvers.base import BaseSolver, SolverResult

from .conftest import random_qubo


class FailingSolver(BaseSolver):
    name = "boom"

    def _solve(self, qubo, seed):
        raise RuntimeError("kaboom")


class FixedSolver(BaseSolver):
    name = "fixed"

    def __init__(self, x):
        self.x = x

    def _solve(self, qubo, seed):
        return SolverResult.from_x(qubo, np.array(self.x))


def test_failed_runs_count_in_denominator(rng):
    q = random_qubo(6, rng)
    with pytest.warns(RuntimeWarning, match="failed"):
        cons = run_grid(
            lambda k: add_cardinality_penalty(q, k),
            [BruteForceSolver(), FailingSolver()],
            ks=[2, 3],
            min_votes=2,
            random_state=0,
        )
    assert cons.n_attempted == 4 and cons.n_ok == 2
    assert sum(r.status == "failed" for r in cons.records) == 2
    assert cons.records[2].error.startswith("RuntimeError")
    assert cons.selection_matrix.shape == (2, 6)
    assert cons.votes.sum() == 5
    assert cons.min_votes == 2
    assert cons.frequency().max() <= 0.5
    df = cons.to_frame()
    assert list(df.columns) == ["feature", "votes", "frequency", "support"]
    assert cons.selection_table().shape == (2, 6)


def test_fractional_min_votes():
    assert resolve_min_votes(0.5, 12) == 6
    assert resolve_min_votes(2, 12) == 2
    assert resolve_min_votes(2.0, 12) == 2
    with pytest.raises(ValueError):
        resolve_min_votes(0, 12)
    with pytest.raises(ValueError):
        resolve_min_votes(1.5, 12)


def test_trim_exact_and_deterministic(rng):
    q = random_qubo(8, rng)
    x = np.ones(8, dtype=np.int8)
    t1, t2 = trim_to_k(q, x, 3), trim_to_k(q, x, 3)
    assert int(t1.sum()) == 3 and np.array_equal(t1, t2)
    assert np.array_equal(trim_to_k(q, x, 8), x)
    # greedy step removes the feature with the largest energy decrease
    sel = np.flatnonzero(x)
    first = sel[int(np.argmin([-q.marginal_gain(x, i) for i in sel]))]
    assert trim_to_k(q, x, 7)[first] == 0


def test_policies(rng):
    q = random_qubo(5, rng)
    over = FixedSolver([1, 1, 1, 1, 0])
    under = FixedSolver([1, 0, 0, 0, 0])
    keep = run_grid(lambda k: q, [over, under], ks=[2], oversize_policy="keep")
    assert keep.n_ok == 2 and keep.votes.tolist() == [2, 1, 1, 1, 0]
    trim = run_grid(lambda k: q, [over], ks=[2], oversize_policy="trim", base_qubo=q)
    assert trim.records[0].n_selected == 2 and trim.records[0].metadata["trimmed_from"] == 4
    flag = run_grid(
        lambda k: q, [over, under], ks=[2], oversize_policy="flag", undersize_policy="flag"
    )
    assert flag.n_ok == 0 and flag.selection_matrix.shape == (0, 5)
    assert {r.status for r in flag.records} == {"oversize", "undersize"}
    with pytest.raises(ValueError):
        run_grid(lambda k: q, [over], ks=[2], oversize_policy="x")
    with pytest.raises(ValueError):
        run_grid(lambda k: q, [over], ks=[2], undersize_policy="x")


def test_seed_order_independent_of_n_jobs(rng):
    q = random_qubo(10, rng)
    sa = SimulatedAnnealingSolver(num_reads=2, num_sweeps=20)
    kw = dict(ks=[2, 3], n_repeats=3, random_state=5)
    a = run_grid(lambda k: add_cardinality_penalty(q, k), [sa], n_jobs=1, **kw)
    b = run_grid(lambda k: add_cardinality_penalty(q, k), [sa], n_jobs=2, **kw)
    assert [r.seed for r in a.records] == [r.seed for r in b.records]
    assert np.array_equal(a.selection_matrix, b.selection_matrix)


def test_empty_qubo_names():
    q = QUBO(np.zeros((2, 2)), names=["a", "b"])
    cons = run_grid(lambda k: q, [BruteForceSolver()], ks=[1], names=["a", "b"])
    assert cons.to_frame()["feature"].tolist() == ["a", "b"]
