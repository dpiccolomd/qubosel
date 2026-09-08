"""Run a grid of (solver, k, seed) jobs and aggregate the selections by vote."""

from __future__ import annotations

import traceback
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray

from qubosel._utils import child_seeds
from qubosel.qubo import QUBO
from qubosel.solvers.base import BaseSolver


@dataclass
class RunRecord:
    """One solver call in the grid."""

    solver: str
    k: int
    repeat: int
    seed: int
    status: str  # "ok" | "failed" | "oversize" | "undersize"
    x: NDArray[np.int8] | None = None
    energy: float | None = None
    n_selected: int | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@dataclass
class ConsensusResult:
    selection_matrix: NDArray[np.int8]  # (n_ok, n_features)
    votes: NDArray[np.int64]  # (n_features,)
    support: NDArray[np.bool_]
    n_attempted: int
    n_ok: int
    min_votes: int
    records: list[RunRecord]
    names: list[str] | None = None

    def frequency(self) -> NDArray[np.float64]:
        """Votes divided by the number of attempted runs."""
        return self.votes / max(self.n_attempted, 1)

    def to_frame(self):
        import pandas as pd

        names = self.names or [f"x{i}" for i in range(self.votes.shape[0])]
        return pd.DataFrame(
            {
                "feature": names,
                "votes": self.votes,
                "frequency": self.frequency(),
                "support": self.support,
            }
        )

    def selection_table(self):
        """Wide table (one row per successful run) mirroring the study's supplementary matrix."""
        import pandas as pd

        names = self.names or [f"x{i}" for i in range(self.votes.shape[0])]
        rows = [r for r in self.records if r.ok]
        idx = [f"{r.solver}_{r.k}_r{r.repeat}" for r in rows]
        return pd.DataFrame(self.selection_matrix, index=idx, columns=names)


def trim_to_k(qubo: QUBO, x: NDArray, k: int) -> NDArray[np.int8]:
    """Iteratively remove the selected feature whose removal lowers ``qubo`` energy most.

    Deterministic: ties are broken by the lowest index.
    """
    xb = np.asarray(x).astype(np.int8).copy()
    while int(xb.sum()) > k:
        sel = np.flatnonzero(xb)
        deltas = np.array([-qubo.marginal_gain(xb, i) for i in sel])
        xb[sel[int(np.argmin(deltas))]] = 0
    return xb


def resolve_min_votes(min_votes: int | float, n_attempted: int) -> int:
    if isinstance(min_votes, float) and not float(min_votes).is_integer():
        if not 0.0 < min_votes <= 1.0:
            raise ValueError("fractional min_votes must be in (0, 1]")
        return max(1, int(np.ceil(min_votes * n_attempted)))
    mv = int(min_votes)
    if mv < 1:
        raise ValueError("min_votes must be >= 1")
    return mv


def _run_one(solver: BaseSolver, qubo: QUBO, k: int, repeat: int, seed: int) -> RunRecord:
    try:
        res = solver.solve(qubo, seed=seed)
    except Exception as exc:
        return RunRecord(
            solver.name,
            k,
            repeat,
            seed,
            "failed",
            error=f"{type(exc).__name__}: {exc}",
            metadata={"traceback": traceback.format_exc()},
        )
    return RunRecord(
        solver.name, k, repeat, seed, "ok", res.x, res.energy, res.n_selected, metadata=res.metadata
    )


def run_grid(
    qubo_for_k: Callable[[int], QUBO],
    solvers: Sequence[BaseSolver],
    ks: Sequence[int],
    n_repeats: int = 1,
    random_state: int | np.random.Generator | None = None,
    min_votes: int | float = 1,
    oversize_policy: str = "trim",
    undersize_policy: str = "keep",
    base_qubo: QUBO | None = None,
    n_jobs: int | None = None,
    names: list[str] | None = None,
    n_features: int | None = None,
) -> ConsensusResult:
    """Solve ``qubo_for_k(k)`` with every solver, ``n_repeats`` times, and vote.

    Seeds are drawn once, in a fixed ``(solver, k, repeat)`` order, so results
    are identical for any ``n_jobs``. Failed runs are kept in ``records`` with
    ``status="failed"`` and still count in the vote denominator.

    ``oversize_policy``: ``"keep"`` (count as is), ``"trim"`` (remove features
    greedily on ``base_qubo`` until ``k`` remain) or ``"flag"`` (exclude from
    the vote, keep the record with ``status="oversize"``).
    ``undersize_policy``: ``"keep"`` or ``"flag"``.
    """
    if oversize_policy not in ("keep", "trim", "flag"):
        raise ValueError(f"unknown oversize_policy {oversize_policy!r}")
    if undersize_policy not in ("keep", "flag"):
        raise ValueError(f"unknown undersize_policy {undersize_policy!r}")
    solvers = list(solvers)
    ks = [int(k) for k in ks]
    jobs: list[tuple[BaseSolver, int, int]] = [
        (s, k, r) for s in solvers for k in ks for r in range(int(n_repeats))
    ]
    seeds = child_seeds(random_state, len(jobs))
    qubos = {k: qubo_for_k(k) for k in ks}
    d = n_features if n_features is not None else next(iter(qubos.values())).n
    if base_qubo is None:
        base_qubo = next(iter(qubos.values()))

    records: list[RunRecord] = Parallel(n_jobs=n_jobs)(
        delayed(_run_one)(s, qubos[k], k, r, seed)
        for (s, k, r), seed in zip(jobs, seeds, strict=True)
    )

    rows: list[NDArray[np.int8]] = []
    for rec in records:
        if rec.status == "failed":
            warnings.warn(
                f"run {rec.solver} k={rec.k} repeat={rec.repeat} failed: {rec.error}",
                RuntimeWarning,
            )
            continue
        assert rec.x is not None
        ns = int(rec.x.sum())
        if ns > rec.k:
            if oversize_policy == "trim":
                rec.x = trim_to_k(base_qubo, rec.x, rec.k)
                rec.metadata["trimmed_from"] = ns
                rec.n_selected = int(rec.x.sum())
            elif oversize_policy == "flag":
                rec.status = "oversize"
                continue
        elif ns < rec.k and undersize_policy == "flag":
            rec.status = "undersize"
            continue
        rows.append(rec.x.astype(np.int8))

    n_attempted = len(records)
    matrix = np.vstack(rows) if rows else np.zeros((0, d), dtype=np.int8)
    votes = matrix.sum(axis=0).astype(np.int64)
    mv = resolve_min_votes(min_votes, n_attempted)
    support = votes >= mv
    return ConsensusResult(matrix, votes, support, n_attempted, len(rows), mv, records, names)
