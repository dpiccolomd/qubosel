"""Exact enumeration of all subsets of a fixed size ``k``."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from qubosel.qubo import QUBO
from qubosel.solvers.base import BaseSolver, SolverResult


class KSubsetSolver(BaseSolver):
    """Enumerate every ``k``-subset and return the one with the lowest energy.

    Exact for the cardinality-constrained problem without any penalty term:
    ``C(n, k)`` evaluations, vectorised in chunks. Feasible for ``k <= 3`` up to
    a few hundred features (``C(200, 3) = 1,313,400``). Ties are broken by the
    lexicographically smallest index tuple, so the result is deterministic.
    """

    name = "ksubset"

    def __init__(self, k: int = 3, chunk_size: int = 1 << 15):
        self.k = k
        self.chunk_size = chunk_size

    def _solve(self, qubo: QUBO, seed: int | None) -> SolverResult:
        n, k = qubo.n, int(self.k)
        if not 0 < k <= n:
            raise ValueError(f"k must be in [1, n]; got k={k}, n={n}")
        Q = qubo.Q
        diag = np.diag(Q)
        best_e, best_idx = np.inf, None
        it = combinations(range(n), k)
        while True:
            chunk = np.fromiter(
                (i for tup in _take(it, self.chunk_size) for i in tup), dtype=np.int64
            ).reshape(-1, k)
            if chunk.shape[0] == 0:
                break
            # energy = sum_i Q_ii + 2 * sum_{i<j} Q_ij over the subset
            e = diag[chunk].sum(axis=1)
            for a in range(k):
                for b in range(a + 1, k):
                    e += 2.0 * Q[chunk[:, a], chunk[:, b]]
            j = int(np.argmin(e))
            if e[j] < best_e:
                best_e, best_idx = float(e[j]), chunk[j].copy()
        x = np.zeros(n, dtype=np.int8)
        x[best_idx] = 1
        return SolverResult(x, best_e + qubo.offset, k, {"exact": True, "k": k})


def _take(iterator, m: int):
    out = []
    for _ in range(m):
        try:
            out.append(next(iterator))
        except StopIteration:
            break
    return out
