"""Exhaustive enumeration for small QUBOs."""

from __future__ import annotations

import numpy as np

from qubosel.qubo import QUBO
from qubosel.solvers.base import BaseSolver, SolverResult


def _bit_matrix(start: int, stop: int, n: int) -> np.ndarray:
    idx = np.arange(start, stop, dtype=np.int64)
    return ((idx[:, None] >> np.arange(n, dtype=np.int64)[None, :]) & 1).astype(np.float64)


class BruteForceSolver(BaseSolver):
    """Enumerate all ``2^n`` assignments (``n <= max_n``).

    Ties are broken deterministically by taking the smallest enumeration index
    (bit ``i`` of the index is ``x_i``), so the result never depends on ``seed``.
    """

    name = "brute_force"

    def __init__(self, max_n: int = 22, chunk_size: int = 1 << 16):
        self.max_n = max_n
        self.chunk_size = chunk_size

    def _solve(self, qubo: QUBO, seed: int | None) -> SolverResult:
        n = qubo.n
        if n > self.max_n:
            raise ValueError(f"BruteForceSolver supports n <= {self.max_n}, got {n}")
        total = 1 << n
        best_e = np.inf
        best_idx = -1
        best_x = None
        for start in range(0, total, self.chunk_size):
            stop = min(start + self.chunk_size, total)
            X = _bit_matrix(start, stop, n)
            E = np.einsum("ij,jk,ik->i", X, qubo.Q, X)
            j = int(np.argmin(E))  # first minimum -> smallest index in the chunk
            if E[j] < best_e:
                best_e = float(E[j])
                best_idx = start + j
                best_x = X[j]
        assert best_x is not None
        x = best_x.astype(np.int8)
        return SolverResult(
            x, best_e + qubo.offset, int(x.sum()), {"index": best_idx, "exact": True}
        )

    def all_energies(self, qubo: QUBO) -> np.ndarray:
        """Energies of every assignment, indexed by the bit-encoded integer."""
        n = qubo.n
        if n > self.max_n:
            raise ValueError(f"BruteForceSolver supports n <= {self.max_n}, got {n}")
        out = np.empty(1 << n)
        for start in range(0, 1 << n, self.chunk_size):
            stop = min(start + self.chunk_size, 1 << n)
            X = _bit_matrix(start, stop, n)
            out[start:stop] = np.einsum("ij,jk,ik->i", X, qubo.Q, X) + qubo.offset
        return out
