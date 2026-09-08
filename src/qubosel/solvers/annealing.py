"""Simulated annealing (Metropolis, geometric beta schedule) in numpy.

The optional ``backend="dwave"`` delegates to ``dwave.samplers.SimulatedAnnealingSampler``.
"""

from __future__ import annotations

import math

import numpy as np

from qubosel._utils import check_random_state
from qubosel.qubo import QUBO
from qubosel.solvers.base import BaseSolver, SolverResult


def neal_beta_range(h: np.ndarray, J: np.ndarray) -> tuple[float, float]:
    """Default beta range (rule of ``dwave-neal``).

    ``beta_hot = ln 2 / (2 max_i sigma_i)`` with ``sigma_i = |h_i| + sum_j |J_ij|``
    (the largest single-flip energy change is accepted with probability 1/2), and
    ``beta_cold = ln 100 / (2 min |bias|)`` over the non-zero linear and quadratic
    biases (the smallest possible energy change is accepted with probability 1/100).
    """
    sigma = np.abs(h) + np.abs(J).sum(axis=1)
    smax = float(sigma.max()) if sigma.size else 0.0
    if smax <= 0.0:
        return 0.1, 1.0
    biases = np.concatenate([np.abs(h).ravel(), np.abs(np.triu(J, 1)).ravel()])
    nonzero = biases[biases > 0]
    bmin = float(nonzero.min()) if nonzero.size else smax
    return math.log(2.0) / (2.0 * smax), math.log(100.0) / (2.0 * bmin)


def swap_cold_beta(
    h: np.ndarray, J: np.ndarray, rng: np.random.Generator, n_states: int = 200, q: float = 0.05
) -> float:
    """Cold beta for swap moves: ``ln 100`` over a low quantile of sampled swap energy changes.

    On penalty-constrained QUBOs the penalty cancels between equal-cardinality
    states, so the energy scale relevant to swaps is much smaller than any bias.
    """
    n = h.shape[0]
    S = rng.choice(np.array([-1.0, 1.0]), size=(n_states, n))
    F = h[None, :] + S @ J
    rows = np.arange(n_states)
    vals = []
    for _ in range(5):
        i = rng.integers(0, n, size=n_states)
        j = rng.integers(0, n, size=n_states)
        si, sj = S[rows, i], S[rows, j]
        ok = (i != j) & (si != sj)
        dE = -2.0 * si * F[rows, i] - 2.0 * sj * F[rows, j] + 4.0 * J[i, j] * si * sj
        vals.append(np.abs(dE[ok]))
    d = np.concatenate(vals)
    d = d[d > 1e-15]
    if d.size == 0:
        return 1.0
    return float(math.log(100.0) / np.quantile(d, q))


class SimulatedAnnealingSolver(BaseSolver):
    """Simulated annealing on the Ising form of the QUBO.

    Parameters
    ----------
    num_reads : int
        Independent annealing runs (vectorised). The lowest-energy read is returned.
    num_sweeps : int
        Sweeps per read; each sweep visits every spin in a fresh random order.
    beta_range : "auto" or (beta_hot, beta_cold)
    swap_moves : bool
        After every flip sweep also propose ``n`` cardinality-preserving swaps
        (one selected and one unselected variable exchanged). Recommended for
        penalty-constrained QUBOs, where single flips freeze the cardinality
        long before the objective terms become thermally resolvable.
    backend : {"numpy", "dwave"}
    """

    name = "sa"

    def __init__(
        self,
        num_reads: int = 50,
        num_sweeps: int = 1000,
        beta_range: str | tuple[float, float] = "auto",
        swap_moves: bool = False,
        backend: str = "numpy",
    ):
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self.beta_range = beta_range
        self.swap_moves = swap_moves
        self.backend = backend

    # ------------------------------------------------------------------ numpy
    def _solve_numpy(self, qubo: QUBO, seed: int | None) -> SolverResult:
        rng = check_random_state(seed)
        ising = qubo.to_ising()
        h, J = ising.h, ising.J
        n = qubo.n
        R = int(self.num_reads)
        if self.beta_range == "auto":
            b_hot, b_cold = neal_beta_range(h, J)
            if self.swap_moves and n >= 2:
                b_cold = max(b_cold, swap_cold_beta(h, J, rng))
        else:
            b_hot, b_cold = (float(v) for v in self.beta_range)
        betas = np.geomspace(b_hot, b_cold, max(int(self.num_sweeps), 1))

        S = rng.choice(np.array([-1.0, 1.0]), size=(R, n))
        F = h[None, :] + S @ J  # local fields
        for beta in betas:
            order = rng.permutation(n)
            U = rng.random((R, n))
            for idx, i in enumerate(order):
                dE = -2.0 * S[:, i] * F[:, i]
                accept = (dE <= 0.0) | (U[:, idx] < np.exp(-beta * np.maximum(dE, 0.0)))
                if not accept.any():
                    continue
                delta = -2.0 * S[accept, i]
                S[accept, i] *= -1.0
                F[accept] += delta[:, None] * J[i][None, :]
            if self.swap_moves and n >= 2:
                self._swap_sweep(S, F, J, beta, rng)
        X = ((1.0 - S) / 2.0).astype(np.int8)
        E = qubo.energies(X)
        j = int(np.argmin(E))
        return SolverResult(
            X[j], float(E[j]), int(X[j].sum()), {"beta_range": (b_hot, b_cold), "read_energies": E}
        )

    @staticmethod
    def _swap_sweep(
        S: np.ndarray, F: np.ndarray, J: np.ndarray, beta: float, rng: np.random.Generator
    ) -> None:
        """``n`` Metropolis swap proposals per read (flip one -1 and one +1 spin together)."""
        R, n = S.shape
        rows = np.arange(R)
        for _ in range(n):
            i = rng.integers(0, n, size=R)
            j = rng.integers(0, n, size=R)
            valid = (i != j) & (S[rows, i] != S[rows, j])
            if not valid.any():
                continue
            si, sj = S[rows, i], S[rows, j]
            dE = -2.0 * si * F[rows, i] - 2.0 * sj * F[rows, j] + 4.0 * J[i, j] * si * sj
            u = rng.random(R)
            accept = valid & ((dE <= 0.0) | (u < np.exp(-beta * np.maximum(dE, 0.0))))
            if not accept.any():
                continue
            idx = rows[accept]
            ia, ja = i[accept], j[accept]
            di, dj = -2.0 * S[idx, ia], -2.0 * S[idx, ja]
            S[idx, ia] *= -1.0
            S[idx, ja] *= -1.0
            F[idx] += di[:, None] * J[ia] + dj[:, None] * J[ja]

    # ------------------------------------------------------------------ dwave
    def _solve_dwave(self, qubo: QUBO, seed: int | None) -> SolverResult:
        try:
            from dwave.samplers import SimulatedAnnealingSampler
        except ImportError as exc:  # pragma: no cover
            raise ImportError("backend='dwave' requires the 'dwave-samplers' package") from exc
        kwargs = {"num_reads": int(self.num_reads), "num_sweeps": int(self.num_sweeps)}
        if seed is not None:
            kwargs["seed"] = int(seed) % (2**31 - 1)
        if self.beta_range != "auto":
            kwargs["beta_range"] = list(self.beta_range)
        sampler = SimulatedAnnealingSampler()
        ss = sampler.sample_qubo(qubo.to_dict(), **kwargs)
        best = ss.first
        x = np.array([best.sample.get(i, 0) for i in range(qubo.n)], dtype=np.int8)
        return SolverResult.from_x(qubo, x, backend="dwave")

    def _solve(self, qubo: QUBO, seed: int | None) -> SolverResult:
        if self.backend == "numpy":
            return self._solve_numpy(qubo, seed)
        if self.backend == "dwave":
            return self._solve_dwave(qubo, seed)
        raise ValueError(f"unknown backend {self.backend!r}")
