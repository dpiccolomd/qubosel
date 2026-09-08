"""QAOA on the Ising form of a QUBO, implemented with PennyLane.

``pennylane`` is imported lazily so the core package has no quantum dependency.
Install it with ``pip install qubosel[qaoa]``.

Conventions
-----------
* ``H_C = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j`` where ``(h, J)`` come from
  :meth:`qubosel.qubo.QUBO.to_ising` (so ``|1>`` on wire ``i`` means feature
  ``i`` is selected and the measured bitstring *is* the selection).
* One layer applies ``RZ(2 gamma h_i)``, ``IsingZZ(2 gamma J_ij)`` and ``RX(2 beta)``.
* Parameters are optimised on an analytic (``shots=None``) QNode; the final
  state is then sampled with ``shots`` shots on a seeded device and the sampled
  bitstring with the lowest QUBO energy is returned (``readout="best_energy"``),
  or the most frequent one (``readout="most_probable"``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from qubosel._utils import check_random_state
from qubosel.qubo import QUBO
from qubosel.solvers.base import BaseSolver, SolverResult

DeviceFactory = Callable[[int, int | None, int | None], Any]


def _import_pennylane():
    try:
        import pennylane as qp
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError("QAOASolver requires pennylane; install qubosel[qaoa]") from exc
    return qp


def cost_hamiltonian(qubo: QUBO):
    """PennyLane observable for the Ising form of ``qubo`` (offset excluded)."""
    qp = _import_pennylane()
    ising = qubo.to_ising()
    n = qubo.n
    coeffs: list[float] = []
    ops: list[Any] = []
    for i in range(n):
        if ising.h[i] != 0.0:
            coeffs.append(float(ising.h[i]))
            ops.append(qp.Z(i))
    for i in range(n):
        for j in range(i + 1, n):
            if ising.J[i, j] != 0.0:
                coeffs.append(float(ising.J[i, j]))
                ops.append(qp.Z(i) @ qp.Z(j))
    if not coeffs:
        coeffs, ops = [0.0], [qp.Identity(0)]
    return qp.dot(coeffs, ops), ising


def _layer(qp, ising, gamma: float, beta: float) -> None:
    n = ising.n
    for i in range(n):
        if ising.h[i] != 0.0:
            qp.RZ(2.0 * gamma * ising.h[i], wires=i)
    for i in range(n):
        for j in range(i + 1, n):
            if ising.J[i, j] != 0.0:
                qp.IsingZZ(2.0 * gamma * ising.J[i, j], wires=[i, j])
    for i in range(n):
        qp.RX(2.0 * beta, wires=i)


class QAOASolver(BaseSolver):
    """QAOA solver (PennyLane).

    Parameters
    ----------
    p : int
        Number of layers.
    optimizer : str
        scipy.optimize method used on the analytic expectation (default ``"cobyla"``).
    max_iter : int
    shots : int
        Shots for the final sampling.
    device : str or callable
        Device name for ``qp.device`` or ``callable(wires, shots, seed) -> device``.
    analytic_optimization : bool
        Optimise on the exact expectation (``shots=None``); if False the
        optimisation QNode uses ``shots`` shots (noisier, slower).
    readout : {"best_energy", "most_probable"}
    n_restarts : int
        Best-of random restarts of the classical optimiser.
    """

    name = "qaoa"

    def __init__(
        self,
        p: int = 2,
        optimizer: str = "cobyla",
        max_iter: int = 100,
        shots: int = 2000,
        device: str | DeviceFactory = "default.qubit",
        analytic_optimization: bool = True,
        readout: str = "best_energy",
        n_restarts: int = 1,
    ):
        self.p = p
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.shots = shots
        self.device = device
        self.analytic_optimization = analytic_optimization
        self.readout = readout
        self.n_restarts = n_restarts

    def _make_device(self, qp, wires: int, shots: int | None, seed: int | None):
        if callable(self.device):
            return self.device(wires, shots, seed)
        kwargs: dict[str, Any] = {"wires": wires, "shots": shots}
        if seed is not None:
            kwargs["seed"] = int(seed)
        return qp.device(self.device, **kwargs)

    def _solve(self, qubo: QUBO, seed: int | None) -> SolverResult:
        from scipy.optimize import minimize

        qp = _import_pennylane()
        rng = check_random_state(seed)
        n = qubo.n
        H, ising = cost_hamiltonian(qubo)
        p = int(self.p)

        opt_shots = None if self.analytic_optimization else int(self.shots)
        dev_opt = self._make_device(qp, n, opt_shots, int(rng.integers(0, 2**31 - 1)))

        @qp.qnode(dev_opt)
        def expectation(params):
            for i in range(n):
                qp.Hadamard(wires=i)
            for layer in range(p):
                _layer(qp, ising, params[layer], params[p + layer])
            return qp.expval(H)

        def objective(params):
            return float(expectation(params))

        best_val, best_params = np.inf, None
        n_evals = 0
        for _ in range(max(int(self.n_restarts), 1)):
            x0 = np.concatenate(
                [rng.uniform(0.0, np.pi, size=p), rng.uniform(0.0, np.pi / 2.0, size=p)]
            )
            res = minimize(
                objective,
                x0,
                method=self.optimizer.upper()
                if self.optimizer.lower() == "cobyla"
                else self.optimizer,
                options={"maxiter": int(self.max_iter)},
            )
            n_evals += int(getattr(res, "nfev", 0))
            if res.fun < best_val:
                best_val, best_params = float(res.fun), np.asarray(res.x)
        assert best_params is not None

        dev_sample = self._make_device(qp, n, int(self.shots), int(rng.integers(0, 2**31 - 1)))

        @qp.qnode(dev_sample)
        def sample(params):
            for i in range(n):
                qp.Hadamard(wires=i)
            for layer in range(p):
                _layer(qp, ising, params[layer], params[p + layer])
            return qp.sample(wires=list(range(n)))

        samples = np.asarray(sample(best_params)).reshape(-1, n).astype(np.int8)
        energies = qubo.energies(samples)
        uniq, counts = np.unique(samples, axis=0, return_counts=True)
        most_probable = uniq[int(np.argmax(counts))]
        if self.readout == "best_energy":
            x = samples[int(np.argmin(energies))]
        elif self.readout == "most_probable":
            x = most_probable
        else:
            raise ValueError(f"unknown readout {self.readout!r}")
        meta: dict[str, Any] = {
            "expectation": best_val + ising.offset,
            "params": best_params,
            "most_probable": most_probable.astype(np.int8),
            "n_function_evals": n_evals,
            "pennylane_version": getattr(qp, "__version__", "unknown"),
            "sample_energy_min": float(energies.min()),
            "sample_energy_mean": float(energies.mean()),
        }
        if n <= 16:
            from qubosel.solvers.brute_force import BruteForceSolver

            all_e = BruteForceSolver().all_energies(qubo)
            emin, emax = float(all_e.min()), float(all_e.max())
            e_x = qubo.energy(x)
            meta["approximation_ratio"] = (emax - e_x) / (emax - emin) if emax > emin else 1.0
            meta["optimal_energy"] = emin
        return SolverResult(x.astype(np.int8), qubo.energy(x), int(x.sum()), meta)


def basis_state_energy(qubo: QUBO, x) -> float:
    """``<x| H_C |x> + offset`` for a computational-basis state (testing helper)."""
    qp = _import_pennylane()
    n = qubo.n
    H, ising = cost_hamiltonian(qubo)
    dev = qp.device("default.qubit", wires=n)

    @qp.qnode(dev)
    def circuit():
        qp.BasisState(np.asarray(x, dtype=int), wires=list(range(n)))
        return qp.expval(H)

    return float(circuit()) + ising.offset
