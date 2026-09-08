"""Simulated bifurcation (ballistic / discrete, Goto et al. 2021) in numpy.

The QUBO is mapped to its Ising form ``E(s) = h.s + 1/2 s^T J s`` and then to
Goto's convention ``E = -1/2 s^T J_G s - h_G . s`` with ``J_G = -J`` and
``h_G = -h``. The linear term is handled either through an ancilla spin
(``linear_mode="ancilla"``, N+1 oscillators, readout relative to the ancilla)
or as an external field (``linear_mode="field"``).
"""

from __future__ import annotations

import numpy as np

from qubosel._utils import check_random_state
from qubosel.qubo import QUBO
from qubosel.solvers.base import BaseSolver, SolverResult


class SimulatedBifurcationSolver(BaseSolver):
    """Simulated bifurcation solver.

    Parameters
    ----------
    variant : {"ballistic", "discrete"}
        bSB uses ``f(x) = x`` in the coupling term, dSB uses ``f(x) = sign(x)``.
    n_agents : int
        Independent oscillator systems integrated in parallel.
    n_steps : int
        Symplectic Euler steps of size ``dt``.
    pressure_slope : float
        Fraction of ``n_steps`` over which the pump ``a(t)`` ramps linearly 0 -> 1.
    xi0 : "auto" or float
        Coupling scale; ``"auto"`` = ``0.5 / (sqrt(N) * rms(J_G offdiag))``.
    linear_mode : {"ancilla", "field"}
    track_best : bool
        Evaluate the energy of ``sign(x)`` every ``eval_every`` steps and keep the best.
    backend : {"numpy", "torch"}
        ``"torch"`` delegates to the ``simulated-bifurcation`` package.
    """

    name = "sb"

    def __init__(
        self,
        variant: str = "ballistic",
        n_agents: int = 128,
        n_steps: int = 2000,
        dt: float = 0.1,
        pressure_slope: float = 0.5,
        xi0: str | float = "auto",
        linear_mode: str = "ancilla",
        track_best: bool = True,
        eval_every: int = 10,
        backend: str = "numpy",
    ):
        self.variant = variant
        self.n_agents = n_agents
        self.n_steps = n_steps
        self.dt = dt
        self.pressure_slope = pressure_slope
        self.xi0 = xi0
        self.linear_mode = linear_mode
        self.track_best = track_best
        self.eval_every = eval_every
        self.backend = backend

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _goto(qubo: QUBO) -> tuple[np.ndarray, np.ndarray, float]:
        ising = qubo.to_ising()
        return -ising.J, -ising.h, ising.offset

    @staticmethod
    def _auto_xi0(JG: np.ndarray) -> float:
        n = JG.shape[0]
        off = JG[~np.eye(n, dtype=bool)]
        rms = float(np.sqrt(np.mean(off**2))) if off.size else 0.0
        if rms <= 0.0:
            rms = 1.0
        return 0.5 / (np.sqrt(n) * rms)

    def _readout(self, X: np.ndarray, use_ancilla: bool) -> np.ndarray:
        S = np.where(X >= 0.0, 1.0, -1.0)
        if use_ancilla:
            S = S[:-1] * S[-1][None, :]
        return S

    # ------------------------------------------------------------------ numpy
    def _solve_numpy(self, qubo: QUBO, seed: int | None) -> SolverResult:
        if self.variant not in ("ballistic", "discrete"):
            raise ValueError(f"unknown variant {self.variant!r}")
        if self.linear_mode not in ("ancilla", "field"):
            raise ValueError(f"unknown linear_mode {self.linear_mode!r}")
        rng = check_random_state(seed)
        JG, hG, _ = self._goto(qubo)
        n = qubo.n
        xi0 = self._auto_xi0(JG) if self.xi0 == "auto" else float(self.xi0)
        use_ancilla = self.linear_mode == "ancilla"
        if use_ancilla:
            Jbig = np.zeros((n + 1, n + 1))
            Jbig[:n, :n] = JG
            Jbig[:n, n] = hG
            Jbig[n, :n] = hG
            J_dyn, h_dyn = Jbig, None
        else:
            J_dyn, h_dyn = JG, hG
        N = J_dyn.shape[0]
        A = int(self.n_agents)
        X = rng.uniform(-0.1, 0.1, size=(N, A))
        Y = rng.uniform(-0.1, 0.1, size=(N, A))
        dt = float(self.dt)
        steps = int(self.n_steps)
        ramp = max(steps * float(self.pressure_slope), 1.0)
        discrete = self.variant == "discrete"

        best_e = np.inf
        best_s: np.ndarray | None = None

        def evaluate(Xc: np.ndarray) -> tuple[float, np.ndarray]:
            S = self._readout(Xc, use_ancilla)
            Xb = ((1.0 - S) / 2.0).T
            E = qubo.energies(Xb)
            j = int(np.argmin(E))
            return float(E[j]), Xb[j]

        for step in range(steps):
            a = min(step / ramp, 1.0)
            Y += dt * (a - 1.0) * X
            X += dt * Y
            F = np.sign(X) if discrete else X
            drive = J_dyn @ F
            if h_dyn is not None:
                drive = drive + h_dyn[:, None]
            Y += dt * xi0 * drive
            out = np.abs(X) > 1.0
            if out.any():
                X[out] = np.sign(X[out])
                Y[out] = 0.0
            if self.track_best and (step % max(int(self.eval_every), 1) == 0):
                e, xb = evaluate(X)
                if e < best_e:
                    best_e, best_s = e, xb
        e, xb = evaluate(X)
        if e < best_e or best_s is None:
            best_e, best_s = e, xb
        x = best_s.astype(np.int8)
        return SolverResult(
            x,
            best_e,
            int(x.sum()),
            {"xi0": xi0, "variant": self.variant, "linear_mode": self.linear_mode},
        )

    # ------------------------------------------------------------------ torch
    def _solve_torch(self, qubo: QUBO, seed: int | None) -> SolverResult:
        try:
            import simulated_bifurcation as sb
            import torch
        except ImportError as exc:  # pragma: no cover
            raise ImportError("backend='torch' requires 'simulated-bifurcation'") from exc
        if seed is not None:
            torch.manual_seed(int(seed))
        JG, hG, _ = self._goto(qubo)
        # package convention: minimise -1/2 s^T J s - h^T s
        spins, _ = sb.minimize(
            torch.tensor(JG, dtype=torch.float32),
            torch.tensor(hG, dtype=torch.float32),
            input_type="spin",
            ballistic=(self.variant == "ballistic"),
            heated=False,
            agents=int(self.n_agents),
            max_steps=int(self.n_steps),
            best_only=True,
            verbose=False,
        )
        s = np.asarray(spins).ravel()[: qubo.n]
        x = ((1 - s) / 2).astype(np.int8)
        return SolverResult.from_x(qubo, x, backend="torch")

    def _solve(self, qubo: QUBO, seed: int | None) -> SolverResult:
        if self.backend == "numpy":
            return self._solve_numpy(qubo, seed)
        if self.backend == "torch":
            return self._solve_torch(qubo, seed)
        raise ValueError(f"unknown backend {self.backend!r}")
