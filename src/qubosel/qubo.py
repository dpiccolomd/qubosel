"""QUBO and Ising containers with a single, fixed sign/scale convention.

Conventions (never change these):

* A :class:`QUBO` stores a **symmetric** matrix ``Q`` and an ``offset``.
  Its energy for a binary vector ``x`` is ``x^T Q x + offset`` (off-diagonal
  terms are therefore counted twice).
* The *upper* (dimod-style) representation is ``Q_upper[i, j] = 2 Q[i, j]``
  for ``i < j`` and ``Q_upper[i, i] = Q[i, i]``.
* The Ising mapping uses ``x = (1 - s) / 2``: spin ``-1`` means *selected*.
  ``h_i = -(Q_ii + sum_{j != i} Q_ij) / 2``, ``J_ij = Q_ij / 2`` for ``i < j``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_binary(x: ArrayLike, n: int) -> NDArray[np.int8]:
    arr = np.asarray(x)
    if arr.ndim != 1 or arr.shape[0] != n:
        raise ValueError(f"expected a binary vector of length {n}, got shape {arr.shape}")
    if not np.all((arr == 0) | (arr == 1)):
        raise ValueError("x must contain only 0/1 values")
    return arr.astype(np.int8)


@dataclass
class Ising:
    """Ising model ``E(s) = sum_i h_i s_i + sum_{i<j} J_ij s_i s_j + offset``.

    ``J`` is stored as a symmetric matrix with zero diagonal; the energy uses
    each unordered pair once.
    """

    h: NDArray[np.float64]
    J: NDArray[np.float64]
    offset: float = 0.0

    def __post_init__(self) -> None:
        self.h = np.asarray(self.h, dtype=np.float64)
        self.J = np.asarray(self.J, dtype=np.float64)
        n = self.h.shape[0]
        if self.J.shape != (n, n):
            raise ValueError("J must be square and match h")
        if not np.allclose(self.J, self.J.T):
            raise ValueError("J must be symmetric")
        self.J = 0.5 * (self.J + self.J.T)
        np.fill_diagonal(self.J, 0.0)
        self.offset = float(self.offset)

    @property
    def n(self) -> int:
        return int(self.h.shape[0])

    def energy(self, s: ArrayLike) -> float:
        s_arr = np.asarray(s, dtype=np.float64)
        if s_arr.shape != (self.n,):
            raise ValueError(f"expected spin vector of length {self.n}")
        if not np.all(np.abs(s_arr) == 1):
            raise ValueError("s must contain only +/-1 values")
        return float(self.h @ s_arr + 0.5 * s_arr @ self.J @ s_arr + self.offset)

    def local_fields(self, s: ArrayLike) -> NDArray[np.float64]:
        """``h_i + sum_j J_ij s_j`` for every spin; flip cost is ``-2 s_i * field_i``."""
        s_arr = np.asarray(s, dtype=np.float64)
        return self.h + self.J @ s_arr


@dataclass
class QUBO:
    """Symmetric QUBO ``E(x) = x^T Q x + offset``."""

    Q: NDArray[np.float64]
    offset: float = 0.0
    names: list[str] | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        Q = np.asarray(self.Q, dtype=np.float64)
        if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
            raise ValueError("Q must be a square matrix")
        if not np.allclose(Q, Q.T, atol=1e-12, rtol=0):
            raise ValueError("Q must be symmetric; use QUBO.from_upper for dimod-style input")
        self.Q = 0.5 * (Q + Q.T)
        self.offset = float(self.offset)
        if self.names is not None:
            self.names = [str(v) for v in self.names]
            if len(self.names) != self.n:
                raise ValueError("names length must match Q")

    # ------------------------------------------------------------------ basic
    @property
    def n(self) -> int:
        return int(self.Q.shape[0])

    def copy(self) -> QUBO:
        return QUBO(
            self.Q.copy(),
            self.offset,
            list(self.names) if self.names else None,
            dict(self.metadata),
        )

    @classmethod
    def from_upper(
        cls, Q_upper: ArrayLike, offset: float = 0.0, names: list[str] | None = None
    ) -> QUBO:
        """Build from a dimod-style upper-triangular matrix (off-diagonals counted once)."""
        U = np.asarray(Q_upper, dtype=np.float64)
        if U.ndim != 2 or U.shape[0] != U.shape[1]:
            raise ValueError("Q_upper must be a square matrix")
        if np.any(np.tril(U, -1) != 0):
            # accept full matrices by folding the lower triangle into the upper one
            U = np.triu(U) + np.tril(U, -1).T
        Q = 0.5 * (U + U.T)
        np.fill_diagonal(Q, np.diag(U))
        return cls(Q, offset, names)

    def to_upper(self) -> NDArray[np.float64]:
        """dimod-style representation: ``2 Q_ij`` above the diagonal, ``Q_ii`` on it."""
        U = np.triu(2.0 * self.Q, 1)
        U[np.diag_indices(self.n)] = np.diag(self.Q)
        return U

    def to_dict(self) -> dict[tuple[int, int], float]:
        """``{(i, j): coefficient}`` in the upper representation, zeros dropped."""
        U = self.to_upper()
        out: dict[tuple[int, int], float] = {}
        for i in range(self.n):
            for j in range(i, self.n):
                if U[i, j] != 0.0:
                    out[(i, j)] = float(U[i, j])
        return out

    # ----------------------------------------------------------------- energy
    def energy(self, x: ArrayLike) -> float:
        xb = _as_binary(x, self.n).astype(np.float64)
        return float(xb @ self.Q @ xb + self.offset)

    def energies(self, X: ArrayLike) -> NDArray[np.float64]:
        """Energies of many binary vectors (rows of ``X``)."""
        Xa = np.asarray(X, dtype=np.float64)
        if Xa.ndim == 1:
            Xa = Xa[None, :]
        if Xa.shape[1] != self.n:
            raise ValueError("X must have n columns")
        return np.einsum("ij,jk,ik->i", Xa, self.Q, Xa) + self.offset

    def marginal_gain(self, x: ArrayLike, i: int) -> float:
        """Energy change of setting ``x_i = 1`` relative to ``x_i = 0``.

        Equals ``Q_ii + 2 * sum_{j != i} Q_ij x_j``. For a selected feature the
        energy change of *removing* it is ``-marginal_gain(x, i)``.
        """
        xb = _as_binary(x, self.n).astype(np.float64)
        others = self.Q[i] @ xb - self.Q[i, i] * xb[i]
        return float(self.Q[i, i] + 2.0 * others)

    # ------------------------------------------------------------------ ising
    def to_ising(self) -> Ising:
        """Map to Ising spins with ``x = (1 - s)/2`` (spin -1 == selected)."""
        Q = self.Q
        diag = np.diag(Q)
        row_off = Q.sum(axis=1) - diag
        h = -(diag + row_off) / 2.0
        J = Q / 2.0
        J = J.copy()
        np.fill_diagonal(J, 0.0)
        offset = self.offset + diag.sum() / 2.0 + np.triu(Q, 1).sum() / 2.0
        return Ising(h, J, offset)

    # ------------------------------------------------------------- utilities
    def max_abs(self) -> float:
        """Largest absolute coefficient in the upper representation."""
        return float(np.max(np.abs(self.to_upper()))) if self.n else 0.0

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"QUBO(n={self.n}, offset={self.offset:.4g})"
