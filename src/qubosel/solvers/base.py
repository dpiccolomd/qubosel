"""Solver protocol and result container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qubosel.qubo import QUBO


@dataclass
class SolverResult:
    """Outcome of a single solver call."""

    x: NDArray[np.int8]
    energy: float
    n_selected: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_x(cls, qubo: QUBO, x: NDArray, **metadata: Any) -> SolverResult:
        xb = np.asarray(x).astype(np.int8)
        return cls(xb, qubo.energy(xb), int(xb.sum()), dict(metadata))

    @property
    def selected(self) -> NDArray[np.intp]:
        return np.flatnonzero(self.x)


class BaseSolver:
    """Base class: subclasses implement :meth:`_solve`."""

    name: str = "base"

    def solve(self, qubo: QUBO, seed: int | None = None) -> SolverResult:
        if qubo.n == 0:
            return SolverResult(np.zeros(0, dtype=np.int8), qubo.offset, 0, {"solver": self.name})
        result = self._solve(qubo, seed)
        result.metadata.setdefault("solver", self.name)
        result.metadata.setdefault("seed", seed)
        return result

    def _solve(self, qubo: QUBO, seed: int | None) -> SolverResult:  # pragma: no cover
        raise NotImplementedError

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Parameters as a dict (used for repr and sklearn cloning)."""
        import inspect

        sig = inspect.signature(type(self).__init__)
        return {k: getattr(self, k) for k in sig.parameters if k != "self" and hasattr(self, k)}

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({params})"

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and self.get_params() == other.get_params()  # type: ignore[union-attr]

    __hash__ = None  # type: ignore[assignment]
