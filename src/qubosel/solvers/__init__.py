"""Solver registry."""

from __future__ import annotations

from typing import Any

from qubosel.solvers.annealing import SimulatedAnnealingSolver
from qubosel.solvers.base import BaseSolver, SolverResult
from qubosel.solvers.bifurcation import SimulatedBifurcationSolver
from qubosel.solvers.brute_force import BruteForceSolver
from qubosel.solvers.ksubset import KSubsetSolver

_REGISTRY: dict[str, type[BaseSolver]] = {
    "sa": SimulatedAnnealingSolver,
    "sb": SimulatedBifurcationSolver,
    "brute_force": BruteForceSolver,
    "ksubset": KSubsetSolver,
}


def _qaoa_class() -> type[BaseSolver]:
    from qubosel.solvers.qaoa import QAOASolver

    return QAOASolver


def available_solvers() -> list[str]:
    return [*_REGISTRY.keys(), "qaoa"]


def get_solver(spec: str | BaseSolver | tuple[str, dict[str, Any]], **kwargs: Any) -> BaseSolver:
    """Resolve ``spec`` to a solver instance.

    ``spec`` may be a name (``"sa"``, ``"sb"``, ``"qaoa"``, ``"brute_force"``),
    an instance (returned unchanged) or a ``(name, kwargs)`` tuple.
    """
    if isinstance(spec, BaseSolver):
        if kwargs:
            raise ValueError("kwargs cannot be combined with a solver instance")
        return spec
    if isinstance(spec, tuple):
        name, extra = spec
        kwargs = {**extra, **kwargs}
    else:
        name = spec
    if not isinstance(name, str):
        raise TypeError(f"unsupported solver spec {spec!r}")
    key = name.lower()
    if key == "qaoa":
        return _qaoa_class()(**kwargs)
    if key not in _REGISTRY:
        raise ValueError(f"unknown solver {name!r}; available: {available_solvers()}")
    return _REGISTRY[key](**kwargs)


__all__ = [
    "BaseSolver",
    "BruteForceSolver",
    "KSubsetSolver",
    "SimulatedAnnealingSolver",
    "SimulatedBifurcationSolver",
    "SolverResult",
    "available_solvers",
    "get_solver",
]
