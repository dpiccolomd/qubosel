"""qubosel: combinatorial (QUBO-based) feature selection with quantum-inspired solvers."""

from qubosel.consensus import ConsensusResult, RunRecord, run_grid
from qubosel.formulations import (
    add_cardinality_penalty,
    alpha_search,
    build_cmi_qubo,
    build_mucke_qubo,
    build_qubo,
    normalize,
)
from qubosel.qubo import QUBO, Ising
from qubosel.selector import QUBOFeatureSelector
from qubosel.simulation import recovery_curve, standard_objectives
from qubosel.solvers import get_solver

__version__ = "0.1.0"

__all__ = [
    "QUBO",
    "ConsensusResult",
    "Ising",
    "QUBOFeatureSelector",
    "RunRecord",
    "__version__",
    "add_cardinality_penalty",
    "alpha_search",
    "build_cmi_qubo",
    "build_mucke_qubo",
    "build_qubo",
    "get_solver",
    "normalize",
    "recovery_curve",
    "run_grid",
    "standard_objectives",
]
