"""Validation utilities: stability, permutation tests, classical benchmarks, downstream CV."""

from qubosel.validation.benchmarks import LassoTopK, classical_selectors, compare_selectors
from qubosel.validation.downstream import bootstrap_auc_ci, evaluate_downstream
from qubosel.validation.null_selection import NullSelectionResult, permuted_label_selection
from qubosel.validation.permutation import PermutationResult, permutation_test
from qubosel.validation.stability import StabilityResult, bootstrap_stability, nogueira_stability

__all__ = [
    "LassoTopK",
    "NullSelectionResult",
    "PermutationResult",
    "StabilityResult",
    "bootstrap_auc_ci",
    "bootstrap_stability",
    "classical_selectors",
    "compare_selectors",
    "evaluate_downstream",
    "nogueira_stability",
    "permutation_test",
    "permuted_label_selection",
]
