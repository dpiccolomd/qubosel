"""scikit-learn compatible QUBO feature selector."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

from qubosel.consensus import ConsensusResult, run_grid
from qubosel.formulations import (
    EPS_MUCKE,
    add_cardinality_penalty,
    alpha_search,
    build_qubo,
    normalize,
)
from qubosel.qubo import QUBO
from qubosel.solvers import BaseSolver, get_solver
from qubosel.solvers.brute_force import BruteForceSolver

SolverSpec = str | BaseSolver | tuple[str, dict[str, Any]]


class QUBOFeatureSelector(SelectorMixin, BaseEstimator):
    """Select features by minimising an information-theoretic QUBO.

    Parameters
    ----------
    k : int, optional
        Target number of features. Ignored when ``ks`` is given. Default:
        ``round(sqrt(n_features))``.
    ks : sequence of int, optional
        Several targets; every ``(solver, k, repeat)`` run votes and features
        with at least ``min_votes`` votes are kept.
    solvers : str, solver, (name, kwargs) or list of those
        ``"auto"`` uses exhaustive search when ``n_features <= brute_force_max_n``
        and simulated annealing otherwise.
    n_repeats : int
        Repeated runs per (solver, k) with different seeds.
    formulation : {"mucke", "cmi", "cmi_cr", "rank"} or callable
        A callable ``formulation(X, y, names) -> QUBO`` builds a custom
        unpenalised objective (its metadata must contain ``"I"`` and ``"R"``
        when ``cardinality="alpha"``).
    cardinality : {"alpha", "penalty", None}
        ``"alpha"`` (Mücke) searches the trade-off ``alpha`` that yields ``k``
        features at the optimum; ``"penalty"`` adds ``lam (sum x - k)^2``;
        ``None`` solves the unconstrained QUBO at ``alpha``.
    alpha : float
        Trade-off used when ``cardinality`` is ``None`` (Mücke) or ``"penalty"``.
    alpha_oracle : {"auto", "brute_force", "solver"}
        Optimiser used inside the alpha bisection.
    lam : float, "auto" or "legacy"
        Penalty strength (``"penalty"`` only).
    normalize : bool
        Rescale the penalised QUBO to max |coefficient| = 1 (dimod ``normalize``).
    n_bins, binning, base, mi_correction, n_null, estimator, n_neighbors, ksg_repeats
        Passed to :func:`qubosel.formulations.build_qubo`. ``mi_correction="expected"``
        subtracts from every MI term its exact permutation-null expectation
        (reliable MI); ``"permutation"`` is the Monte-Carlo version;
        ``estimator="ksg"`` uses the kNN estimator; ``formulation="rank"`` needs no estimator.
    min_votes : int or float
        Absolute votes or fraction of attempted runs.
    oversize_policy : {"trim", "keep", "flag"}
    undersize_policy : {"keep", "flag"}
    random_state : int or None
    n_jobs : int or None
    """

    def __init__(
        self,
        k: int | None = None,
        ks: Sequence[int] | None = None,
        solvers: SolverSpec | Sequence[SolverSpec] = "auto",
        n_repeats: int = 1,
        formulation: str = "mucke",
        cardinality: str | None = "alpha",
        alpha: float = 0.5,
        alpha_oracle: str = "auto",
        alpha_max_iter: int = 40,
        lam: float | str = "auto",
        normalize: bool = False,
        n_bins: int | None = None,
        binning: str = "quantile",
        base: float | None = None,
        mi_correction: str | None = None,
        n_null: int = 100,
        estimator: str = "histogram",
        n_neighbors: int = 3,
        ksg_repeats: int = 3,
        eps: float = EPS_MUCKE,
        min_votes: int | float = 1,
        oversize_policy: str = "trim",
        undersize_policy: str = "keep",
        brute_force_max_n: int = 16,
        random_state: int | None = None,
        n_jobs: int | None = None,
    ):
        self.k = k
        self.ks = ks
        self.solvers = solvers
        self.n_repeats = n_repeats
        self.formulation = formulation
        self.cardinality = cardinality
        self.alpha = alpha
        self.alpha_oracle = alpha_oracle
        self.alpha_max_iter = alpha_max_iter
        self.lam = lam
        self.normalize = normalize
        self.n_bins = n_bins
        self.binning = binning
        self.base = base
        self.mi_correction = mi_correction
        self.n_null = n_null
        self.estimator = estimator
        self.n_neighbors = n_neighbors
        self.ksg_repeats = ksg_repeats
        self.eps = eps
        self.min_votes = min_votes
        self.oversize_policy = oversize_policy
        self.undersize_policy = undersize_policy
        self.brute_force_max_n = brute_force_max_n
        self.random_state = random_state
        self.n_jobs = n_jobs

    # ---------------------------------------------------------------- sklearn
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        tags.input_tags.allow_nan = False
        return tags

    def _get_support_mask(self):
        check_is_fitted(self, "support_")
        return self.support_

    # ---------------------------------------------------------------- helpers
    def _resolve_ks(self, d: int) -> list[int]:
        if self.ks is not None:
            ks = [int(k) for k in self.ks]
        elif self.k is not None:
            ks = [int(self.k)]
        else:
            ks = [max(1, min(d, round(math.sqrt(d))))]
        out = []
        for k in ks:
            if k < 1:
                raise ValueError("k must be >= 1")
            if k > d:
                warnings.warn(f"k={k} exceeds n_features={d}; clamping to {d}", UserWarning)
                k = d
            out.append(k)
        return out

    def _resolve_solvers(self, d: int) -> list[BaseSolver]:
        spec = self.solvers
        if isinstance(spec, (str, BaseSolver, tuple)):
            specs: list[Any] = [spec]
        else:
            specs = list(spec)
        out: list[BaseSolver] = []
        for s in specs:
            if isinstance(s, str) and s == "auto":
                s = "brute_force" if d <= self.brute_force_max_n else "sa"
            out.append(get_solver(s))
        if not out:
            raise ValueError("at least one solver is required")
        return out

    def _oracle(self, solvers: list[BaseSolver], d: int, rng: np.random.Generator):
        mode = self.alpha_oracle
        if mode == "auto":
            mode = "brute_force" if d <= 20 else "solver"
        if mode == "brute_force":
            bf = BruteForceSolver(max_n=max(20, d))
            return lambda q: bf.solve(q).x
        if mode == "solver":
            solver = next((s for s in solvers if s.name != "qaoa"), None)
            if solver is None:
                solver = get_solver("sa")
            seed = int(rng.integers(0, 2**31 - 1))

            def oracle(q: QUBO):
                best = None
                for r in range(3):
                    res = solver.solve(q, seed=seed + r)
                    if best is None or res.energy < best.energy:
                        best = res
                return best.x  # type: ignore[union-attr]

            return oracle
        raise ValueError(f"unknown alpha_oracle {self.alpha_oracle!r}")

    # -------------------------------------------------------------------- fit
    def fit(self, X, y):
        X, y = validate_data(self, X, y, dtype=np.float64, ensure_min_samples=2, y_numeric=False)
        check_classification_targets(y)
        n, d = X.shape
        if n < 2:
            raise ValueError("at least 2 samples are required")
        known = ("mucke", "cmi", "cmi_cr", "rank")
        if not callable(self.formulation) and self.formulation not in known:
            raise ValueError(f"unknown formulation {self.formulation!r}")
        if self.cardinality not in ("alpha", "penalty", None):
            raise ValueError(f"unknown cardinality {self.cardinality!r}")
        if self.cardinality == "alpha" and self.formulation == "cmi":
            raise ValueError("cardinality='alpha' requires a relevance/redundancy formulation")

        rng = np.random.default_rng(self.random_state)
        ks = self._resolve_ks(d)
        solvers = self._resolve_solvers(d)
        names = list(getattr(self, "feature_names_in_", [f"x{i}" for i in range(d)]))

        constant = np.all(X == X[0], axis=0)
        if constant.any():
            warnings.warn(
                f"{int(constant.sum())} constant feature(s) carry no information", UserWarning
            )

        if callable(self.formulation):
            base = self.formulation(X, y, names)
        else:
            base = build_qubo(
                X,
                y,
                formulation=self.formulation,
                alpha=self.alpha,
                n_bins=self.n_bins,
                binning=self.binning,
                base=self.base,
                mi_correction=self.mi_correction,
                eps=self.eps,
                names=names,
                n_null=self.n_null,
                random_state=int(rng.integers(0, 2**31 - 1)),
                estimator=self.estimator,
                n_neighbors=self.n_neighbors,
                ksg_repeats=self.ksg_repeats,
            )
        self.base_qubo_ = base
        self.qubos_: dict[int, QUBO] = {}
        self.alphas_: dict[int, float] = {}
        self.alpha_status_: dict[int, str] = {}

        if self.cardinality == "alpha":
            if "I" not in base.metadata or "R" not in base.metadata:
                raise ValueError(
                    "cardinality='alpha' needs a QUBO with 'I' and 'R' in its metadata "
                    "(relevance/redundancy formulation)"
                )
            imp, red = base.metadata["I"], base.metadata["R"]
            oracle = self._oracle(solvers, d, rng)
            for k in ks:
                res = alpha_search(
                    imp, red, k, oracle, eps=self.eps, max_iter=self.alpha_max_iter, names=names
                )
                self.qubos_[k] = res.qubo
                self.alphas_[k] = res.alpha
                self.alpha_status_[k] = res.status
                if res.status != "exact":
                    warnings.warn(
                        f"alpha search for k={k} ended inexact "
                        f"(closest optimum has {res.n_selected} features)",
                        UserWarning,
                    )
        elif self.cardinality == "penalty":
            for k in ks:
                q = add_cardinality_penalty(base, k, self.lam)
                self.qubos_[k] = normalize(q) if self.normalize else q
        else:
            for k in ks:
                self.qubos_[k] = base

        grid_seed = int(rng.integers(0, 2**31 - 1))
        cons = run_grid(
            lambda k: self.qubos_[k],
            solvers,
            ks,
            n_repeats=self.n_repeats,
            random_state=grid_seed,
            min_votes=self.min_votes,
            oversize_policy=self.oversize_policy,
            undersize_policy=self.undersize_policy,
            base_qubo=base,
            n_jobs=self.n_jobs,
            names=names,
            n_features=d,
        )
        self.consensus_: ConsensusResult = cons
        self.records_ = cons.records
        self.votes_ = cons.votes
        support = cons.support.copy()
        if not support.any():
            warnings.warn(
                "no feature reached min_votes; falling back to the top-voted features", UserWarning
            )
            k0 = ks[0]
            order = np.lexsort((np.arange(d), -cons.votes))
            support[order[:k0]] = True
        self.support_ = support
        self.solvers_ = solvers
        self.ks_ = ks
        return self
