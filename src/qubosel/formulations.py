"""QUBO formulations for feature selection.

Two formulations are provided:

* **Mücke et al. (2023)** — ``Q(alpha) = R - alpha * (R + diag(I))`` where
  ``I_i = I(x_i; y)`` and ``R_ij = I(x_i; x_j)``. There is no cardinality
  constraint; the number of selected features is controlled through ``alpha``
  (:func:`alpha_search`).
* **CMI (Nguyen et al. 2014 / D-Wave example)** —
  diagonal ``-I(x_i; y)`` and pair terms ``-[I(x_j; y | x_i) + I(x_i; y | x_j)]``
  with an explicit quadratic cardinality penalty.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qubosel.information import (
    cmi_matrix,
    conditional_redundancy_matrix,
    discretize,
    ksg_mi_matrix,
    mi_matrix,
    rank_relevance,
    spearman_redundancy,
)
from qubosel.qubo import QUBO

EPS_MUCKE = 1e-8


# --------------------------------------------------------------------- builders
def build_mucke_qubo(
    I: ArrayLike,  # noqa: E741 - paper notation
    R: ArrayLike,
    alpha: float = 0.5,
    eps: float = EPS_MUCKE,
    names: list[str] | None = None,
) -> QUBO:
    """Mücke et al. 2023 QUBO ``Q = R - alpha (R + diag(I))``.

    Features whose scaled importance ``alpha * I_i`` is below ``eps`` get a
    diagonal ``mu = max(Q) > 0`` so that they are never selected.
    """
    imp = np.asarray(I, dtype=np.float64).ravel()
    red = np.asarray(R, dtype=np.float64)
    d = imp.shape[0]
    if red.shape != (d, d):
        raise ValueError("R must be (d, d)")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    red = 0.5 * (red + red.T)
    red = red.copy()
    np.fill_diagonal(red, 0.0)
    Q = (1.0 - alpha) * red - alpha * np.diag(imp)
    mu = float(Q.max()) if d else 0.0
    if mu <= 0.0:
        mu = 1.0
    zero = alpha * imp < eps
    if zero.any():
        Q[np.diag_indices(d)] = np.where(zero, mu, np.diag(Q))
    return QUBO(
        Q,
        0.0,
        names,
        {"formulation": "mucke", "alpha": alpha, "mu": mu, "n_zero_importance": int(zero.sum())},
    )


def build_cmi_qubo(I: ArrayLike, C: ArrayLike, names: list[str] | None = None) -> QUBO:  # noqa: E741
    """CMI QUBO: ``Q_ii = -I_i``, ``Q_upper[i, j] = -(C_ij + C_ji)``.

    In the symmetric representation the pair term is ``-(C_ij + C_ji) / 2``.
    """
    imp = np.asarray(I, dtype=np.float64).ravel()
    cm = np.asarray(C, dtype=np.float64)
    d = imp.shape[0]
    if cm.shape != (d, d):
        raise ValueError("C must be (d, d)")
    Q = -0.5 * (cm + cm.T)
    np.fill_diagonal(Q, -imp)
    return QUBO(Q, 0.0, names, {"formulation": "cmi"})


def build_qubo(
    X: ArrayLike,
    y: ArrayLike,
    formulation: str = "mucke",
    alpha: float = 0.5,
    n_bins: int | None = None,
    binning: str = "quantile",
    base: float | None = None,
    mi_correction: str | None = None,
    eps: float = EPS_MUCKE,
    names: list[str] | None = None,
    n_null: int = 100,
    random_state: int | None = 0,
    estimator: str = "histogram",
    n_neighbors: int = 3,
    ksg_repeats: int = 3,
) -> QUBO:
    """Build the (unpenalised) feature-selection QUBO from ``X`` and ``y``.

    ``formulation``: ``"mucke"`` (relevance/redundancy, Mücke 2023), ``"cmi"``
    (D-Wave example: pair term ``-[I(x_j;y|x_i) + I(x_i;y|x_j)]``), ``"cmi_cr"``
    (conditional-MI variant that rewards class-conditional redundancy: pair term
    ``-2 I(x_i;x_j|y)``, found in implementations derived from the D-Wave
    example) or ``"rank"``
    (Mücke structure with discretisation-free terms: relevance ``2|AUC - 0.5|``,
    redundancy ``Spearman rho^2``; binary outcome only).

    ``estimator``: ``"histogram"`` (plug-in MI on discretised codes, with
    optional ``mi_correction``) or ``"ksg"`` (kNN estimator of Kraskov et al.,
    no discretisation; ``"mucke"`` only).
    """
    yv = np.asarray(y).ravel()
    Xa = np.asarray(X, dtype=np.float64)
    if yv.shape[0] != Xa.shape[0]:
        raise ValueError("X and y have different lengths")
    _, y_codes = np.unique(yv, return_inverse=True)
    meta: dict = {
        "binning": None,
        "n_bins": None,
        "base": base,
        "mi_correction": None,
        "n_null": None,
        "estimator": estimator,
    }
    if formulation == "rank":
        imp = rank_relevance(Xa, yv)
        red = spearman_redundancy(Xa)
        q = build_mucke_qubo(imp, red, alpha=alpha, eps=eps, names=names)
        q.metadata.update(meta)
        q.metadata.update({"formulation": "rank", "I": imp, "R": red, "estimator": "rank"})
        return q
    if estimator == "ksg":
        if formulation != "mucke":
            raise ValueError("estimator='ksg' is only available with formulation='mucke'")
        imp, red = ksg_mi_matrix(
            Xa, yv, n_neighbors=n_neighbors, n_repeats=ksg_repeats, random_state=random_state
        )
        q = build_mucke_qubo(imp, red, alpha=alpha, eps=eps, names=names)
        q.metadata.update({"I": imp, "R": red, "n_neighbors": n_neighbors})
        q.metadata.update(meta)
        return q
    if estimator != "histogram":
        raise ValueError(f"unknown estimator {estimator!r}")
    codes = discretize(Xa, n_bins=n_bins, method=binning)
    mi_kw = {
        "base": base,
        "correction": mi_correction,
        "n_null": n_null,
        "random_state": random_state,
    }
    imp, red = mi_matrix(codes, y_codes, **mi_kw)
    if formulation == "mucke":
        q = build_mucke_qubo(imp, red, alpha=alpha, eps=eps, names=names)
        q.metadata.update({"I": imp, "R": red})
    elif formulation == "cmi":
        cm = cmi_matrix(codes, y_codes, **mi_kw)
        q = build_cmi_qubo(imp, cm, names=names)
        q.metadata.update({"I": imp, "C": cm})
    elif formulation == "cmi_cr":
        D = conditional_redundancy_matrix(codes, y_codes, **mi_kw)
        q = build_cmi_qubo(imp, D, names=names)  # symmetric D -> upper pair term -2 D_ij
        q.metadata.update({"formulation": "cmi_cr", "I": imp, "D": D})
    else:
        raise ValueError(f"unknown formulation {formulation!r}")
    meta.update(
        {"binning": binning, "n_bins": n_bins, "mi_correction": mi_correction, "n_null": n_null}
    )
    q.metadata.update(meta)
    return q


# -------------------------------------------------------------------- penalty
def auto_penalty(qubo: QUBO) -> float:
    """Penalty strength that makes ``k`` exact under an exact solver.

    ``1 + sum_i |Q_ii| + sum_{i<j} 2 |Q_ij|`` bounds the energy range of the
    unpenalised QUBO, so any cardinality violation costs more than any gain.
    """
    Q = qubo.Q
    return float(1.0 + np.abs(np.diag(Q)).sum() + 2.0 * np.abs(np.triu(Q, 1)).sum())


def resolve_penalty(qubo: QUBO, k: int, lam: float | str) -> float:
    if isinstance(lam, str):
        if lam == "legacy":
            return 10.0 * k
        if lam == "auto":
            return auto_penalty(qubo)
        raise ValueError(f"unknown lam {lam!r}")
    return float(lam)


def add_cardinality_penalty(qubo: QUBO, k: int, lam: float | str = "auto") -> QUBO:
    """Return ``qubo + lam * (sum_i x_i - k)^2``.

    ``lam`` may be a float, ``"auto"`` (see :func:`auto_penalty`) or
    ``"legacy"`` (``10 k``, a soft multiplicative rule used by some SDKs).
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    lam_v = resolve_penalty(qubo, k, lam)
    Q = qubo.Q.copy()
    n = qubo.n
    Q += lam_v * (np.ones((n, n)) - np.eye(n))
    Q[np.diag_indices(n)] += lam_v * (1.0 - 2.0 * k)
    out = QUBO(Q, qubo.offset + lam_v * k * k, qubo.names, dict(qubo.metadata))
    out.metadata.update({"k": k, "lam": lam_v, "penalty": lam if isinstance(lam, str) else "float"})
    return out


def normalize(qubo: QUBO) -> QUBO:
    """Scale so that the largest |coefficient| in the upper representation is 1.

    Mirrors ``dimod.BinaryQuadraticModel.normalize()`` with default ranges: the
    offset is scaled too, and scaling is applied even when the maximum is < 1.
    """
    scale = qubo.max_abs()
    if scale == 0.0:
        return qubo.copy()
    out = QUBO(qubo.Q / scale, qubo.offset / scale, qubo.names, dict(qubo.metadata))
    out.metadata["normalize_scale"] = scale
    return out


# --------------------------------------------------------------- alpha search
Oracle = Callable[[QUBO], NDArray[np.int8]]


@dataclass
class AlphaSearchResult:
    alpha: float
    x: NDArray[np.int8]
    qubo: QUBO
    status: str  # "exact" | "inexact"
    n_selected: int
    trace: list[tuple[float, int]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def alpha_search(
    I: ArrayLike,  # noqa: E741
    R: ArrayLike,
    k: int,
    oracle: Oracle,
    eps: float = EPS_MUCKE,
    max_iter: int = 40,
    tol: float = 1e-10,
    names: list[str] | None = None,
) -> AlphaSearchResult:
    """Bisection on ``alpha`` (Mücke et al. 2023, Algorithm 1).

    ``oracle(qubo)`` must return a minimiser (binary vector). Start with
    ``a = 0, b = 1``; if the optimum has more than ``k`` features move ``b``
    down, otherwise move ``a`` up. Stops at the first exact hit. If no exact
    hit occurs within ``max_iter`` the closest visited ``alpha`` is returned
    with ``status="inexact"``.
    """
    imp = np.asarray(I, dtype=np.float64).ravel()
    d = imp.shape[0]
    if not 0 <= k <= d:
        raise ValueError("k must be in [0, d]")
    a, b = 0.0, 1.0
    trace: list[tuple[float, int]] = []
    best: tuple[int, float, NDArray[np.int8], QUBO] | None = None
    for _ in range(max_iter):
        alpha = 0.5 * (a + b)
        qubo = build_mucke_qubo(imp, R, alpha=alpha, eps=eps, names=names)
        x = np.asarray(oracle(qubo)).astype(np.int8)
        kp = int(x.sum())
        trace.append((alpha, kp))
        gap = abs(kp - k)
        if best is None or gap < best[0]:
            best = (gap, alpha, x, qubo)
        if kp == k:
            return AlphaSearchResult(alpha, x, qubo, "exact", kp, trace)
        if kp > k:
            b = alpha
        else:
            a = alpha
        if b - a < tol:
            break
    assert best is not None
    _, alpha, x, qubo = best
    return AlphaSearchResult(alpha, x, qubo, "inexact", int(x.sum()), trace)
