"""Diagnostics for the label-permutation control: ties, saturation, type composition, independent recomputation.

For the class-conditional-redundancy CMI objective (cmi_cr) and the Mücke plug-in objective on
the mixed-type clinical tables subsampled to n = 31, this script reports

1. the number of 3-subsets tied with the optimum (|E - E_min| < 1e-9) and the
   gap to the best non-tied subset, under the real labels;
2. whether the selected features are all "continuous-type" (reach the 10-bin cap
   in the subsample) or all binary, under real and permuted labels;
3. the identical-set fraction with the feature order randomly permuted before
   enumeration (breaks lexicographic tie-breaking);
4. saturation of the pair term: mean I(x_i; x_j | y) for continuous pairs versus
   its maximum H(x_i | y);
5. an INDEPENDENT recomputation of the cmi_cr objective with numpy.histogramdd
   (no qubosel code) and exhaustive enumeration of 3-subsets, compared with the
   package's selection on the same subsample.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from public_small_n import load_mixed_datasets, stratified_subsample

from qubosel import build_qubo
from qubosel.datasets import make_clinical_cohort


# ---------------------------------------------------------------- independent cmi_cr objective
def _prob(arr: np.ndarray) -> np.ndarray:
    bins = [min(len(np.unique(arr[:, c])), 10) for c in range(arr.shape[1])]
    freq, _ = np.histogramdd(arr, bins)
    return freq / freq.sum()


def _H(p: np.ndarray) -> float:
    f = p.ravel()
    f = f[f > 0]
    return float(-(f * np.log2(f)).sum())


def _cond_H(p: np.ndarray, *cond: int) -> float:
    axis = tuple(i for i in range(p.ndim) if i not in cond)
    return _H(p) - _H(p.sum(axis=axis))


def _mi(p: np.ndarray, j: int) -> float:
    return _H(p.sum(axis=j)) - _cond_H(p, j)


def _cmi(p: np.ndarray, j: int, *cond: int) -> float:
    mcond = [i - 1 if i > j else i for i in cond]
    return _cond_H(p.sum(axis=j), *mcond) - _cond_H(p, j, *cond)


def cmi_cr_qubo_independent(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Upper-triangular Q of the cmi_cr objective, recomputed from scratch with histogramdd."""
    Xs = (X - X.min(0)) / np.where(X.max(0) > X.min(0), X.max(0) - X.min(0), 1.0)
    d = X.shape[1]
    Q = np.zeros((d, d))
    for i in range(d):
        Q[i, i] = -_mi(_prob(np.column_stack([Xs[:, i], y])), 1)
    for i, j in itertools.combinations(range(d), 2):
        p01 = _prob(np.column_stack([Xs[:, i], Xs[:, j], y]))
        p10 = _prob(np.column_stack([Xs[:, j], Xs[:, i], y]))
        Q[i, j] = -(_cmi(p01, 1, 2) + _cmi(p10, 1, 2))
    return Q


def energies_k3(Q_upper: np.ndarray):
    d = Q_upper.shape[0]
    subs = np.array(list(itertools.combinations(range(d), 3)))
    diag = np.diag(Q_upper)
    e = diag[subs].sum(1)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        e = e + Q_upper[subs[:, a], subs[:, b]]
    return subs, e


def ties_and_gap(e: np.ndarray, tol: float = 1e-9) -> tuple[int, float]:
    emin = e.min()
    tied = int((np.abs(e - emin) < tol).sum())
    rest = e[np.abs(e - emin) >= tol]
    return tied, float(rest.min() - emin) if rest.size else float("nan")


def mucke_upper(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    q = build_qubo(X, y, formulation="mucke", alpha=0.5, binning="legacy", n_bins=10)
    return q.to_upper()


def saturation(X: np.ndarray, y: np.ndarray, cont: np.ndarray) -> tuple[float, float]:
    """Mean I(x_i;x_j|y) over continuous pairs and mean of its bound H(x_i|y)."""
    Xs = (X - X.min(0)) / np.where(X.max(0) > X.min(0), X.max(0) - X.min(0), 1.0)
    idx = np.flatnonzero(cont)
    vals, bounds = [], []
    for i, j in itertools.combinations(idx, 2):
        p = _prob(np.column_stack([Xs[:, i], Xs[:, j], y]))
        vals.append(_cmi(p, 1, 2))
        bounds.append(_cond_H(_prob(np.column_stack([Xs[:, i], y])), 1))
    return float(np.mean(vals)), float(np.mean(bounds))


def main() -> None:
    rng = np.random.default_rng(1)
    data = load_mixed_datasets(Path.home() / ".cache" / "qubosel")
    Xs, ys, _ = make_clinical_cohort(n_samples=31, random_state=0)
    data["synthetic_cohort_n31"] = (Xs.to_numpy(float), ys, "synthetic")
    n_perm, n_sub = 100, 10
    for dname, (X, y, _desc) in data.items():
        _, yc = np.unique(y, return_inverse=True)
        yc = yc.ravel()
        print(f"\n== {dname} (n={X.shape[0]}, d={X.shape[1]})")
        for objective, build in (
            ("cmi_cr", cmi_cr_qubo_independent),
            ("mucke_hist10", mucke_upper),
        ):
            rows = []
            for r in range(n_sub if X.shape[0] > 31 else 1):
                idx = stratified_subsample(yc, 31, rng) if X.shape[0] > 31 else np.arange(31)
                Xi, yi = X[idx], yc[idx]
                nu = np.array([np.unique(Xi[:, j]).shape[0] for j in range(Xi.shape[1])])
                cont = nu >= 10
                Q = build(Xi, yi)
                subs, e = energies_k3(Q)
                best = subs[int(np.argmin(e))]
                tied, gap = ties_and_gap(e)
                # permutations: identical set with fixed order; identical with random feature order;
                # type composition of the selected set
                prng = np.random.default_rng(r)
                same_fixed = same_random = all_cont = all_bin = 0
                for _ in range(n_perm):
                    yp = prng.permutation(yi)
                    Qp = build(Xi, yp)
                    subs_p, ep = energies_k3(Qp)
                    sel = set(subs_p[int(np.argmin(ep))])
                    same_fixed += sel == set(best)
                    perm = prng.permutation(Xi.shape[1])  # random tie-breaking via feature order
                    Qr = Qp[np.ix_(perm, perm)]
                    Qr = np.triu(Qr) + np.triu(Qr.T, 1)  # keep upper form after reordering
                    subs_r, er = energies_k3(Qr)
                    sel_r = set(perm[subs_r[int(np.argmin(er))]])
                    same_random += sel_r == set(best)
                    all_cont += cont[list(sel)].all()
                    all_bin += (nu[list(sel)] == 2).all()
                sat, bound = saturation(Xi, yi, cont) if cont.sum() >= 2 else (np.nan, np.nan)
                rows.append(
                    (
                        tied,
                        gap,
                        same_fixed / n_perm,
                        same_random / n_perm,
                        all_cont / n_perm,
                        all_bin / n_perm,
                        sat,
                        bound,
                        cont[best].all(),
                        (nu[best] == 2).all(),
                    )
                )
            a = np.array(rows, dtype=float)
            print(
                f"  {objective:13s} ties@opt median {np.median(a[:, 0]):.0f} (max {a[:, 0].max():.0f}); "
                f"gap to next median {np.median(a[:, 1]):.3g}; identical: fixed order {a[:, 2].mean():.2f}, "
                f"random order {a[:, 3].mean():.2f}; selected all-continuous under null {a[:, 4].mean():.2f}, "
                f"all-binary {a[:, 5].mean():.2f}; real-label selection all-continuous {a[:, 8].mean():.2f}, "
                f"all-binary {a[:, 9].mean():.2f}; pair term mean {np.nanmean(a[:, 6]):.2f} vs bound {np.nanmean(a[:, 7]):.2f} bits"
            )
        # independent recomputation vs package on one subsample
        idx = (
            stratified_subsample(yc, 31, np.random.default_rng(7))
            if X.shape[0] > 31
            else np.arange(31)
        )
        Qi = cmi_cr_qubo_independent(X[idx], yc[idx])
        Qp = build_qubo(
            X[idx], yc[idx], formulation="cmi_cr", binning="legacy", n_bins=10, base=2
        ).to_upper()
        subs, ei = energies_k3(Qi)
        _, ep = energies_k3(Qp)
        print(
            f"  independent vs package cmi_cr QUBO: max |dQ| = {np.abs(Qi - Qp).max():.2e}; "
            f"same k=3 optimum: {set(subs[int(np.argmin(ei))]) == set(subs[int(np.argmin(ep))])}"
        )


if __name__ == "__main__":
    main()
