"""Label-permutation control on public small datasets.

For each dataset the exact k=3 optimum of several relevance/redundancy
objectives is recomputed under 200 label permutations. A selection that is
identical under shuffled labels does not depend on the outcome. Datasets are
downloaded at run time into ``--cache`` (default ``~/.cache/qubosel``); nothing
is stored in the repository.

    uv run python examples/public_small_n.py [--permutations 200] [--n-jobs -1] [--max-features 100]

Datasets: four benchmark sets used by the classical corrected-MI papers at small n, two mixed-type
clinical tables (UCI Hepatitis, Statlog Heart) and the qubosel synthetic cohort:
Lung Cancer (UCI, 32 x 56, 3 classes), Breast Tissue (UCI, 106 x 9, 6 classes),
Colon (Alon et al., 62 x 2000, binary) and Lymphoma (96 x 4026, 9 classes), the
last two reduced to the ``--max-features`` most variable genes before
selection, as stated in the output.
"""

from __future__ import annotations

import argparse
import io
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from qubosel import QUBOFeatureSelector
from qubosel.solvers import KSubsetSolver
from qubosel.validation import permuted_label_selection

URLS = {
    "lung-cancer.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/lung-cancer/lung-cancer.data",
    "BreastTissue.xls": "https://archive.ics.uci.edu/ml/machine-learning-databases/00192/BreastTissue.xls",
    "colon.mat": "https://github.com/jundongl/scikit-feature/raw/master/skfeature/data/colon.mat",
    "lymphoma.mat": "https://github.com/jundongl/scikit-feature/raw/master/skfeature/data/lymphoma.mat",
    "hepatitis.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data",
    "heart.dat": "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat",
}


def fetch(name: str, cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / name
    if not path.exists():
        with urllib.request.urlopen(URLS[name], timeout=60) as r:
            path.write_bytes(r.read())
    return path


def load_datasets(cache: Path, max_features: int) -> dict[str, tuple[np.ndarray, np.ndarray, str]]:
    from scipy.io import loadmat

    out = {}
    raw = pd.read_csv(fetch("lung-cancer.data", cache), header=None, na_values="?")
    y = raw.iloc[:, 0].to_numpy()
    X = raw.iloc[:, 1:].apply(lambda c: c.fillna(c.mode().iloc[0])).to_numpy(float)
    out["lung_cancer"] = (X, y, "UCI Lung Cancer, 32 x 56 ordinal features, 3 classes")

    bt = pd.read_excel(io.BytesIO(fetch("BreastTissue.xls", cache).read_bytes()), sheet_name="Data")
    y = bt["Class"].to_numpy()
    X = bt.drop(columns=["Case #", "Class"]).to_numpy(float)
    out["breast_tissue"] = (X, y, "UCI Breast Tissue, 106 x 9 continuous features, 6 classes")

    for name, desc in (
        ("colon", "Colon (Alon 1999), 62 x 2000, binary"),
        ("lymphoma", "Lymphoma, 96 x 4026, 9 classes"),
    ):
        m = loadmat(fetch(f"{name}.mat", cache))
        X, y = m["X"].astype(float), m["Y"].ravel()
        keep = np.argsort(-X.var(axis=0))[:max_features]
        out[name] = (X[:, np.sort(keep)], y, f"{desc}; top {max_features} genes by variance")
    return out


def load_mixed_datasets(cache: Path) -> dict[str, tuple[np.ndarray, np.ndarray, str]]:
    """Small clinical tables with binary indicators next to continuous measurements."""
    out = {}
    h = pd.read_csv(fetch("hepatitis.data", cache), header=None, na_values="?")
    y = h[0].to_numpy()
    X = h.iloc[:, 1:].apply(lambda c: c.fillna(c.median())).to_numpy(float)
    out["hepatitis"] = (X, y, "UCI Hepatitis, 155 x 19 (13 binary, 6 continuous), binary outcome")
    d = pd.read_csv(fetch("heart.dat", cache), header=None, sep=r"\s+")
    y = d[13].to_numpy()
    X = d.iloc[:, :13].to_numpy(float)
    out["statlog_heart"] = (
        X,
        y,
        "UCI Statlog Heart, 270 x 13 (mixed binary/categorical/continuous), binary outcome",
    )
    return out


def stratified_subsample(y: np.ndarray, n: int, rng: np.random.Generator, min_per_class: int = 5):
    classes = np.unique(y)
    for _ in range(100):
        idx = rng.choice(y.shape[0], size=n, replace=False)
        if all((y[idx] == c).sum() >= min_per_class for c in classes):
            return np.sort(idx)
    raise RuntimeError("could not draw a subsample with enough cases per class")


def objectives(k: int, binary: bool) -> dict[str, QUBOFeatureSelector]:
    common = {"k": k, "cardinality": None, "alpha": 0.5, "solvers": KSubsetSolver(k=k)}
    objs = {
        "cmi_hist10": QUBOFeatureSelector(
            k=k,
            cardinality=None,
            solvers=KSubsetSolver(k=k),
            formulation="cmi",
            binning="legacy",
            n_bins=10,
            base=2,
        ),
        "cmi_cr_hist10": QUBOFeatureSelector(
            k=k,
            cardinality=None,
            solvers=KSubsetSolver(k=k),
            formulation="cmi_cr",
            binning="legacy",
            n_bins=10,
            base=2,
        ),
        "cmi_cr_quantile": QUBOFeatureSelector(
            k=k,
            cardinality=None,
            solvers=KSubsetSolver(k=k),
            formulation="cmi_cr",
            binning="quantile",
            base=2,
        ),
        "hist10_plugin": QUBOFeatureSelector(binning="legacy", n_bins=10, **common),
        "quantile_plugin": QUBOFeatureSelector(binning="quantile", **common),
        "quantile_expected": QUBOFeatureSelector(
            binning="quantile", mi_correction="expected", **common
        ),
    }
    if binary:
        objs["rank"] = QUBOFeatureSelector(formulation="rank", **common)
    return objs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--max-features", type=int, default=100)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--datasets", default="", help="comma-separated subset of dataset names")
    ap.add_argument("--cache", default=str(Path.home() / ".cache" / "qubosel"))
    ap.add_argument("--out", default=str(Path(__file__).with_name("public_small_n")))
    args = ap.parse_args()
    t0 = time.time()
    rows, lines = (
        [],
        [f"# Label-permutation control, k = {args.k}, {args.permutations} permutations\n"],
    )
    wanted = {d.strip() for d in args.datasets.split(",") if d.strip()}
    bench = (
        load_datasets(Path(args.cache), args.max_features)
        if (not wanted or wanted & {"lung_cancer", "breast_tissue", "colon", "lymphoma"})
        else {}
    )
    for dname, (X, y, desc) in bench.items():
        if wanted and dname not in wanted:
            continue
        binary = np.unique(y).shape[0] == 2
        n_unique = np.array([np.unique(X[:, j]).shape[0] for j in range(X.shape[1])])
        print(f"\n== {dname}: {desc} (n={X.shape[0]}, d={X.shape[1]})")
        lines.append(f"\n## {dname}\n\n{desc} (n = {X.shape[0]}, d = {X.shape[1]})\n")
        lines.append(
            "| objective | selected (index: n_unique) | identical set under shuffled labels | "
            "null frequency of selected |"
        )
        lines.append("|---|---|---|---|")
        for oname, sel in objectives(args.k, binary).items():
            res = permuted_label_selection(
                sel, X, y, n_permutations=args.permutations, random_state=0, n_jobs=args.n_jobs
            )
            picked = np.flatnonzero(res.observed)
            desc_sel = ", ".join(f"{i}:{n_unique[i]}" for i in picked)
            nullf = ", ".join(f"{res.null_frequency[i]:.2f}" for i in picked)
            print(
                f"  {oname:18s} selected [{desc_sel}]  identical {res.same_as_observed:.0%}  "
                f"null freq [{nullf}]"
            )
            lines.append(f"| `{oname}` | {desc_sel} | {res.same_as_observed:.2f} | {nullf} |")
            rows.append(
                {
                    "dataset": dname,
                    "n": X.shape[0],
                    "d": X.shape[1],
                    "objective": oname,
                    "selected": "|".join(map(str, picked)),
                    "selected_n_unique": "|".join(map(str, n_unique[picked])),
                    "identical_fraction": res.same_as_observed,
                    "null_frequency_selected": "|".join(
                        f"{v:.3f}" for v in res.null_frequency[picked]
                    ),
                    "median_n_unique_all": float(np.median(n_unique)),
                }
            )
    # ---- mixed-type clinical tables: full size and subsampled to n = 31
    lines.append("\n# Mixed-type clinical tables\n")
    sub_n, n_sub, sub_perm = 31, 20, 100
    mixed = load_mixed_datasets(Path(args.cache))
    from qubosel.datasets import make_clinical_cohort

    Xs, ys, _ = make_clinical_cohort(n_samples=sub_n, random_state=0)
    mixed["synthetic_cohort_n31"] = (
        Xs.to_numpy(float),
        ys,
        "qubosel synthetic clinical cohort (7 binary, 7 continuous/ordinal), n = 31",
    )
    for dname, (X, y, desc) in mixed.items():
        if wanted and dname not in wanted:
            continue
        n_unique = np.array([np.unique(X[:, j]).shape[0] for j in range(X.shape[1])])
        binary = np.unique(y).shape[0] == 2
        print(
            f"\n== {dname}: {desc} (n={X.shape[0]}, d={X.shape[1]}); n_unique={n_unique.tolist()}"
        )
        lines.append(f"\n## {dname}\n\n{desc}; n_unique per feature = {n_unique.tolist()}\n")
        lines.append(
            "| objective | run | selected (index: n_unique) | identical set under shuffled labels |"
        )
        lines.append("|---|---|---|---|")
        for oname, sel in objectives(args.k, binary).items():
            res = permuted_label_selection(
                sel, X, y, n_permutations=args.permutations, random_state=0, n_jobs=args.n_jobs
            )
            picked = np.flatnonzero(res.observed)
            desc_sel = ", ".join(f"{i}:{n_unique[i]}" for i in picked)
            print(
                f"  {oname:18s} full n={X.shape[0]}: selected [{desc_sel}]  identical {res.same_as_observed:.0%}"
            )
            lines.append(
                f"| `{oname}` | full n = {X.shape[0]} | {desc_sel} | {res.same_as_observed:.2f} |"
            )
            rows.append(
                {
                    "dataset": dname,
                    "n": X.shape[0],
                    "d": X.shape[1],
                    "objective": oname,
                    "run": "full",
                    "selected": "|".join(map(str, picked)),
                    "selected_n_unique": "|".join(map(str, n_unique[picked])),
                    "identical_fraction": res.same_as_observed,
                    "null_frequency_selected": "|".join(
                        f"{v:.3f}" for v in res.null_frequency[picked]
                    ),
                    "median_n_unique_all": float(np.median(n_unique)),
                }
            )
            if X.shape[0] > sub_n:
                rng = np.random.default_rng(1)
                idents, high_card, low_card = [], [], []
                for r in range(n_sub):
                    idx = stratified_subsample(y, sub_n, rng)
                    rs = permuted_label_selection(
                        sel,
                        X[idx],
                        y[idx],
                        n_permutations=sub_perm,
                        random_state=r,
                        n_jobs=args.n_jobs,
                    )
                    idents.append(rs.same_as_observed)
                    nu = np.array([np.unique(X[idx][:, j]).shape[0] for j in range(X.shape[1])])
                    high_card.append(
                        set(np.flatnonzero(rs.observed))
                        <= set(np.flatnonzero(nu >= np.sort(nu)[-args.k]))
                    )
                    low_card.append(
                        set(np.flatnonzero(rs.observed))
                        <= set(np.flatnonzero(nu <= np.sort(nu)[args.k - 1]))
                    )
                print(
                    f"  {oname:18s} n=31 x{n_sub}: mean identical {np.mean(idents):.2f} "
                    f"(min {np.min(idents):.2f}, max {np.max(idents):.2f}); "
                    f"selected set within the {args.k} highest-cardinality features in {np.mean(high_card):.0%}, within the {args.k} lowest in {np.mean(low_card):.0%} of subsamples"
                )
                lines.append(
                    f"| `{oname}` | n = 31, {n_sub} subsamples, {sub_perm} perms | "
                    f"top-cardinality set in {np.mean(high_card):.0%}, bottom-cardinality set in {np.mean(low_card):.0%} | "
                    f"mean {np.mean(idents):.2f} (min {np.min(idents):.2f}, max {np.max(idents):.2f}) |"
                )
                rows.append(
                    {
                        "dataset": dname,
                        "n": sub_n,
                        "d": X.shape[1],
                        "objective": oname,
                        "run": f"subsample_x{n_sub}",
                        "selected": "",
                        "selected_n_unique": "",
                        "identical_fraction": float(np.mean(idents)),
                        "null_frequency_selected": "",
                        "median_n_unique_all": float(np.median(n_unique)),
                        "top_cardinality_fraction": float(np.mean(high_card)),
                        "bottom_cardinality_fraction": float(np.mean(low_card)),
                    }
                )
    pd.DataFrame(rows).to_csv(args.out + ".csv", index=False)
    Path(args.out + ".md").write_text("\n".join(lines) + "\n")
    print(f"\ndone in {time.time() - t0:.0f}s -> {args.out}.csv")


if __name__ == "__main__":
    main()
