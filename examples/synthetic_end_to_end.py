"""End-to-end example on synthetic data (no external data, < 2 minutes).

Steps: consensus QUBO selection with two solvers and three k values, bootstrap
stability, permutation test with the selector inside the pipeline, classical
baselines at the same cardinality and a fold-enclosed downstream comparison.
"""

import time

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from qubosel import QUBOFeatureSelector
from qubosel.datasets import make_clinical_synthetic
from qubosel.validation import (
    bootstrap_stability,
    classical_selectors,
    compare_selectors,
    evaluate_downstream,
    permutation_test,
    permuted_label_selection,
)


def main() -> None:
    t0 = time.time()
    X, y, truth = make_clinical_synthetic(n_samples=80, n_features=14, random_state=0)
    print(f"data: {X.shape}, positives {y.mean():.2f}, informative columns {truth['informative']}")

    sel = QUBOFeatureSelector(
        ks=[2, 3, 4], solvers=["sa", "sb"], n_repeats=3, min_votes=0.25, random_state=0, n_jobs=-1
    ).fit(X, y)
    print("\n== consensus (2 solvers x 3 k x 3 repeats)")
    print(sel.consensus_.to_frame().sort_values("votes", ascending=False).to_string(index=False))
    print("alphas:", sel.alphas_, sel.alpha_status_)
    selected = list(sel.get_feature_names_out())
    print("selected:", selected)

    print("\n== selection under shuffled labels (50 permutations, k=3, exact solver)")
    null = permuted_label_selection(
        QUBOFeatureSelector(k=3, solvers="brute_force"),
        X,
        y,
        n_permutations=50,
        random_state=0,
        n_jobs=-1,
    )
    print(null.to_frame().head(5).to_string(index=False))
    print(f"identical set under shuffled labels: {null.same_as_observed:.0%}")

    print("\n== bootstrap stability (50 resamples, k=3, exact solver)")
    stab = bootstrap_stability(
        QUBOFeatureSelector(k=3, solvers="brute_force"),
        X,
        y,
        n_resamples=50,
        random_state=0,
        n_jobs=-1,
    )
    print(stab.to_frame().head(6).to_string(index=False))
    print(f"Nogueira stability = {stab.nogueira:.3f}, mean Jaccard = {stab.jaccard_mean:.3f}")

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    clf = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42
    )
    pipe = Pipeline([("select", QUBOFeatureSelector(k=3, solvers="brute_force")), ("clf", clf)])
    print("\n== permutation test (selection nested in the pipeline, 100 permutations)")
    perm = permutation_test(pipe, X, y, cv=cv, n_permutations=100, random_state=0, n_jobs=-1)
    print(
        f"CV AUC = {perm.score:.3f}, null mean = {perm.null_scores.mean():.3f}, "
        f"p = {perm.p_value:.3f}"
    )

    print("\n== classical baselines at k=3")
    table = compare_selectors(classical_selectors(3), X, y)
    table["qubo"] = [int(c in selected) for c in X.columns]
    print(table.to_string())

    print("\n== downstream comparison (fold-enclosed, 500 bootstrap AUC resamples)")
    sets = {"qubo": selected, "truth": truth["informative"]}
    for name in table.columns[:-1]:
        sets[name] = list(table.index[table[name] == 1])
    res = evaluate_downstream(sets, X, y, clf, cv=cv, n_boot=500)
    cols = ["feature_set", "n_features", "auc_mean", "auc_ci_low", "auc_ci_high", "brier"]
    print(res[cols].round(3).to_string(index=False))
    print(f"\ndone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
