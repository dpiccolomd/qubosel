# Quickstart

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from qubosel import QUBOFeatureSelector
from qubosel.datasets import make_clinical_synthetic

X, y, truth = make_clinical_synthetic(n_samples=80, random_state=0)

sel = QUBOFeatureSelector(k=3, random_state=0).fit(X, y)
print(sel.get_feature_names_out())  # selected columns
print(sel.alphas_, sel.alpha_status_)  # alpha found by bisection, "exact"/"inexact"

# nest the selector in cross-validation to avoid selection leakage
pipe = Pipeline(
    [
        ("select", QUBOFeatureSelector(k=3, random_state=0)),
        ("clf", LogisticRegression(max_iter=500)),
    ]
)
print(cross_val_score(pipe, X, y, cv=5, scoring="roc_auc").mean())
```

## Several solvers, several k, consensus

```python
sel = QUBOFeatureSelector(
    ks=[2, 3, 4], solvers=["sa", "sb"], n_repeats=5, min_votes=0.25, random_state=0, n_jobs=-1
).fit(X, y)
print(sel.consensus_.to_frame())  # votes, frequency, support per feature
print(sel.consensus_.selection_table())  # one row per run
```

## Lower-level API

```python
from qubosel import build_qubo, add_cardinality_penalty, get_solver

qubo = build_qubo(X, y, formulation="mucke", alpha=0.6)  # unconstrained
res = get_solver("sb").solve(qubo, seed=0)
print(res.selected, res.energy)

qubo_k = add_cardinality_penalty(build_qubo(X, y, formulation="cmi"), k=3, lam="auto")
print(get_solver("brute_force").solve(qubo_k).selected)
```

Continue with [Theory](theory.md) for the formulations and
[Validation](validation.md) for stability, permutation tests and benchmarks.
A complete script is in `examples/synthetic_end_to_end.py`.
