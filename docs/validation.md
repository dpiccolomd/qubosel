# Validation and leakage

Selecting features on the whole dataset and then cross-validating a model on
the selected columns leaks label information into the evaluation. Several published analyses declare this limitation. qubosel makes the nested procedure the
easy path: `QUBOFeatureSelector` is a `SelectorMixin`, so

```python
pipe = Pipeline([("select", QUBOFeatureSelector(k=3)), ("clf", GradientBoostingClassifier())])
cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42))
```

refits the selection inside every fold.

## Selection under shuffled labels

`permuted_label_selection(selector, X, y, n_permutations=200)` refits a clone
of the selector on permuted labels and returns, per feature, the null
selection frequency and a one-sided p-value for the selected features,
together with the fraction of permutations whose selected set is identical to
the observed one. A selector that returns the same set with shuffled labels is
describing the columns, not the outcome. Run it before any stability or
downstream analysis.

## Bootstrap stability

`bootstrap_stability(selector, X, y, n_resamples=200)` refits a clone on
bootstrap resamples (single-class resamples are redrawn) and returns per-feature
selection frequencies, the Nogueira–Sechidis–Brown stability
$\Phi = 1 - \frac{\frac1d\sum_f s_f^2}{\bar k/d\,(1-\bar k/d)}$ with
$s_f^2 = \frac{M}{M-1} p_f (1-p_f)$, and the mean pairwise Jaccard index.

## Permutation test

`permutation_test(estimator, X, y, cv, n_permutations=1000)` recomputes the CV
score on permuted labels; permutation $i$ uses `default_rng(seed + i)` and a
fresh clone, so when the estimator is a pipeline the *selection is part of the
null*. $p = (1 + \#\{\text{null} \ge \text{score}\}) / (n + 1)$.

## Classical baselines

`classical_selectors(k)` returns ANOVA (`SelectKBest`), `LassoTopK` (largest
|coef| of a `LassoCV`) and RFE with a random forest at the same cardinality;
`compare_selectors` tabulates them.

## Downstream evaluation

`evaluate_downstream(feature_sets, X, y, estimator, cv)` reports fold-mean AUC
and accuracy, pooled sensitivity/specificity/Brier, and a percentile bootstrap
CI of the pooled held-out AUC (`bootstrap_auc_ci`).
