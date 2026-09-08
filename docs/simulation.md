# Ground-truth simulation

`qubosel.simulation` answers the question a small clinical study cannot answer
from its own data: **at which sample size does a QUBO feature-selection
objective recover predictors that are known to be true?**

## Design

- `make_clinical_cohort(n, truth, effect)` draws a 14-column cohort with the
  structure of an small clinical cohort: sex, age and disease duration
  (correlated), two zero-inflated event counts and their sum, an ordinal
  score, seven binary indicators. The binary outcome is a logistic function of the ``truth``
  columns (standardised, coefficient 0.9, single-feature AUC around 0.7).
- `standard_objectives(k)` returns candidate selectors that share the Mücke
  structure, α = 0.5, an exact cardinality penalty and exhaustive enumeration,
  and differ only in the estimator:

| name | relevance / redundancy | discretisation |
|---|---|---|
| `mucke_hist10` | plug-in MI | equal-width, 10 bins |
| `mucke_mm_quantile4` | plug-in MI with Miller-Madow correction | 4 quantile bins |
| `mucke_permutation` | plug-in MI minus its Monte-Carlo permutation mean | 4 quantile bins |
| `mucke_expected` | plug-in MI minus its exact permutation expectation (reliable MI, Mandros et al.) | 4 quantile bins |
| `mucke_ksg` | kNN (Kraskov) MI | none |
| `rank` | 2\|AUC − 0.5\| and Spearman ρ² | none |
| `anova_topk` | univariate F-test (no redundancy) | none |

- `recovery_curve(objectives, n_list, scenarios, n_replicates)` reports, per
  scenario, ``n`` and objective, the exact-recovery rate of the true set, its
  binomial standard error and the mean Jaccard index. Every objective sees the
  same cohorts (seed = ``random_state + 1000 n + replicate``). Solver noise is
  excluded by construction; this isolates the objective.

```bash
uv run python examples/recovery_curve.py --replicates 60
```

## Results (60 replicates, k = 3, seed 0)

**continuous/ordinal truth** (exact-recovery rate)

| n | `mucke_hist10` | `mucke_mm_quantile4` | `mucke_permutation` | `mucke_expected` | `mucke_ksg` | `rank` | `anova_topk` |
|---|---|---|---|---|---|---|---|
| 31 | 0.00 | 0.02 | 0.03 | 0.03 | 0.03 | 0.20 | 0.07 |
| 60 | 0.00 | 0.10 | 0.15 | 0.10 | 0.03 | 0.42 | 0.08 |
| 100 | 0.00 | 0.07 | 0.13 | 0.10 | 0.05 | 0.47 | 0.18 |
| 200 | 0.00 | 0.33 | 0.35 | 0.37 | 0.17 | 0.82 | 0.35 |
| 300 | 0.00 | 0.48 | 0.48 | 0.50 | 0.17 | 0.83 | 0.38 |
| 500 | 0.00 | 0.50 | 0.53 | 0.50 | 0.30 | 0.88 | 0.42 |

**one binary truth** (exact-recovery rate)

| n | `mucke_hist10` | `mucke_mm_quantile4` | `mucke_permutation` | `mucke_expected` | `mucke_ksg` | `rank` | `anova_topk` |
|---|---|---|---|---|---|---|---|
| 31 | 0.00 | 0.05 | 0.05 | 0.07 | 0.08 | 0.08 | 0.15 |
| 60 | 0.00 | 0.13 | 0.18 | 0.17 | 0.03 | 0.30 | 0.43 |
| 100 | 0.00 | 0.35 | 0.38 | 0.33 | 0.28 | 0.50 | 0.50 |
| 200 | 0.00 | 0.68 | 0.70 | 0.68 | 0.23 | 0.92 | 0.77 |
| 300 | 0.00 | 0.93 | 0.93 | 0.93 | 0.28 | 0.97 | 0.90 |
| 500 | 0.00 | 0.97 | 0.98 | 0.97 | 0.45 | 1.00 | 0.82 |

Raw output: `examples/recovery_curve.csv`. Runtime about 2 minutes on 8 cores.

## Reading the table

Histogram MI without correction (`mucke_hist10`) never recovers the truth,
at any ``n`` up to 500: it ranks features by cardinality (see
[Label-permutation control on public data](public-small-n.md) for the real-data counterpart). At
``n = 31`` no objective exceeds 20 %. The rank-based objective is reliable
from about 200 samples; the Miller-Madow, Monte-Carlo permutation and exact
reliable-MI (`mucke_expected`) histogram estimators behave alike, reliable
from about 300 when a true predictor is binary and only partially (about
50 %) when all three are continuous or ordinal. The kNN
estimator with three neighbours is weak on this mixed continuous/binary
design. The redundancy term is what lets the rank objective beat the
univariate baseline when true predictors are correlated (the total count with
its component): 0.82 versus 0.35 at ``n = 200``.

## Limits

Two scenarios, one effect size, one correlation structure, exact ``k``.
Other effect sizes, unknown ``k``, α values, and the behaviour of heuristic
solvers are not covered. Sixty replicates give a standard error of up to
about 6.5 percentage points on a rate.
