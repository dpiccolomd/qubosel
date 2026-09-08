# Label-permutation control on public data

`examples/public_small_n.py` recomputes the exact k = 3 optimum of several
objectives under label permutations on public datasets: four benchmark sets
used by the classical corrected-MI literature at small sample sizes (Sharmin
et al. 2019; Naghibi et al. 2015), UCI Lung Cancer (n = 32), UCI Breast Tissue
(n = 106), Colon (n = 62) and Lymphoma (n = 96), the last two reduced to the
100 most variable genes; two mixed-type clinical tables, UCI Hepatitis (n = 155)
and Statlog Heart (n = 270), at full size and subsampled twenty times to n = 31;
and qubosel's synthetic mixed cohort at n = 31. Datasets are downloaded at run
time; nothing is stored in the repository.

Objectives are solved exactly by enumerating all 3-subsets (`KSubsetSolver`);
the Mücke-structure ones use α = 0.5:

| name | estimator | bins |
|---|---|---|
| `cmi_cr_hist10` | class-conditional-redundancy CMI objective (`formulation="cmi_cr"`): `-I(x_i;y)`, `-2 I(x_i;x_j|y)`, plug-in, bits | equal-width, min(n_unique, 10) |
| `cmi_hist10` | D-Wave example objective (`formulation="cmi"`): `-I(x_i;y)`, `-[I(x_j;y|x_i)+I(x_i;y|x_j)]`, plug-in, bits | equal-width, min(n_unique, 10) |
| `hist10_plugin` | plug-in MI, Mücke structure | equal-width, min(n_unique, 10) |
| `quantile_plugin` | plug-in MI | quantile, `max(2, floor(sqrt(n)/2))` |
| `quantile_expected` | reliable MI (plug-in minus exact permutation expectation) | quantile |
| `rank` (binary outcomes only) | 2\|AUC − 0.5\|, Spearman ρ² | none |

For each objective the table reports the selected features with their number
of distinct values, the fraction of permutations in which the selected set is
identical to the observed one, and each selected feature's selection
frequency under the null.

## Results

**Identical selected set under shuffled labels** (fraction of permutations; mean over the 20 subsamples where noted; 200 permutations at full size, 100 per subsample)

| dataset | `cmi_cr_hist10` | `cmi_hist10` | `hist10_plugin` | `quantile_plugin` | `quantile_expected` | `rank` |
|---|---|---|---|---|---|---|
| Lung Cancer 32 × 56 (all features 2–4 levels) | 0.09 | 0.01 | 0.00 | 0.00 | 0.00 | – |
| Breast Tissue 106 × 9 (all continuous) | 0.91 | 0.12 | 0.00 | 0.34 | 0.10 | – |
| Colon 62 × 100 (all 3 levels) | 0.73 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Lymphoma 96 × 100 (all 3 levels) | 0.29 | 0.00 | 0.00 | 0.00 | 0.00 | – |
| Hepatitis 155 × 19 (13 binary, 6 continuous) | 0.57 | 0.19 | 0.00 | 0.00 | 0.00 | 0.00 |
| Hepatitis 155 × 19 (13 binary, 6 continuous), n = 31, 20 subsamples | 0.66 | 0.17 | 0.03 | 0.01 | 0.00 | 0.00 |
| Statlog Heart 270 × 13 (mixed) | 0.18 | 0.13 | 0.00 | 0.00 | 0.01 | 0.00 |
| Statlog Heart 270 × 13 (mixed), n = 31, 20 subsamples | 0.60 | 0.34 | 0.25 | 0.04 | 0.01 | 0.01 |
| synthetic mixed cohort, n = 31 | 1.00 | 0.25 | 0.10 | 0.02 | 0.00 | 0.01 |

**Diagnostics at n = 31** (`examples/public_small_n_diagnostics.py`; 10 subsamples, 100 permutations each)

| dataset | objective | tied subsets at the optimum | gap to runner-up (bits) | identical set, fixed feature order | identical set, random feature order | selected set all-continuous under null | all-binary under null |
|---|---|---|---|---|---|---|---|
| Hepatitis | cmi_cr | 1 | 0.30 | 0.63 | 0.63 | 1.00 | 0.00 |
| Hepatitis | Mücke hist10 | 1 | 0.003 | 0.03 | 0.03 | 0.00 | 0.96 |
| Statlog Heart | cmi_cr | 1 | 0.38 | 0.59 | 0.59 | 1.00 | 0.00 |
| Statlog Heart | Mücke hist10 | 1 | 0.007 | 0.10 | 0.10 | 0.00 | 0.29 |
| synthetic cohort | cmi_cr | 1 | 0.99 | 0.98 | 0.98 | – | 0.00 |
| synthetic cohort | Mücke hist10 | 1 | 0.002 | 0.12 | 0.12 | 0.00 | 0.80 |

"Continuous" = reaches the 10-bin cap in the subsample. The mean pair term
I(x_i; x_j | y) over continuous pairs is 1.0–1.8 bits against a bound H(x_i | y)
of 2.2–2.8 bits (inflated, not saturated). An independent recomputation of the
cmi_cr objective (`numpy.histogramdd`, target last, no qubosel
code) agrees with the package to 5e-15 and selects the same optimum.

Raw output: `examples/public_small_n.csv`. Runtime about 10 minutes on 8 cores.

## Reading the tables

- The conditional-MI formulation adopted from the D-Wave example (`cmi_cr_hist10`,
  the SDK's exact objective: diagonal `-I(x_i; y)`, pair term `-2 I(x_i; x_j | y)`, plug-in
  MI on `min(n_unique, 10)` equal-width bins) returns the same set whatever the
  labels in 91 % of permutations on Breast Tissue, 73 % on Colon, and in 60–66 %
  on average (up to 100 % for single subsamples) on the mixed-type clinical
  tables at n = 31. Both its terms are plug-in estimates that inflate with the
  number of distinct values and both enter with the same sign, so under
  shuffled labels the selected set consists only of continuous features in
  100 % of permutations. The optimum is unique (no ties), the gap to the
  runner-up is 0.3–1.0 bits, and randomising the feature order changes nothing:
  the invariance is a property of the objective, not of tie-breaking.
- The D-Wave form (`cmi_hist10`, pair term `-[I(x_j; y | x_i) + I(x_i; y | x_j)]`)
  shows the same behaviour more weakly (12–34 %).
- The Mücke relevance-minus-redundancy objective with the same bins
  (`hist10_plugin`) is biased the opposite way: its redundancy term inflates
  too, pairs of many-valued features are penalised, and the selected set
  consists only of binary features in 80–96 % of permutations. Which binary
  features are chosen does depend on the labels, but the energy gap to the
  runner-up is 0.002–0.007 bits, so that choice is fragile.
- Quantile bins, the reliable-MI correction (`quantile_expected`) and the rank
  objective bring the identical-set fraction to 0–4 % everywhere and show no
  type preference in either direction.
- On benchmark sets where every feature has the same number of levels (Lung
  Cancer, Colon, Lymphoma as distributed by scikit-feature) the type-driven
  artefact cannot express itself; there the Mücke objective is label-dependent
  and the cmi_cr objective still shows partial invariance (Colon 73 %).

## Data handling and limits

Lung Cancer: 5 missing cells in two columns imputed by the column mode.
Hepatitis: 167 missing cells (67 of them in the protime column) imputed by the
column median; class coding 1 = die, 2 = live. Statlog Heart: no missing
values; 1 = absent, 2 = present. Breast Tissue: six classes of 14–22 cases.
Lymphoma: nine classes, the smallest with 2 cases, hence run at full size only.
Colon and Lymphoma as distributed by scikit-feature are already discretised to
three levels and were reduced to the 100 most variable genes.

k = 3 only; one bin rule per objective; 20 subsamples of 31 rows (10 in the
diagnostics); Breast Tissue and Lymphoma are multiclass, for which the rank
objective is not defined. One to two hundred permutations give a standard
error of a few percentage points on a fraction.

