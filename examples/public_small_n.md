# Label-permutation control, k = 3, 200 permutations


## lung_cancer

UCI Lung Cancer, 32 x 56 ordinal features, 3 classes (n = 32, d = 56)

| objective | selected (index: n_unique) | identical set under shuffled labels | null frequency of selected |
|---|---|---|---|
| `cmi_hist10` | 2:4, 5:3, 11:4 | 0.01 | 0.63, 0.16, 0.45 |
| `cmi_cr_hist10` | 25:3, 30:3, 31:3 | 0.09 | 0.10, 0.68, 0.68 |
| `cmi_cr_quantile` | 40:3, 41:3, 51:3 | 0.00 | 0.54, 0.30, 0.14 |
| `hist10_plugin` | 2:4, 5:3, 55:2 | 0.00 | 0.14, 0.17, 0.05 |
| `quantile_plugin` | 19:3, 39:3, 52:3 | 0.00 | 0.07, 0.05, 0.08 |
| `quantile_expected` | 19:3, 39:3, 52:3 | 0.00 | 0.07, 0.07, 0.06 |

## breast_tissue

UCI Breast Tissue, 106 x 9 continuous features, 6 classes (n = 106, d = 9)

| objective | selected (index: n_unique) | identical set under shuffled labels | null frequency of selected |
|---|---|---|---|
| `cmi_hist10` | 1:105, 2:103, 8:105 | 0.12 | 0.99, 1.00, 0.12 |
| `cmi_cr_hist10` | 0:95, 7:105, 8:105 | 0.91 | 1.00, 0.91, 1.00 |
| `cmi_cr_quantile` | 4:105, 5:105, 6:105 | 0.00 | 0.11, 0.00, 0.22 |
| `hist10_plugin` | 1:105, 4:105, 8:105 | 0.00 | 0.04, 1.00, 0.00 |
| `quantile_plugin` | 1:105, 2:103, 3:105 | 0.34 | 0.99, 1.00, 0.34 |
| `quantile_expected` | 1:105, 2:103, 3:105 | 0.10 | 0.59, 0.85, 0.18 |

## colon

Colon (Alon 1999), 62 x 2000, binary; top 100 genes by variance (n = 62, d = 100)

| objective | selected (index: n_unique) | identical set under shuffled labels | null frequency of selected |
|---|---|---|---|
| `cmi_hist10` | 11:3, 17:3, 21:3 | 0.00 | 0.02, 0.04, 0.03 |
| `cmi_cr_hist10` | 23:3, 67:3, 83:3 | 0.73 | 0.74, 0.98, 0.98 |
| `cmi_cr_quantile` | 23:3, 67:3, 83:3 | 0.73 | 0.74, 0.98, 0.98 |
| `hist10_plugin` | 21:3, 71:3, 94:3 | 0.00 | 0.01, 0.20, 0.03 |
| `quantile_plugin` | 21:3, 71:3, 94:3 | 0.00 | 0.01, 0.20, 0.03 |
| `quantile_expected` | 21:3, 71:3, 77:3 | 0.00 | 0.06, 0.10, 0.08 |
| `rank` | 17:3, 21:3, 94:3 | 0.00 | 0.15, 0.07, 0.08 |

## lymphoma

Lymphoma, 96 x 4026, 9 classes; top 100 genes by variance (n = 96, d = 100)

| objective | selected (index: n_unique) | identical set under shuffled labels | null frequency of selected |
|---|---|---|---|
| `cmi_hist10` | 18:3, 44:3, 81:3 | 0.00 | 0.01, 0.04, 0.00 |
| `cmi_cr_hist10` | 10:3, 11:3, 12:3 | 0.29 | 0.29, 0.29, 0.29 |
| `cmi_cr_quantile` | 10:3, 11:3, 12:3 | 0.29 | 0.29, 0.29, 0.29 |
| `hist10_plugin` | 21:3, 34:3, 94:3 | 0.00 | 0.02, 0.15, 0.01 |
| `quantile_plugin` | 21:3, 34:3, 94:3 | 0.00 | 0.02, 0.15, 0.01 |
| `quantile_expected` | 21:3, 29:3, 81:3 | 0.00 | 0.01, 0.07, 0.01 |

# Mixed-type clinical tables


## hepatitis

UCI Hepatitis, 155 x 19 (13 binary, 6 continuous), binary outcome; n_unique per feature = [49, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 34, 83, 84, 29, 45, 2]

| objective | run | selected (index: n_unique) | identical set under shuffled labels |
|---|---|---|---|
| `cmi_hist10` | full n = 155 | 0:49, 16:29, 17:45 | 0.19 |
| `cmi_hist10` | n = 31, 20 subsamples, 100 perms | top-cardinality set in 0%, bottom-cardinality set in 0% | mean 0.17 (min 0.00, max 0.96) |
| `cmi_cr_hist10` | full n = 155 | 0:49, 14:83, 17:45 | 0.57 |
| `cmi_cr_hist10` | n = 31, 20 subsamples, 100 perms | top-cardinality set in 10%, bottom-cardinality set in 0% | mean 0.66 (min 0.00, max 1.00) |
| `cmi_cr_quantile` | full n = 155 | 13:34, 14:83, 16:29 | 0.70 |
| `cmi_cr_quantile` | n = 31, 20 subsamples, 100 perms | top-cardinality set in 0%, bottom-cardinality set in 70% | mean 0.61 (min 0.00, max 1.00) |
| `hist10_plugin` | full n = 155 | 2:2, 9:2, 11:2 | 0.00 |
| `hist10_plugin` | n = 31, 20 subsamples, 100 perms | top-cardinality set in 0%, bottom-cardinality set in 90% | mean 0.03 (min 0.00, max 0.24) |
| `quantile_plugin` | full n = 155 | 1:2, 10:2, 17:45 | 0.00 |
| `quantile_plugin` | n = 31, 20 subsamples, 100 perms | top-cardinality set in 0%, bottom-cardinality set in 35% | mean 0.01 (min 0.00, max 0.06) |
| `quantile_expected` | full n = 155 | 0:49, 2:2, 11:2 | 0.00 |
| `quantile_expected` | n = 31, 20 subsamples, 100 perms | top-cardinality set in 0%, bottom-cardinality set in 45% | mean 0.00 (min 0.00, max 0.03) |
| `rank` | full n = 155 | 0:49, 5:2, 16:29 | 0.00 |
| `rank` | n = 31, 20 subsamples, 100 perms | top-cardinality set in 0%, bottom-cardinality set in 5% | mean 0.00 (min 0.00, max 0.02) |

## statlog_heart

UCI Statlog Heart, 270 x 13 (mixed binary/categorical/continuous), binary outcome; n_unique per feature = [41, 2, 4, 47, 144, 2, 3, 90, 2, 39, 3, 4, 3]

| objective | run | selected (index: n_unique) | identical set under shuffled labels |
|---|---|---|---|
| `cmi_hist10` | full n = 270 | 0:41, 7:90, 9:39 | 0.13 |
| `cmi_hist10` | n = 31, 20 subsamples, 100 perms | top-cardinality set in 25%, bottom-cardinality set in 0% | mean 0.34 (min 0.01, max 0.80) |
| `cmi_cr_hist10` | full n = 270 | 0:41, 7:90, 9:39 | 0.18 |
| `cmi_cr_hist10` | n = 31, 20 subsamples, 100 perms | top-cardinality set in 40%, bottom-cardinality set in 0% | mean 0.60 (min 0.02, max 1.00) |
| `cmi_cr_quantile` | full n = 270 | 0:41, 3:47, 7:90 | 0.71 |
| `cmi_cr_quantile` | n = 31, 20 subsamples, 100 perms | top-cardinality set in 0%, bottom-cardinality set in 0% | mean 0.42 (min 0.00, max 0.99) |
| `hist10_plugin` | full n = 270 | 1:2, 10:3, 11:4 | 0.00 |
| `hist10_plugin` | n = 31, 20 subsamples, 100 perms | top-cardinality set in 0%, bottom-cardinality set in 15% | mean 0.25 (min 0.00, max 0.84) |
| `quantile_plugin` | full n = 270 | 6:3, 11:4, 12:3 | 0.00 |
| `quantile_plugin` | n = 31, 20 subsamples, 100 perms | top-cardinality set in 0%, bottom-cardinality set in 0% | mean 0.04 (min 0.00, max 0.14) |
| `quantile_expected` | full n = 270 | 6:3, 11:4, 12:3 | 0.01 |
| `quantile_expected` | n = 31, 20 subsamples, 100 perms | top-cardinality set in 0%, bottom-cardinality set in 0% | mean 0.01 (min 0.00, max 0.05) |
| `rank` | full n = 270 | 2:4, 10:3, 11:4 | 0.00 |
| `rank` | n = 31, 20 subsamples, 100 perms | top-cardinality set in 0%, bottom-cardinality set in 0% | mean 0.01 (min 0.00, max 0.03) |

## synthetic_cohort_n31

qubosel synthetic clinical cohort (7 binary, 7 continuous/ordinal), n = 31; n_unique per feature = [2, 31, 31, 9, 4, 10, 6, 2, 2, 2, 2, 1, 2, 2]

| objective | run | selected (index: n_unique) | identical set under shuffled labels |
|---|---|---|---|
| `cmi_hist10` | full n = 31 | 1:31, 2:31, 6:6 | 0.25 |
| `cmi_cr_hist10` | full n = 31 | 1:31, 3:9, 5:10 | 1.00 |
| `cmi_cr_quantile` | full n = 31 | 3:9, 5:10, 12:2 | 0.61 |
| `hist10_plugin` | full n = 31 | 0:2, 7:2, 10:2 | 0.10 |
| `quantile_plugin` | full n = 31 | 1:31, 3:9, 8:2 | 0.02 |
| `quantile_expected` | full n = 31 | 1:31, 3:9, 7:2 | 0.00 |
| `rank` | full n = 31 | 1:31, 3:9, 8:2 | 0.01 |
