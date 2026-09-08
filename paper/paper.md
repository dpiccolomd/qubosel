---
title: 'qubosel: combinatorial (QUBO-based) feature selection with quantum-inspired solvers, consensus and validation'
tags:
  - Python
  - feature selection
  - QUBO
  - simulated annealing
  - simulated bifurcation
  - QAOA
  - scikit-learn
authors:
  - name: Daniele Piccolo
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Independent researcher, Italy
    index: 1
date: 8 September 2026
bibliography: paper.bib
---

# Summary

Feature selection asks which subset of measured variables best explains an
outcome while avoiding redundancy. Written as a Quadratic Unconstrained Binary
Optimisation (QUBO), the problem assigns one binary variable per feature, a
diagonal term rewarding relevance (mutual information with the outcome) and
off-diagonal terms penalising redundancy (mutual information between
features) [@mucke2023; @nguyen2014]. `qubosel` builds these objectives,
minimises them with interchangeable solvers—exhaustive search, simulated
annealing, simulated bifurcation [@goto2021] and QAOA [@farhi2014] through
PennyLane [@bergholm2018]—and aggregates repeated runs into a consensus. The
selector is a scikit-learn transformer, so selection can be nested inside
cross-validation, and the package ships the validation tools a clinical study
needs: bootstrap stability [@nogueira2018], permutation tests with the
selector inside the null, classical baselines at the same cardinality and
fold-enclosed downstream evaluation with bootstrap confidence intervals.

# Statement of need

Small clinical datasets (tens of patients, a dozen candidate predictors) are
where joint, non-greedy selection matters most and where evaluation is most
easily contaminated by leakage. Existing tools cover isolated pieces:
`seleqt` implements the Mücke et al. formulation with a single exact solver,
D-Wave's `dimod` offers the QUBO container and `neal` a sampler, and the
`simulated-bifurcation` package implements SB on top of PyTorch. None of them
provides a scikit-learn selector, a consensus layer with an honest
denominator, or the negative controls that small clinical samples require. In
particular, plug-in mutual information is biased upwards in proportion to the
number of distinct values of a variable; with tens of samples an MI-based QUBO
ranks features by cardinality rather than by association, whatever the solver.
`qubosel` fills that gap with a dependency-light core (NumPy, SciPy,
scikit-learn, pandas, joblib), optional quantum extras, seeded and unit-tested
solvers, a selection control under permuted labels, and a bias-subtracted MI
estimator. A label-permutation control on public small datasets illustrates
both the need and the controls.

# Functionality

- `QUBO`/`Ising` containers with a single fixed convention, property-tested.
- `build_qubo`: discretisation, plug-in MI/CMI (optional Miller–Madow
  correction), Mücke or CMI formulation; `alpha_search` (bisection on α) and
  `add_cardinality_penalty` with an `"auto"` strength that guarantees the target
  cardinality under an exact solver.
- Solvers: `BruteForceSolver`, `SimulatedAnnealingSolver` (numpy; optional
  `dwave-samplers` backend), `SimulatedBifurcationSolver` (ballistic/discrete,
  ancilla or field linear term; optional torch backend), `QAOASolver`
  (PennyLane, injectable device).
- `run_grid`/`ConsensusResult`: (solver, k, seed) grid, votes, failure records,
  oversize policies; `QUBOFeatureSelector` (scikit-learn `SelectorMixin`).
- `validation`: `bootstrap_stability`, `permutation_test`, `classical_selectors`,
  `evaluate_downstream`, `bootstrap_auc_ci`.
- `validation.permuted_label_selection` and `mi_correction="permutation"`.
- `datasets.make_clinical_synthetic`.

# Example

```python
from qubosel import QUBOFeatureSelector
from qubosel.datasets import make_clinical_synthetic
from qubosel.validation import bootstrap_stability, permutation_test
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

X, y, truth = make_clinical_synthetic(n_samples=80, random_state=0)
sel = QUBOFeatureSelector(
    ks=[2, 3, 4], solvers=["sa", "sb"], n_repeats=5, min_votes=0.25, random_state=0
).fit(X, y)
print(sel.consensus_.to_frame())
pipe = Pipeline([("select", sel), ("clf", LogisticRegression(max_iter=500))])
print(permutation_test(pipe, X, y, cv=5, n_permutations=200, random_state=0).p_value)
print(bootstrap_stability(sel, X, y, n_resamples=100, random_state=0).nogueira)
```

# Comparison with related software

| | seleqt | dimod/neal | simulated-bifurcation | qubosel |
|---|---|---|---|---|
| Mücke formulation | yes | – | – | yes |
| CMI + penalty formulation | no | example | – | yes |
| Solvers | exact | SA | SB | exact/SA/SB/QAOA |
| scikit-learn selector | no | no | no | yes |
| Consensus / seeds | no | no | no | yes |
| Null-selection control, bias-corrected MI | no | no | no | yes |
| Validation utilities | no | no | no | yes |

# Acknowledgements

The design generalises the feature-selection protocol of a vagus-nerve
stimulation response study; the clinical data are not distributed with the
package.

# References
