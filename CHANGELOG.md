# Changelog

## 0.1.0 (unreleased)

- Initial release: QUBO/Ising core, Mücke 2023 and CMI formulations,
  α-search, cardinality penalties, brute-force / SA / SB / QAOA solvers,
  consensus grid, scikit-learn selector, validation utilities, synthetic
  clinical dataset, exact Falcondale 0.2.4 legacy QUBO and forensic tests.
- Simulated annealing: optional cardinality-preserving swap moves with a
  data-driven cold temperature for penalty-constrained QUBOs.
- `validation.permuted_label_selection`: selection control under shuffled labels.
- `mi_correction="expected"` (exact reliable-MI correction of Mandros et al. via the
  hypergeometric closed form) and `"permutation"` (Monte-Carlo version): subtract from
  every MI term, including redundancy terms, its expectation under the permutation null.
- `QUBOFeatureSelector(formulation=callable)`: user-supplied QUBO builders.
- `formulation="rank"`, `estimator="ksg"`, `datasets.make_clinical_cohort` and
  `simulation.recovery_curve` / `standard_objectives` (ground-truth learning curve).
- `KSubsetSolver` (exact k-subset enumeration) and `examples/public_small_n.py`:
  label-permutation control on public datasets showing that the conditional-MI
  QUBO formulation is cardinality-driven at n = 31 while corrected estimators are not.
- Third-party SDK audit (objective reconstruction, forensic tests) lives in
  `audit/`, outside the package.
