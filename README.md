# qubosel

[![CI](https://github.com/dpiccolomd/qubosel/actions/workflows/ci.yml/badge.svg)](https://github.com/dpiccolomd/qubosel/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Combinatorial (QUBO-based) feature selection with quantum-inspired solvers,
consensus and built-in validation. scikit-learn compatible.

```python
from qubosel import QUBOFeatureSelector

sel = QUBOFeatureSelector(k=3, solvers=["sa", "sb"], n_repeats=5, random_state=0).fit(X, y)
sel.get_feature_names_out()
```

- **Formulations**: Mücke et al. 2023 (default; cardinality by bisection on α)
  and conditional-MI with a quadratic penalty (Nguyen et al. 2014).
- **Solvers**: exhaustive brute force, simulated annealing, simulated
  bifurcation (bSB/dSB, Goto et al. 2021), QAOA (PennyLane, `pip install qubosel[qaoa]`).
  All seed-deterministic, all cross-checked against brute force in the tests.
- **Consensus**: grid of (solver, k, seed) runs, votes with an honest
  denominator, oversize/undersize policies.
- **Validation**: selection under shuffled labels (does the selector depend on
  the outcome at all?), bootstrap stability (Nogueira Φ, Jaccard), permutation
  tests with the selector inside the null, classical baselines at the same k,
  fold-enclosed downstream evaluation with bootstrap AUC intervals.
- **Estimators**: histogram MI (plug-in, Miller-Madow, reliable-MI corrected exactly or by permutation),
  kNN MI, and a discretisation-free rank-based objective; or any callable
  `formulation(X, y, names) -> QUBO`.
- **Ground-truth simulation**: `recovery_curve` measures, on synthetic cohorts
  with known predictors, the sample size at which each objective becomes
  reliable.

qubosel is a classical package: simulated annealing and simulated bifurcation
are quantum-*inspired* heuristics and QAOA runs on a simulator. No quantum
advantage is claimed. With small samples, histogram mutual information ranks
features by their number of distinct values rather than by association; use
`mi_correction="expected"` and always check the selection against shuffled
labels (`permuted_label_selection`).

## Install

```bash
pip install qubosel            # core
pip install "qubosel[qaoa]"    # + PennyLane (Python >= 3.11)
```

## Documentation

Full documentation (theory, solvers, validation, API):
<https://dpiccolomd.github.io/qubosel>. A complete example runs in under two
minutes:

```bash
uv run python examples/synthetic_end_to_end.py
```

## Citation

See `CITATION.cff`. Key references: Mücke, Heese, Müller, Wolter & Piatkowski
(2023) *Feature selection on quantum computers*, Quantum Machine Intelligence;
Goto et al. (2021) *High-performance combinatorial optimization based on
classical mechanics*, Science Advances; Nogueira, Sechidis & Brown (2018) *On
the stability of feature selection algorithms*, JMLR.

## License

MIT.
