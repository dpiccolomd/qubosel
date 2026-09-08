# qubosel

**qubosel** selects features by minimising a *Quadratic Unconstrained Binary
Optimisation* (QUBO) objective built from mutual information: each binary
variable decides whether a feature is kept, the diagonal rewards relevance to
the outcome and the off-diagonal terms penalise redundancy between features.

The package provides

- two formulations, **Mücke et al. (2023)** (default, cardinality controlled by a
  bisection on the trade-off parameter α) and the **conditional-MI** formulation
  with an explicit penalty (Nguyen et al. 2014), plus any user-supplied
  QUBO builder;
- four interchangeable solvers: exhaustive **brute force**, **simulated
  annealing**, **simulated bifurcation** (ballistic/discrete, Goto et al. 2021)
  and **QAOA** (PennyLane, optional extra);
- a **consensus** layer that runs a grid of (solver, k, seed) jobs and votes;
- a **scikit-learn** selector (`QUBOFeatureSelector`) usable inside a
  `Pipeline`, so that selection can be nested in cross-validation;
- **validation** utilities: selection under shuffled labels (does the
  selector depend on the outcome at all?), bootstrap stability (Nogueira Φ,
  Jaccard), label permutation tests with the selector inside the null,
  classical baselines at the same cardinality, and fold-enclosed downstream
  evaluation with bootstrap AUC intervals.

## Positioning

qubosel is a *classical* package. Simulated annealing and simulated bifurcation
are quantum-*inspired* heuristics; QAOA runs on a state-vector simulator. No
claim of quantum advantage is made or implied. The value of the QUBO
formulation is that relevance and redundancy are optimised jointly over
feature subsets instead of greedily, and that several very different solvers
can be cross-checked on the same objective.

## Small samples

Histogram mutual information is biased upwards in proportion to the number of
distinct values of a variable. With tens of samples that bias dominates, and
an MI-based QUBO then ranks features by cardinality rather than by
association with the outcome, whatever the solver. qubosel ships the negative
control (`permuted_label_selection`) and a bias-subtracted estimator
(`mi_correction="permutation"`); see [Theory](theory.md).

## Origin

The design generalises the QUBO feature-selection protocols of the D-Wave
example lineage and of Mücke et al. (2023) to small clinical tables. The
label-independence of histogram-MI selection at small sample sizes is
demonstrated on public datasets in
[Label-permutation control on public data](public-small-n.md).
