# Consensus

`run_grid(qubo_for_k, solvers, ks, n_repeats, random_state, ...)` runs every
(solver, k, repeat) combination and returns a `ConsensusResult` with

- `selection_matrix` (successful runs × features), `votes`, `support`;
- `n_attempted` and `n_ok`: the vote denominator is the number of *attempted*
  runs, so a failed solver lowers the frequencies instead of silently shrinking
  the denominator (the original study hard-coded 12);
- `records`: one `RunRecord` per run with status `ok`, `failed`, `oversize` or
  `undersize`, the seed, energy and solver metadata; failures also emit a
  `RuntimeWarning`.

Seeds are drawn once in a fixed order from `random_state`, so results are
identical for any `n_jobs`.

`min_votes` is an absolute count or a fraction of attempted runs.

## Cardinality policies

With the Mücke formulation the solver works on the unconstrained α-QUBO and a
heuristic may return more or fewer than `k` features:

- `oversize_policy="trim"` (default) removes features greedily by the best
  marginal energy gain on the base QUBO until `k` remain (deterministic);
- `"keep"` counts the run as is (the protocol of the D-Wave example lineage);
- `"flag"` excludes the run from the vote and keeps its record;
- `undersize_policy="keep"|"flag"` analogously.
