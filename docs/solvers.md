# Solvers

All solvers implement `solve(qubo, seed) -> SolverResult` with `x` (int8
vector), `energy` (always recomputed as `qubo.energy(x)`), `n_selected` and
`metadata`. Use `get_solver("sa")`, `get_solver(("sb", {"variant": "discrete"}))`
or pass an instance.

| name | class | default | notes |
|---|---|---|---|
| `brute_force` | `BruteForceSolver` | `max_n=22` | exact; deterministic tie-break (lowest index) |
| `sa` | `SimulatedAnnealingSolver` | 50 reads × 1000 sweeps | numpy Metropolis with geometric β schedule; `backend="dwave"` optional |
| `sb` | `SimulatedBifurcationSolver` | bSB, 128 agents × 2000 steps | Goto et al. 2021; ancilla spin for the linear term; `backend="torch"` optional |
| `qaoa` | `QAOASolver` | p=2, COBYLA, 2000 shots | PennyLane `default.qubit`; analytic optimisation, seeded sampling |

## Simulated annealing

State is a `(num_reads, n)` spin matrix with incrementally updated local
fields. Each sweep visits every spin in a fresh random order. The β range
follows the `dwave-neal` rule: $\beta_{hot} = \ln 2 / (2\max_i \sigma_i)$ with
$\sigma_i = |h_i| + \sum_j |J_{ij}|$ and $\beta_{cold} = \ln 100 / (2 \min|b|)$
over non-zero biases. The lowest-energy read is returned (all read energies are
in `metadata["read_energies"]`).

**Penalty-constrained QUBOs.** With a large cardinality penalty the single-flip
dynamics freeze the cardinality long before the objective terms become
thermally resolvable (a flip costs about λ, a difference between two valid
subsets is orders of magnitude smaller). `swap_moves=True` adds, after each
flip sweep, `n` Metropolis proposals that exchange one selected and one
unselected variable (energy change computed from the local fields), and sets
the cold temperature from a low quantile of sampled swap energy changes. Use it
whenever `cardinality="penalty"` is solved with SA; the default Mücke
formulation with α-search needs no penalty and no swap moves.

## Simulated bifurcation

The Ising problem is written in Goto's convention $E = -\tfrac12 s^\top J_G s -
h_G^\top s$ ($J_G = -J$, $h_G = -h$). The linear term is embedded through an
ancilla spin (block matrix $\begin{bmatrix} J_G & h_G \\ h_G^\top & 0\end{bmatrix}$)
and the readout is taken relative to the ancilla, or applied as an external
field (`linear_mode="field"`). Symplectic Euler steps with $dt = 0.1$:

$$y \mathrel{+}= dt\,(a(t)-1)\,x,\quad x \mathrel{+}= dt\,y,\quad y \mathrel{+}= dt\,\xi_0\,J f(x)$$

with $f(x)=x$ (ballistic) or $\mathrm{sign}(x)$ (discrete), inelastic walls
($|x|>1 \Rightarrow x=\pm1, y=0$), $a(t)$ ramping linearly from 0 to 1 over
`pressure_slope · n_steps`, $\xi_0 = 0.5/(\sqrt{N}\,\mathrm{rms}(J_G))$ computed on
the original couplings. `track_best=True` evaluates $\mathrm{sign}(x)$ every
`eval_every` steps and keeps the best configuration seen, which is more robust
than the final readout at small $N$.

## QAOA

See the module docstring of `qubosel.solvers.qaoa`. The device is injectable
through a callable `device(wires, shots, seed)` for future hardware backends.
The approximation ratio against brute force is recorded for $n \le 16$.
QAOA is never used as the α-search oracle.

## Accuracy on random instances

`tests/test_solvers.py` checks, on 30 random dense Gaussian QUBOs with
$n \in \{8, 12, 16\}$, that SA (50 reads) finds the optimum in ≥ 95 % of cases
and bSB/dSB (128 agents, 2000 steps) in ≥ 90 % with a relative gap ≤ 2 %
otherwise. Seeds make every solver deterministic.
