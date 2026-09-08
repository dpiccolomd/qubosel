# Theory

## Notation

Let $x \in \{0,1\}^d$ mark the selected features. A QUBO has energy
$E(x) = x^\top Q x + c$ with $Q$ **symmetric**; off-diagonal terms are counted
twice. `QUBO.to_upper()` gives the dimod-style matrix
$Q^{u}_{ij} = 2Q_{ij}$ ($i<j$). Spins use $x = (1-s)/2$, so $s=-1$ means
*selected* and, in QAOA, a measured $|1\rangle$ is a selected feature:
$h_i = -\tfrac12\,(Q_{ii} + \sum_{j\ne i} Q_{ij})$, $J_{ij} = Q_{ij}/2$.

## Information measures

Features are discretised into integer codes (`qubosel.information.discretize`).
The default uses quantile edges with a bin budget
$\min(n_{\text{unique}}, \max(2, \lfloor\sqrt{n}/2\rfloor))$, leaving binary and
one-hot columns intact. Mutual information and conditional mutual information
are plug-in estimates on contingency tables; `mi_correction="miller_madow"`
adds the Miller–Madow bias correction.

!!! warning "Small samples: check the selection against shuffled labels"
    Plug-in MI is biased upwards by roughly ``(cells - 1) / (2 n)`` nats. With
    tens of samples and continuous variables binned into several levels the
    bias exceeds the signal, and any MI-based QUBO then ranks features by their
    number of distinct values (Roulston 1999; Mandros, Boley & Vreeken 2020
    give the closed form ``(|V(X,Y)| - |V(X)| - |V(Y)| + 1) / 2n``). Always run
    `qubosel.validation.permuted_label_selection` and prefer
    `mi_correction="expected"` when ``n_samples`` is below a few hundred: it
    subtracts from every term its exact expectation under the permutation
    null, i.e. the *reliable mutual information* of Mandros et al. (2017,
    2020), computed with the hypergeometric closed form of Vinh, Epps & Bailey
    (2010) and applied term-wise to the relevance **and** the redundancy
    entries. ``"permutation"`` is the Monte-Carlo version of the same
    correction. [Label-permutation control on public data](public-small-n.md)
    shows public datasets on which the uncorrected optimum is identical in
    most label shuffles.

## Mücke et al. (2023) formulation (default)

With importance $I_i = I(x_i;y)$ and redundancy $R_{ij} = I(x_i;x_j)$,
$R_{ii}=0$:

$$Q_{ij}(\alpha) = R_{ij} - \alpha\,(R_{ij} + \delta_{ij} I_i), \qquad \alpha\in[0,1].$$

There is no cardinality constraint. Proposition 1 of the paper states that
the size of the optimal subset is non-decreasing in α, so for every $k$ there
is an α whose optimum has exactly $k$ features. `alpha_search` implements
Algorithm 1 (bisection, $a=0$, $b=1$; if the optimum has more than $k$
features set $b=\alpha$, otherwise $a=\alpha$). Features with $\alpha I_i <
\varepsilon$ ($\varepsilon = 10^{-8}$) receive a diagonal $\mu = \max Q > 0$ and
are never selected.

The oracle inside the bisection is exhaustive search for $d \le 20$ and the
first non-QAOA solver (best of three seeds) otherwise. If no exact hit occurs
the closest α is returned with `status="inexact"`; the selector then warns.

## Estimators without histograms

`estimator="ksg"` replaces plug-in MI by the kNN estimator of Kraskov et al.
(scikit-learn's implementation, averaged over a few seeds because it jitters
continuous columns). `formulation="rank"` keeps the Mücke structure but uses
``2 |AUC_i - 0.5|`` as relevance (a rank statistic, binary outcome) and the
squared Spearman correlation as redundancy: no bins, no cardinality bias.
The [ground-truth simulation](simulation.md) compares them.

## Conditional-MI formulations with penalty

`formulation="cmi"` (D-Wave example): $Q_{ii} = -I(x_i;y)$ and, per unordered
pair, $Q^{u}_{ij} = -[\,I(x_j;y\mid x_i) + I(x_i;y\mid x_j)\,]$.
A custom objective can be supplied as a callable ``formulation(X, y, names) -> QUBO``
(this is the form of the D-Wave example lineage; see `formulation="cmi_cr"` for the
class-conditional-redundancy variant).
Cardinality is imposed with $\lambda(\sum_i x_i - k)^2$:

- `lam="auto"`: $\lambda = 1 + \sum_i |Q_{ii}| + \sum_{i<j} 2|Q_{ij}|$ bounds the
  energy range of the unpenalised objective, so every violation costs more than
  any possible gain and an exact solver returns exactly $k$ features;
- `lam="legacy"`: $\lambda = 10k$ (a soft multiplicative rule used by some SDKs). This is a *soft*
  constraint: `tests/test_formulations.py::test_legacy_penalty_is_soft` gives a
  QUBO where the optimum violates $k$. On the study data the legacy value was
  large enough.

`normalize=True` rescales the penalised QUBO to $\max|Q^u| = 1$ as
`dimod.BinaryQuadraticModel.normalize()` does; it changes nothing for exact
solvers but affects heuristics' temperature scales.

## Sizes of subsets in Ising form

The energy of the Ising form equals the QUBO energy for every assignment
(`tests/test_qubo.py::test_qubo_ising_identity`, property-based), and the QAOA
cost Hamiltonian evaluated on a basis state equals the QUBO energy minus the
offset (`tests/qaoa/test_qaoa.py`).
