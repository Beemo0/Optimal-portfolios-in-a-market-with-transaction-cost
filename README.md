# Optimal Portfolios in a Market with Transaction Costs

**ING3 CY Tech 2025–2026 | M222 - Dauphine 2026–2027**

**Corentin Stephan**

## Abstract

An investor who trades continuously in a frictionless market can maintain
the constant risky-asset proportion prescribed by the classical Merton
solution. Once proportional transaction costs are introduced, continuous
rebalancing is no longer optimal: every trade is costly, and the optimal
strategy becomes characterized by a **no-transaction region**, inside which
the investor leaves the portfolio unchanged and trades only when the
portfolio reaches a boundary.

This project studies optimal portfolio choice under proportional transaction
costs from both theoretical and numerical perspectives. The thesis reviews
the discrete-time and continuous-time literature, with particular emphasis
on two questions: whether the duality methods of the frictionless problem
extend to markets with transaction costs, and how optimal trading policies
depend on the frequency at which an investor can rebalance.

The theoretical part covers:

* the frictionless Merton benchmark;
* discrete-time markets with proportional transaction costs;
* continuous-time stochastic control and the associated
  variational inequality;
* duality methods and shadow prices;
* small-transaction-cost asymptotics;
* finite-frequency rebalancing;
* utility maximisation under additional convex portfolio constraints.

The numerical part solves a finite-horizon **CARA utility** problem using a
recombining binomial tree and a policy-iteration / Gauss-Seidel scheme.
The implementation exploits the CARA translation invariance to remove the
cash dimension and reduces the problem to a dynamic optimisation over stock
holdings.

The numerical solver is validated against:

1. a closed-form binomial-tree calibration;
2. an independently written reference implementation;
3. a two-period brute-force enumeration;
4. convergence checks in the time and portfolio grids;
5. explicit sub-grid interpolation of the free boundaries.

The main numerical experiment studies the scaling of the width of the
no-transaction region as transaction costs become small. For horizons
$T \geq 3$, the fitted exponent stabilises around **0.349**, close to but
persistently above the theoretical $1/3$ scaling derived for a related
log-utility, infinite-horizon problem. The discrepancy is analysed
numerically but not resolved.

The same numerical machinery is also used to compute **utility-indifference
prices for a European contingent claim**, showing numerically how
transaction costs destroy perfect replication and create a spread between
buyer and seller indifference prices.

The repository therefore provides a reproducible implementation of the main
numerical results of the thesis, together with the diagnostics used to
validate the numerical scheme.

## Overview

This repository implements the theoretical and numerical pipeline for
portfolio optimisation with proportional transaction costs.

The project is organised around four main themes:

* **Frictionless benchmark:** the Merton constant-mix portfolio and the
  corresponding utility optimisation problem.
* **Transaction-cost model:** bid-ask spreads, solvency, singular stochastic
  control and the emergence of a no-transaction region.
* **Numerical resolution:** finite-horizon CARA utility, dimension reduction,
  binomial-tree discretisation and policy iteration.
* **Applications and diagnostics:** small-cost asymptotics, convergence
  analysis, utility-indifference pricing and independent numerical
  verification.

## Repository Structure

```text
├── figures/
│   └── # Generated figures and numerical results
│
├── notebooks/
│   └── 01_solver_validation_and_figures.ipynb
│       # Main numerical experiments, validation and figure generation
│
├── src/
│   └── txcost/
│       ├── __init__.py
│       ├── asymptotics.py       # Small-cost scaling and asymptotic diagnostics
│       ├── cara_solver.py       # Finite-horizon CARA policy-iteration solver
│       ├── diagnostics.py       # Numerical validation and convergence diagnostics
│       ├── frictionless.py      # Frictionless benchmark and closed-form quantities
│       └── tree.py              # Recombining binomial-tree construction
│
├── tests/
│   ├── test_cara_solver.py      # Tests for the transaction-cost solver
│   ├── test_diagnostics.py      # Numerical diagnostics and validation tests
│   ├── test_frictionless.py     # Frictionless benchmark tests
│   └── test_tree.py             # Binomial-tree construction tests
│
├── pyproject.toml
└── README.md
```

The repository currently contains one consolidated research notebook,
`01_solver_validation_and_figures.ipynb`, which combines the solver
validation, numerical experiments and figure generation.


# Mathematical Framework

## 1. Frictionless benchmark

Consider a market with one riskless asset earning rate $r$ and one risky
asset satisfying

$dS_t = S_t(\mu\,dt+\sigma\,dW_t).$

In the frictionless Merton problem, the investor can continuously rebalance
the portfolio. For CRRA utility with relative risk aversion $\gamma$, the
optimal risky proportion is

$\pi^* = \frac{\mu-r}{\gamma\sigma^2}$

This constant target is the benchmark against which the transaction-cost
solution is compared.

The key difficulty is that continuously maintaining $\pi^*$ requires
infinitely frequent trading. With any strictly positive proportional
transaction cost, the resulting turnover is no longer admissible as an
optimal strategy.



## 2. Proportional transaction costs

The risky asset is traded through an asymmetric bid-ask spread.

If $\lambda$ denotes the proportional buying cost and $\nu$ the
proportional selling cost, buying one unit at price $S_t$ costs

$(1+\lambda)S_t,$

while selling one unit yields

$(1-\nu)S_t.$

The investor's state is described by:

* $X_t$: cash holding;
* $Y_t$: number of shares;
* $S_t$: risky-asset price.

Trading is represented by two non-decreasing processes:

$L_t = \text{cumulative purchases},\qquad M_t = \text{cumulative sales}$

The portfolio dynamics are

$dX_t = rX_t\,dt-c_t\,dt -(1+\lambda)dL_t +(1-\nu)dM_t,$

and

$dY_t = Y_t\frac{dS_t}{S_t} + dL_t-dM_t$

The liquidation value of the risky position is

$\ell(y,S) = yS -\nu Sy^+ -\lambda Sy^-$

The transaction-cost problem is therefore a **singular stochastic control
problem** rather than an ordinary portfolio optimisation problem.



## 3. No-transaction region

The optimal policy is not a single target portfolio.

Instead, there exist two boundaries:

$y_minus(t,S) <= y_plus(t,S)$

such that:

- below y_minus(t,S), the investor buys;
- between y_minus(t,S) and y_plus(t,S), the investor does nothing;
- above y_plus(t,S), the investor sells.

The region between y_minus(t,S) and y_plus(t,S) is the **no-transaction region**.

As transaction costs decrease, the region shrinks towards the frictionless Merton target.



# Numerical Formulation

## 4. CARA dimension reduction

The general infinite-horizon CRRA formulation is difficult to discretise
directly because the state space contains both cash and risky holdings.

The numerical implementation therefore considers a finite-horizon problem
with **CARA utility**

$U(x)=-e^{-\gamma x}$

For terminal liquidation value

$X_T+\ell(Y_T,S_T)-\phi(S_T)$

the objective is

$J(t,x,y,S) = \sup_{(L,M)} \mathbb{E} \left[ -e^{-\gamma(X_T+\ell(Y_T,S_T)-\phi(S_T))} \mid X_t=x,Y_t=y,S_t=S \right]$

CARA translation invariance gives the factorisation

$J(t,x,y,S) = -e^{-\gamma x e^{r(T-t)}}Q(t,y,S)$

The cash variable $x$ therefore disappears from the reduced problem.

The numerical solver only needs to track:

* time $t$;
* stock price $S$;
* number of shares $y$.

This reduces the free-boundary problem to a pair of curves in the
$(t,S)$ state space.


# Core Modules

## `src/txcost/frictionless.py`

Provides the frictionless benchmark used throughout the numerical
experiments.

The module contains the analytical quantities needed to compute:

* the frictionless Merton target;
* benchmark portfolio values;
* frictionless terminal quantities;
* comparisons between frictionless and transaction-cost policies.

The benchmark is used both as a theoretical reference and as a diagnostic
for the numerical solver.


## `src/txcost/tree.py`

Constructs the recombining binomial tree used to discretise the risky asset.

For a time step

$\Delta t = \frac{T}{N}$

the tree uses

$u=e^{\sigma\sqrt{\Delta t}}$

$d=e^{-\sigma\sqrt{\Delta t}}$

with transition probability

$p = \frac{e^{\mu\Delta t}-d}{u-d}$

Importantly, the probability is the **physical probability**, rather than
the risk-neutral probability, because the numerical object being computed
is an expected-utility value rather than an arbitrage price.


## `src/txcost/cara_solver.py`

This is the central numerical module of the project.

It implements the finite-horizon CARA transaction-cost problem using:

* a recombining binomial tree for $S$;
* a uniform grid for stock holdings $Y$;
* backward induction;
* local buy / sell / no-trade optimisation;
* Gauss-Seidel policy iteration;
* convergence tolerances;
* extraction of the free boundaries.

At maturity,

$Q_N(j,k) = \exp\left[-\gamma\left(\ell(y_k,S_j)-\phi(S_j)\right)\right]$

At an earlier node, three candidates are compared.

### No-trade candidate

$Q_n^N(j,k) = pQ_{n+1}(j+1,k) + (1-p)Q_{n+1}(j,k)$

### Buy candidate

Buying moves the portfolio from $y_k$ to $y_{k+1}$ and introduces the
corresponding transaction cost.

### Sell candidate

Selling moves the portfolio from $y_k$ to $y_{k-1}$ and introduces the
corresponding transaction cost.

The recursion takes the minimum of the three candidates because $Q$ is
defined through the negative exponential utility representation.

Because buy and sell decisions refer to neighbouring values of the current
time slice, the resulting local problem is a fixed point. It is solved by
alternating Gauss-Seidel sweeps until convergence.

This is the discrete finite-grid analogue of the policy-iteration approach
used for free-boundary problems with transaction costs.


## `src/txcost/asymptotics.py`

Contains the diagnostics used to study the small-cost behaviour of the
no-transaction region.

The theoretical asymptotic balance is

$h^*\propto\lambda^{1/3}$

where $h$ denotes the width of the no-transaction region.

The numerical implementation estimates the empirical relationship

$\text{width}(\lambda)\approx C\lambda^\beta$

through a log-log regression.

The exponent $\beta$ is then compared with the theoretical benchmark

$\beta=\frac13$

The module also supports the analysis of the dependence of the fitted
exponent on the investment horizon $T$.


## `src/txcost/diagnostics.py`

Contains the independent numerical checks used to establish that the
observed results are not numerical artefacts.

The diagnostics include:

* tree-refinement checks;
* share-grid refinement;
* grid half-width convergence;
* policy-iteration convergence;
* comparison with closed-form tree quantities;
* boundary interpolation;
* brute-force validation in the two-period case.

A particularly important diagnostic is the **sub-grid interpolation of
the free boundaries**.

Simply reading the boundary from the closest share-grid point introduces
quantisation error. This error becomes especially important for small
transaction costs because the no-transaction region itself becomes narrow.

The implementation therefore interpolates the zero of the difference between
the relevant trading candidate and the no-trade candidate to obtain a
continuous estimate of the boundary.


# Notebook

## `01_solver_validation_and_figures.ipynb`

The main research notebook reproduces the numerical experiments and figures
of the thesis.

It covers:

1. construction of the binomial tree;
2. validation of the frictionless benchmark;
3. execution of the CARA transaction-cost solver;
4. convergence checks;
5. no-transaction-region extraction;
6. sub-grid boundary interpolation;
7. small-cost scaling regressions;
8. horizon dependence;
9. boundary-layer effects near maturity;
10. utility-indifference pricing of a European call;
11. two-period brute-force verification;
12. generation of the figures used in the thesis.

The notebook is intentionally kept as the main entry point for reproducing
the numerical results, while the reusable numerical routines live in
`src/txcost/`.


# Numerical Results

## 1. Baseline calibration

The main numerical experiments use

$\mu=0.08,\qquad r=0.02,\qquad \sigma=0.20,\qquad \gamma=1$

with symmetric transaction costs

$\lambda=\nu$

At $S_0=1$, the frictionless target is therefore

$y^* = \frac{\mu-r}{\gamma\sigma^2} = 1.5$

The transaction-cost solution produces a no-transaction interval centred
close to this target.

For $T=1$, the interpolated widths obtained for

$\lambda \in \{0.0005,0.001,0.002,0.004,0.008\}$

are approximately

$0.222,\quad 0.283,\quad 0.361,\quad 0.464,\quad 0.645$

The width increases monotonically with transaction costs, as predicted by
the small-cost theory.


## 2. Small-cost scaling

A log-log regression of the no-transaction-region width against the
transaction-cost parameter gives:

| Horizon ($T$) | Number of tree steps ($N$) | Fitted exponent |
| ----------: | -----------------------: | --------------: |
|           1 |                      150 |           0.380 |
|           3 |                      450 |           0.348 |
|           6 |                      900 |           0.350 |
|          10 |                     1500 |           0.348 |

The main finding is therefore a rapid decline from

$0.380$

at $T=1$ to approximately $0.349$ for $T\geq3$, followed by an apparent
plateau.

The theoretical $1/3$ exponent comes from the small-cost asymptotics of a
related problem involving logarithmic utility and an infinite horizon.
The numerical solver here instead solves a finite-horizon CARA problem with
terminal liquidation.

Consequently, the numerical experiment is **consistent with the qualitative
$1/3$-type small-cost scaling without reproducing the exact theoretical
exponent**.

The remaining discrepancy is approximately 5% for the longest horizons
tested.

## 3. Boundary-layer effect

The finite-horizon formulation creates a visible boundary effect near
maturity.

For example, at

$\lambda=0.01$

the lower no-transaction boundary is pulled towards

$y=0$

as $t\rightarrow T$.

This behaviour comes from the terminal liquidation condition and has no
direct counterpart in the infinite-horizon formulation used by the closed
form asymptotic results.

This provides a numerical explanation for part of the discrepancy observed
at short horizons.


# Utility-Indifference Pricing

The same solver can be used to value a European contingent claim under
transaction costs.

For a European call,

$\phi(S_T)=(S_T-K)^+$

the seller's indifference price is

$p_s = \frac{1}{\gamma e^{rT}}\log\left(\frac{Q(0,0,S_0;\phi)}{Q(0,0,S_0;0)}\right)$

while the buyer's price is obtained by solving the corresponding problem
with $-\phi$.

For

$K=S_0=1,\qquad T=1$

the frictionless Black-Scholes price is approximately

$0.0892$

The numerical results are:

| $\lambda=\nu$ | Seller price | Buyer price | Spread |
| ------------: | -----------: | ----------: | -----: |
|             0 |       0.0890 |      0.0890 | 0.0000 |
|         0.001 |       0.0901 |      0.0893 | 0.0008 |
|         0.002 |       0.0912 |      0.0901 | 0.0011 |
|         0.005 |       0.0949 |      0.0929 | 0.0020 |
|         0.010 |       0.1013 |      0.0986 | 0.0027 |
|         0.020 |       0.1152 |      0.1108 | 0.0044 |

As transaction costs increase, the buyer and seller indifference prices
separate.

This provides a numerical illustration of the loss of perfect replication
in a market with transaction costs.

An interesting feature of this calibration is that both indifference prices
lie above the frictionless Black-Scholes price for positive transaction
costs. The thesis reports this as an empirical observation rather than as a
general theorem; the investor already has a positive demand for the risky
asset in the chosen calibration, which may interact with the cost of
hedging.


# Two-Period Independent Verification

A particularly strong validation of the numerical implementation is provided
by a two-period binomial model.

For

$\mu=0.08,\quad r=0.02,\quad \sigma=0.20,\quad \gamma=1,\quad \lambda=\nu=0.02$

with

$S_0=1,\qquad T=1$

a direct brute-force enumeration gives

$Q(0,0,S_0) = 0.994534817$

with optimal first-period holding approximately

$y_1^* \approx 0.521$

The recursive solver produces the same value to every digit shown after
refinement of the share grid and policy-iteration tolerance.

This check is important because it validates the general recursive algorithm
against an independently constructed optimisation problem for which
exhaustive enumeration is computationally feasible.


# Key Results

| Experiment                       | Result                                                                                                               |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Frictionless benchmark**       | The numerical framework reproduces the Merton risky-asset target ($y^*=1.5$) for the main calibration.                 |
| **No-transaction region**        | The optimal policy becomes a band rather than a single target; the band widens monotonically with transaction costs. |
| **Small-cost scaling, $T=1$**    | Log-log regression gives an exponent of **0.380**.                                                                   |
| **Small-cost scaling, $T\geq3$** | The fitted exponent stabilises around **0.349**, approximately 5% above $1/3$.                                       |
| **Boundary interpolation**       | Sub-grid interpolation removes the grid-snapping bias that affected earlier scaling estimates.                       |
| **Horizon effect**               | The exponent decreases from 0.380 at $T=1$ to approximately 0.349 at $T=3$, then plateaus through $T=10$.            |
| **Numerical convergence**        | Results are stable under the reported tree, share-grid, Gauss-Seidel and grid-width convergence checks.              |
| **Two-period brute force**       | Recursive solver agrees with independent enumeration: $Q(0,0,S_0)=0.994534817$.                                      |
| **Indifference pricing**         | Buyer/seller prices converge to Black-Scholes as transaction costs vanish and separate as costs increase.            |
| **Replication**                  | Positive transaction costs create a non-zero buyer/seller indifference-price spread.                                 |
| **Open question**                | The numerical exponent does not reach the theoretical $1/3$ within the computationally accessible horizons.          |


# Validation Strategy

The numerical results are not based on a single implementation or a single
convergence test.

The solver is checked through several independent mechanisms.

### 1. Closed-form tree calibration

Special cases for which the binomial-tree quantities are known analytically
are used to validate the tree construction and transition probabilities.

### 2. Independent reference implementation

The main numerical recursion is compared against an independently written
reference implementation to detect implementation-level errors.

### 3. Two-period brute force

For $N=2$, the dynamic-programming recursion is compared against direct
enumeration over the possible holdings.

### 4. Time-grid convergence

The number of binomial steps is increased while monitoring the stability of
the estimated boundaries and scaling exponents.

### 5. Share-grid convergence

The portfolio grid spacing $\Delta y$ is refined to ensure that the
numerical free boundaries are not grid-dependent.

### 6. Grid-width convergence

The half-width $M$ of the share grid is increased to verify that the
optimiser does not interact with an artificial grid boundary.

### 7. Sub-grid boundary interpolation

The free boundaries are interpolated between grid points instead of being
identified with discrete grid indices.

This last correction is particularly important for the small-cost scaling
experiment, because the smallest transaction costs correspond to the narrowest
no-transaction regions and therefore suffer the largest relative
grid-quantisation error.


# Important Implementation Detail

The binomial-tree recursion is performed under the **physical probability**.

This is deliberate.

The reduced value function $Q$ represents an expected-utility quantity,

$Q = \mathbb{E}[\text{terminal utility transformation}]$

rather than a risk-neutral derivative price.

Consequently, the tree probability is

$p = \frac{e^{\mu\Delta t}-d}{u-d}$

rather than the standard risk-neutral probability used for Black-Scholes
option pricing.

The utility-indifference pricing experiment still uses the same physical
measure because it is an expected-utility valuation problem rather than a
replication-based risk-neutral pricing problem.


# Limitations and Open Questions

The numerical results leave two main questions open.

## CARA finite horizon versus logarithmic infinite horizon

The theoretical $1/3$ scaling benchmark used for comparison comes from a
different problem:

* logarithmic utility;
* infinite horizon;
* no terminal liquidation effect;
* normalisation $r=0$.

The numerical solver instead considers:

* CARA utility;
* finite horizon;
* terminal liquidation;
* $r=0.02$.

The observed plateau around $0.349$ could therefore either represent the
true asymptotic behaviour of the CARA finite-horizon formulation or simply a
pre-asymptotic regime that would eventually converge to $1/3$ for horizons
much larger than $T=10$.

Resolving this would require either substantially longer numerical horizons
or a dedicated small-cost asymptotic analysis of the finite-horizon CARA
problem.

## Transaction costs combined with convex portfolio constraints

The thesis also formulates the problem with an additional convex constraint

$\pi_t\in K$

where $K$ may represent, for example:

* a leverage limit;
* a no-short-sale constraint;
* a position limit;
* a sector or concentration constraint.

The interaction between the no-transaction region and such constraints is
formulated theoretically but is not implemented in the current numerical
solver.

This is left as a natural extension of the project.

# Requirements

The project requires:

* Python `>= 3.10`
* `numpy >= 1.24`
* `scipy >= 1.10`

Development and notebook dependencies:

* `pytest >= 7.0`
* `matplotlib >= 3.7`
* `jupyter`

These requirements correspond to the current project configuration.


# Installation

Clone the repository:

```bash
git clone https://github.com/Beemo0/Optimal-portfolios-in-a-market-with-transaction-cost.git
cd Optimal-portfolios-in-a-market-with-transaction-cost
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the package in editable mode with development and notebook
dependencies:

```bash
pip install -e ".[dev]"
```

---

# Running the Tests

Run the complete test suite with:

```bash
python -m pytest tests/ -v
```

The test suite covers:

```text
tests/
├── test_cara_solver.py
├── test_diagnostics.py
├── test_frictionless.py
└── test_tree.py
```

The current repository exposes four dedicated test modules covering the
solver, diagnostics, frictionless benchmark and binomial-tree construction.



# Running the Notebook

Start Jupyter from the project root:

```bash
jupyter notebook
```

Then open:

```text
notebooks/01_solver_validation_and_figures.ipynb
```

The notebook contains the validation experiments and the main figures
associated with the numerical section of the thesis.

# Reproducibility

The numerical workflow is designed around the following pipeline:

```text
Market parameters
       │
       ▼
Binomial tree construction
       │
       ▼
Terminal liquidation value
       │
       ▼
CARA dimension reduction
       │
       ▼
Backward induction
       │
       ▼
Buy / Sell / No-trade policy iteration
       │
       ▼
Free-boundary extraction
       │
       ▼
Sub-grid interpolation
       │
       ▼
Convergence diagnostics
       │
       ├───────────────┐
       ▼               ▼
Small-cost scaling   Indifference pricing
       │               │
       ▼               ▼
Figures / tables / thesis results
```

The implementation is intentionally separated into reusable source modules,
tests and a research notebook so that the numerical results can be checked
independently of the presentation layer.

# References

- Merton, R. C. (1969). *Lifetime Portfolio Selection under Uncertainty:
  The Continuous-Time Case*. Review of Economics and Statistics, 51(3),
  247–257.
- Jouini, E. & Kallal, H. (1995). *Martingales and Arbitrage in Securities
  Markets with Transaction Costs*. Journal of Economic Theory, 66, 178–197.
- Davis, M. H. A. & Norman, A. R. (1990). *Portfolio Selection with
  Transaction Costs*. Mathematics of Operations Research, 15(4), 676–713.
- Shreve, S. E. & Soner, H. M. (1994). *Optimal Investment and Consumption
  with Transaction Costs*. Annals of Applied Probability, 4(3), 609–692.
- Cvitanić, J. & Karatzas, I. (1996). *Hedging and Portfolio Optimization
  under Transaction Costs: A Martingale Approach*. Mathematical Finance,
  6(2), 133–165.
- Kallsen, J. & Muhle-Karbe, J. (2010). *On Using Shadow Prices in Portfolio
  Optimization with Transaction Costs*. Annals of Applied Probability,
  20(4), 1341–1358.
- Gerhold, S., Muhle-Karbe, J. & Schachermayer, W. (2013). *The Dual
  Optimizer for the Growth-Optimal Portfolio under Transaction Costs*.
  Finance and Stochastics, 17(2), 325–354.
- Benedetti, G., Campi, L., Kallsen, J. & Muhle-Karbe, J. (2013).
  *On the Existence of Shadow Prices*. Finance and Stochastics, 17(4),
  801–818.
- Soner, H. M. & Touzi, N. (2013). *Homogenization and Asymptotics for
  Small Transaction Costs*. SIAM Journal on Control and Optimization,
  51(4), 2893–2921.
- Possamaï, D., Soner, H. M. & Touzi, N. (2015). *Homogenization and
  Asymptotics for Small Transaction Costs: The Multidimensional Case*.
  Communications in Partial Differential Equations, 40(11), 2005–2046.
- Bichuch, M. (2012). *Asymptotic Analysis for Optimal Investment in Finite
  Time with Transaction Costs*. SIAM Journal on Financial Mathematics,
  3(1), 433–458.
- Janeček, K. & Shreve, S. E. (2004). *Asymptotic Analysis for Optimal
  Investment and Consumption with Transaction Costs*. Finance and
  Stochastics, 8(2), 181–206.
- Davis, M. H. A., Panas, V. G. & Zariphopoulou, T. (1993). *European
  Option Pricing with Transaction Costs*. SIAM Journal on Control and
  Optimization, 31(2), 470–493.
- Cox, J. C., Ross, S. A. & Rubinstein, M. (1979). *Option Pricing:
  A Simplified Approach*. Journal of Financial Economics, 7(3), 229–263.
