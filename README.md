# Surrogate gap in reinforcement learning for asset allocation

Reference implementation for the research proposal *What Additive Risk-Adjusted
Rewards Actually Optimise in Reinforcement Learning for Asset Allocation*.

**Start with `python -m mvrl.gap`.** That module tests whether the research gap
is real, without any reinforcement learning: the surrogate optimum is computed
exactly, so nothing it reports can be attributed to a learning algorithm.

## Install

```bash
pip install -r requirements.txt
```

Python 3.10+, numpy and scipy only.

---

This stage builds the ground truth: both reference policies of the proposal, the
exact evaluator for the mean-variance objective, and the first term of the error
decomposition. No reinforcement learning yet — until the reference policies exist
and are independently verified, nothing measured against them would mean anything.

## Contents

| File | Purpose |
|---|---|
| `market.py` | I.i.d. multi-period market with known first two moments |
| `policies.py` | Li–Ng pre-committed closed form; myopic policy; affine container |
| `equilibrium.py` | Equilibrium (consistent planning) policy by backward recursion |
| `evaluate.py` | Exact moment-propagation evaluator for `J`, plus Monte Carlo |
| `decomposition.py` | The error decomposition, as far as it is currently computable |
| `surrogate.py` | Exact optimum of an additive risk-adjusted reward |
| `gap.py` | **The decisive experiment: does the research gap exist?** |
| `agent.py` | Policy-gradient agent trained on the additive reward |
| `phase1.py` | **The complete three-term decomposition, measured** |
| `inconsistency.py` | Numerical demonstration of equation (8) of the proposal |
| `verify.py` | Ten independent checks on all of the above |

## Running

```bash
python -m mvrl.gap             # the decisive experiment  <-- start here
python -m mvrl.phase1          # the complete decomposition (slow, ~10 min)
python -m mvrl.verify          # full verification suite (10 checks)
python -m mvrl.inconsistency   # time-inconsistency demonstration
python -m mvrl.decomposition   # decomposition terms and sensitivity
```

## What is implemented

**Pre-committed policy** (`policies.li_ng_precommitted`). The Li–Ng embedding in
the auxiliary problem `max E[lambda x_T - phi x_T^2]`, giving a control affine in
wealth, `u*_t(x) = alpha_t x + beta_t`. The multiplier is fixed by
`lambda = 1 + 2 phi E[x_T]`; since `E[x_T]` is affine in `lambda`, this is solved
exactly from two evaluations rather than by iteration.

**Equilibrium policy** (`equilibrium.solve_equilibrium`). Consistent planning in
the sense of Strotz and Björk–Murgoci. The moment functions `f(t,x) = E[x_T]` and
`g(t,x) = E[x_T^2]` under the equilibrium continuation are affine and quadratic
respectively, which closes a backward recursion with

    D_{t+1} = B_{t+1} M - A_{t+1}^2 m m^T,   d = D_{t+1}^{-1} m
    alpha_t = s (A_{t+1}^2 - B_{t+1}) d
    beta_t  = [A_{t+1} - phi b_{t+1} + 2 phi A_{t+1} a_{t+1}] d / (2 phi)

Concavity of the one-step problem requires `D_{t+1}` positive definite; this is
checked at every step rather than assumed.

## Verification

Ten checks, all passing. The ones that carry weight:

- **Eight L-BFGS-B restarts from random points**, given no knowledge of the
  embedding, converge to the closed-form value to 2.6e-12. Independent derivation
  route, so agreement is real evidence the algebra is right.
- The gradient of `J` vanishes at the closed-form policy (1.7e-9).
- **The equilibrium condition is tested directly**: at a range of times and wealth
  levels the optimiser is asked to beat the equilibrium action with the
  continuation held fixed. Best improvement found: 0.
- Monte Carlo agrees with the exact evaluator; the embedding fixed point
  reproduces the policy to 2.7e-15; no perturbation out of 4000 improves on it.

An earlier version of the first check used Nelder–Mead, which failed to converge
in 24 dimensions and so passed vacuously by only confirming the optimiser did not
*beat* the closed form. The current version requires agreement in both directions.

## Does the research gap exist?

`gap.py` tests the proposal's central claim directly. No reinforcement learning
is involved: in this market the maximiser of an additive quadratic reward is
computed to numerical precision, so nothing below can be blamed on a learning
algorithm. Any gap that appears is a property of the reward.

**1. The most common reward in the literature has no optimum.** The plain-return
reward carries no risk term, so the additive objective rewards unbounded
leverage. Along a ray of increasing position size it grows without bound
(0.88, 8.1, 80, 801, 8010, 80098). The mean-variance problem it stands in for is
well posed and bounded; the surrogate is not. Anyone running an algorithm on this
reward is not solving a poorly-conditioned version of the right problem — they
are solving one with no answer, and what they get is whatever their leverage
limits and early stopping produce.

**2. With a risk term, the surrogate optimum is a third policy.** For the
variance-penalised reward at its best `kappa`:

    J(pi_pre)               = 1.425799294
    J(pi_eq)                = 1.323643886
    J(pi_r) at kappa*       = 1.326118461      kappa* = 0.808485

    eps_surr = J(pi_eq) - J(pi_r)  = -0.002474575
    shortfall vs pre-committed     =  0.099680834   (97.6% of eps_incons)
    distance(pi_r, pi_eq)          =  0.192068
    distance(pi_r, pi_pre)         =  3.233002

`pi_r` is not the equilibrium policy, and no `kappa` makes it so. It is nowhere
near the pre-committed policy. And `eps_surr` is **negative**, so the surrogate
is not a degraded equilibrium policy — it is a distinct third object. That is
what the proposal claims and what the literature does not address.

Note also that `kappa* = 0.808 != phi = 1.0`. Setting the reward's risk penalty
equal to one's actual risk aversion — the obvious thing to do — is not optimal,
and costs 0.0095 here.

**3. The Sharpe ratio ranks these policies in the wrong order.** This is the
sharpest result:

| policy | Sharpe | J |
|---|---|---|
| pre-committed | 1.3126 | 1.425799 |
| equilibrium | 1.1501 | 1.323644 |
| surrogate, kappa* | 1.1542 | 1.326118 |
| surrogate, kappa=phi | **1.1955** | **1.316580** |

The `kappa = phi` surrogate has a **higher** Sharpe ratio than the equilibrium
policy and a **lower** mean-variance objective. The two orderings disagree. A
study comparing these policies by reported Sharpe ratio — the standard practice
the surveys describe — would conclude the surrogate is better on precisely the
criterion it is worse on.

This is Gap 3 of the proposal, demonstrated rather than asserted. No amount of
backtesting reveals the error reward design introduces, because the reported
statistic does not order policies the way the objective does. The error has to be
measured against a known optimum, which is what the rest of this package provides.

## Phase 1 result: the complete decomposition

With `agent.py` all four policies exist, so the decomposition can be evaluated
term by term. At `phi = kappa = 1`, five seeds:

    J(pi_pre)  = 1.425799294     pre-committed optimum      exact
    J(pi_eq)   = 1.323643886     equilibrium                exact
    J(pi_r)    = 1.316579788     surrogate optimum          exact
    J(pi_hat)  = 1.307875869     trained agent              sd 0.0032

    eps_incons = +0.102155409
    eps_surr   = +0.007064097
    eps_learn  = +0.008703920
    -------------------------
    sum        = +0.117923426  = J(pi_pre) - J(pi_hat)   (closes to 0.0e+00)

The agent is trained on the additive reward alone and never sees `J`.

**The ranking of the three terms depends on the configuration.** `eps_incons`
decays as `1/phi` while `eps_surr` grows, so they cross near `phi = 12`:

| phi | eps_incons | eps_surr | eps_learn | dominant |
|---|---|---|---|---|
| 0.5 | 0.204311 | 0.010937 | 0.047494 | incons |
| 1 | 0.102155 | 0.007064 | 0.008635 | incons |
| 4 | 0.025539 | 0.004919 | 0.000573 | incons |
| 8 | 0.012769 | 0.005575 | -0.000129 | incons |
| 16 | 0.006385 | 0.007639 | 0.000053 | **surr** |
| 32 | 0.003192 | 0.012145 | -0.000346 | **surr** |
| 64 | 0.001596 | 0.021345 | -0.000901 | **surr** |

So "is reward design the problem?" has no configuration-free answer. A study
reporting a single risk aversion reports one point on this curve without saying
which, and two studies disagreeing about the importance of reward design may
simply sit on opposite sides of the crossover. `eps_learn` is negligible
throughout and occasionally negative, the agent being a stochastic optimiser of
an objective whose exact optimum is known.

## Supporting results

Three further facts came out of running the code rather than out of the proposal.

**1. Time-inconsistency is driven by deviation from the expected path.** The
committed plan and a fresh re-solve at `t = 1` disagree, and the disagreement is
affine in wealth and vanishes at exactly one point: `E[x_1]`, the wealth the
time-0 self expected, to 1.8e-15 across markets, horizons, risk aversions and
asset counts. The mechanism is in the embedding — `lambda` is pinned by `E[x_T]`,
so when realised wealth matches expectation the re-solve recovers the same
`lambda`. The reversal is therefore a property of realised paths, invisible to any
analysis working in expectation, and largest away from the mean path, which is
where a learning algorithm spends its sampling time.

**2. The two reference policies differ structurally.** The equilibrium control is
wealth-independent (`alpha_t = 0` identically, equivalently `A_t^2 = B_t`): it
holds a constant *amount* in the risky assets, where the pre-committed control
holds an affine function of wealth with negative slope. This reproduces the known
structure of the equilibrium mean-variance policy and is independent evidence the
recursion is correct. It also sharpens the proposal's point: "approximating the
mean-variance optimum" is ambiguous between two policies of different functional
form, not merely different parameters.

**3. `eps_incons` scales exactly as `1/phi` and does not depend on `x0`.**
Verified to 9e-13 and 5e-15 respectively. This matters for experimental design in
Step 3: the price of time-consistency can be made arbitrarily small relative to
the other error terms by raising `phi`, which is how the surrogate gap will be
isolated. It grows with the horizon, so a short horizon keeps it out of the way.

For the market in `decomposition.py` (3 assets, 4 periods, `phi = 1`):

    J(pi_pre)  = 1.425799294
    J(pi_eq)   = 1.323643886
    eps_incons = 0.102155409     (7.2% of J(pi_pre))

All three findings should be checked against Li & Ng (2000) and Basak &
Chabakauri (2010) before being described as new. The second is certainly known.
The first and third fall out of the embedding naturally enough that they may be
too, in which case cite rather than claim.

## Not yet implemented

- `pi_r`, the maximiser of the additive surrogate, and hence `eps_surr` and
  `eps_learn`. These are Steps 2 and 4 of the proposal.
- The augmented MDP for the differential Sharpe ratio (Step 2).
- Transaction costs; the dynamics are frictionless.

## Assumptions

Returns are i.i.d. across periods with known first two moments; no transaction
costs, no short-sale constraint, no leverage limit. `u` is an amount, not a
weight, so the simplex constraint of the proposal is not imposed here — the
pre-committed solution is unconstrained by construction and imposing the
constraint would forfeit the closed form. Reconciling the two is a design
question for Step 3, not an oversight.
