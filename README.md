# Surrogate gap in reinforcement learning for asset allocation

Code and paper for *What Additive Risk-Adjusted Rewards Optimise in
Reinforcement Learning for Asset Allocation*.

**The problem.** The field trains RL agents on risk-adjusted per-period rewards
— a Sharpe ratio, a variance penalty — and maximises the sum. That sum is not
the mean-variance criterion it is meant to represent, and nobody has quantified
what it actually optimises. This work does.

**Start with `python -m mvrl.gap`.** It tests whether the gap is real using no
reinforcement learning at all: the surrogate optimum is computed exactly, so
nothing it reports can be blamed on a learning algorithm.

## Install

```bash
pip install -r requirements.txt
```

Python 3.10+, numpy and scipy only. See `REPRODUCE.md` for expected values to
check against.

## The main result

For the variance-penalised reward `r = d - kappa d^2` in an i.i.d. market, the
surrogate optimum recovers the mean-variance **equilibrium** policy exactly
when the gross riskless return is 1 and

    kappa* = phi (1 - nu),        nu = m^T M^-1 m

— not at `kappa = phi`, the obvious choice. Verified to 1e-16 for every horizon
and asset count tested, `nu` from 0.001 to 0.98, `phi` from 0.1 to 50.

The correction is a **discrete-time effect**: `nu = SR^2 * Delta + O(Delta^2)`,
so it vanishes as the rebalancing interval shrinks. 11% at annual rebalancing,
2.7% quarterly, under 0.25% weekly. This is also why it does not appear in the
equilibrium mean-variance literature, which works in continuous time.

When the riskless return is not 1, no `kappa` recovers the equilibrium and the
minimal attainable policy distance is linear in `s - 1`, with the constant in
closed form at `T = 2`.

## Contents

| File | Purpose |
|---|---|
| `market.py` | I.i.d. multi-period market with known first two moments |
| `policies.py` | Li–Ng pre-committed closed form; myopic policy; affine container |
| `equilibrium.py` | Equilibrium (consistent planning) policy by backward recursion |
| `evaluate.py` | Exact moment-propagation evaluator for `J`, plus Monte Carlo |
| `surrogate.py` | Surrogate optimum: exact backward recursion, and an independent optimiser |
| `agent.py` | Policy-gradient agent trained on the additive reward |
| `inconsistency.py` | Numerical demonstration that the Bellman equation fails |
| `decomposition.py` | `eps_incons` and its sensitivity |
| `gap.py` | **Does the research gap exist?** |
| `phase1.py` | The complete three-term decomposition, measured |
| `lambda_functional.py` | **The main result: the structure of Lambda** |
| `dsr.py` | The differential Sharpe ratio and its implied risk aversion |
| `phase5.py` | Out-of-model evaluation, and the statistical power of a backtest |
| `verify.py` | Ten independent checks on all of the above |

## Running

```bash
python -m mvrl.verify              # 10 checks, ~4 s  <-- run this first
python -m mvrl.gap                 # the decisive experiment, ~90 s
python -m mvrl.lambda_functional   # the main result, ~1 s
python -m mvrl.dsr                 # differential Sharpe ratio, ~3 min
python -m mvrl.phase5              # out of model, ~1 min
python -m mvrl.phase1              # full decomposition with RL, ~10 min
python -m mvrl.inconsistency       # Bellman failure, <1 s
python -m mvrl.decomposition       # eps_incons sensitivity, <1 s
```

## Results

### The gap is real

**The most common reward in the literature has no optimum.** Plain return
carries no risk term, so the additive objective rewards unbounded leverage:
0.88, 8.1, 80, 801, 8010, 80098 along a ray. The mean-variance problem it stands
in for is bounded. Anyone running an algorithm on it is solving a problem with no
answer, and gets whatever their leverage limits and stopping rule produce.

**With a risk term, the surrogate optimum is a third policy.** At the best
`kappa`: distance 0.19 from the equilibrium, 3.23 from the pre-committed, and
`eps_surr` is **negative** (-0.0025) — so it is not a degraded equilibrium
policy but a distinct object.

**The Sharpe ratio ranks policies in the wrong order.**

| policy | Sharpe | J |
|---|---|---|
| pre-committed | 1.3126 | 1.425799 |
| equilibrium | 1.1501 | 1.323644 |
| surrogate, kappa* | 1.1542 | 1.326118 |
| surrogate, kappa=phi | **1.1955** | **1.316580** |

Not a constructed example: over 519 random (market, phi) pairs the two orderings
disagree in **93.1%** of cases.

### The complete decomposition

    J(pi_pre) = 1.425799294   exact        eps_incons = +0.102155409
    J(pi_eq)  = 1.323643886   exact        eps_surr   = +0.007064097
    J(pi_r)   = 1.316579788   exact        eps_learn  = +0.008703920
    J(pi_hat) = 1.307875869   5 seeds      ---------------------------
                              sd 0.0032    sum        = +0.117923426

The identity closes to 0.0e+00. The agent is trained on the additive reward
alone and never sees `J`.

**Which term dominates depends on `phi`.** `eps_incons` decays as `1/phi` while
`eps_surr` grows, so they cross — near `phi = 12` in this market, between 4 and
10 across random markets. So "is reward design the problem?" has no
configuration-free answer, and two studies disagreeing may sit on opposite sides
of the crossover.

`eps_learn` is small and occasionally negative. This is structural, not seed
noise: it is measured in `J` while the agent optimises `J_r`, so a worse
maximiser of `J_r` can sit closer to the `J`-optimum.

### The differential Sharpe ratio is a variance penalty in disguise

Expanding the increments gives `r^DSR ~ rho - kappa_eff rho^2` with
`kappa_eff = A/(2B)`. Since `A`, `B` estimate the return's first two moments,
`kappa_eff -> E[rho]/(2 E[rho^2]) = 22.51` for this market — a quantity with **no
preference parameter in it**. Optimising the DSR and finding the closest member
of the variance-penalised family gives `kappa = 21.56` against 22.51 predicted,
residual distance 0.019.

Inverting the main result: `phi_implied = 27.9`. A practitioner choosing this
reward believes they are asking for risk-adjusted return; they are asking for
mean-variance at a risk aversion the market sets and nothing reports.

### Out of model

Under processes matched to the same first two moments:

| process | pre-committed | equilibrium | surrogate k* | surrogate k=phi |
|---|---|---|---|---|
| Gaussian | 1.4267 | 1.3236 | 1.3261 | 1.3166 |
| Student-t(4) | 1.4252 | 1.3238 | 1.3263 | 1.3166 |
| vol clustering | 1.4037 | 1.3234 | 1.3258 | 1.3163 |
| AR(1) momentum | **0.9285** | 1.1886 | 1.1865 | **1.2270** |

Fat tails and volatility clustering leave the ordering intact. **Serial
correlation reverses it**, and the pre-committed policy collapses. Every closed
form here assumes independence across periods; the results should not be read as
applying to markets with predictability. This is the most important open problem.

**A backtest gets it wrong, and more data makes it worse.** Probability that a
Sharpe comparison of `n` independent replications picks the `J`-better policy:
0.022 at `n=10`, 0.003 at 50, 0.000 at 200 and above. More data increases
confidence in the wrong answer, because it estimates the wrong quantity with
increasing precision.

## Verification

Ten checks in `verify.py`, all passing. The ones that carry weight:

- **Eight L-BFGS-B restarts from random points**, given no knowledge of the
  Li–Ng embedding, converge to the closed-form value to 2.6e-12.
- The gradient of `J` vanishes at the closed-form policy (1.7e-9).
- **The equilibrium condition is tested against its definition**: the optimiser
  is asked to beat the equilibrium action with the continuation held fixed. Best
  improvement found: 0.
- The equilibrium recursion, derived here rather than quoted, reproduces Basak &
  Chabakauri's continuous-time formula to 0.76%, 0.25%, 0.06%, 0.01%, 0.00% as
  one year is split into 4, 12, 52, 252, 2520 steps.

An earlier version of the first check used Nelder–Mead, which failed to converge
in 24 dimensions and passed vacuously by only confirming the optimiser did not
*beat* the closed form. See `WORKLOG.md` for the other twelve errors caught.

## Documents

- `paper.pdf` / `paper.tex` — the paper. **Section 9 lists what is not done.**
- `proposal.pdf` / `proposal.tex` — the original research proposal
- `FINDINGS.md` — current state of knowledge, reproduced results kept separate from new
- `WORKLOG.md` — full history including all 13 errors caught
- `REPRODUCE.md` — expected values for local verification
- `CONTRIBUTING.md` — working discipline

## Status

This is a working draft, not a submission. Outstanding, in priority order:

1. **Dependent returns.** The out-of-model result above shows the ordering
   reversing under serial correlation. Needs fixing or clear scope bounding.
2. **Error bounds for the `T = 2` constant.** Currently a series expansion with
   numerical confirmation, not a rigorous bound.
3. **The constant for `T >= 3`.** Grows superlinearly (log-log exponent 1.16),
   no closed form.
4. **A formal priority search.** `kappa* = phi(1-nu)` was not found in the
   literature and the discrete-time argument explains why it would not be, but
   that is an argument, not a proof of novelty.

## Assumptions

Returns i.i.d. across periods with known first two moments; no transaction
costs, no short-sale constraint, no leverage limit. `u` is an amount, not a
weight, so the simplex constraint is not imposed — the pre-committed solution is
unconstrained by construction and imposing the closed form would forfeit it.
