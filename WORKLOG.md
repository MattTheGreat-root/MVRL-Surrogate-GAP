# Work log

Everything done from the start of implementation, in order, including the
mistakes. Written so that a reader can reconstruct why each piece exists and
which claims have been checked against what.

---

## Stage 1 — Ground truth

**Goal.** Nothing downstream can be measured until the reference optimum exists
and is known to be correct.

**Built.**

- `market.py` — i.i.d. multi-period market with specified first two moments.
  Exposes `nu = m^T M^-1 m`, the invariant everything later depends on.
- `policies.py` — the Li–Ng pre-committed optimum. The embedding multiplier
  satisfies `lambda = 1 + 2 phi E[x_T]`; since `E[x_T]` is affine in `lambda`,
  this fixed point is solved exactly from two evaluations rather than iterated.
- `evaluate.py` — two evaluators for `J = E[x_T] - phi Var(x_T)`: exact moment
  propagation for affine policies, and Monte Carlo for anything. Having both
  matters, because the Monte Carlo one is validated against the exact one before
  being trusted on learned policies.
- `verify.py` — the check suite.

**Mistake caught here.** The first version of the "is the closed form optimal?"
check used Nelder–Mead, which failed to converge in 24 dimensions and therefore
passed *vacuously*: it only confirmed the optimiser did not **beat** the closed
form. Replaced with eight L-BFGS-B restarts requiring agreement in **both**
directions. They now converge to 2.6e-12 with spread 7e-12, and the gradient of
`J` vanishes at the closed-form policy (1.7e-9).

**Finding, unplanned.** `inconsistency.py` compares the committed plan at `t=0`
against a fresh re-solve at `t=1`. The disagreement is affine in wealth and
vanishes at exactly one point — `E[x_1]`, the wealth the earlier self expected,
to 1.8e-15 across markets, horizons, risk aversions and asset counts. Mechanism:
`lambda` is pinned by `E[x_T]`, so when realised wealth matches expectation the
re-solve recovers the same `lambda`.

**Mistake caught here.** The first write-up of that finding claimed the reversal
was "present on the expected path, not only in the tails". The opposite is true.
Corrected before it went anywhere.

---

## Stage 2 — The second reference policy

**Built.** `equilibrium.py` — the consistent-planning (Björk–Murgoci) policy by
backward recursion. The moment functions `f = E[x_T]` and `g = E[x_T^2]` under
the equilibrium continuation are affine and quadratic, which closes a recursion
in five coefficients.

**Verified against the definition directly**: at a range of times and wealth
levels, L-BFGS-B is asked to beat the equilibrium action with the continuation
held fixed. Best improvement found: 0.

**Structural result.** `alpha_t = 0` identically, equivalently `A_t^2 = B_t`.
The equilibrium control holds a constant *amount*, not a constant fraction.
This reproduces the known structure (Basak & Chabakauri) and is the main
independent evidence the recursion is right.

**Result.** `eps_incons = J(pi_pre) - J(pi_eq) = 0.102155409` for the demo
market, 7.2% of `J(pi_pre)`. Also: `eps_incons` scales exactly as `1/phi` and is
independent of `x0` (9e-13, 5e-15). Practically useful — it means the
time-consistency price can be dialled small relative to the other terms.

---

## Stage 3 — Is the research gap real?

**The question that forced this.** Everything to this point reproduced known
results. The proposal's contribution rested on `eps_surr != 0`, which had not
been shown.

**Built.** `surrogate.py`, `gap.py`. The key realisation: in this setting the
maximiser of an additive quadratic reward can be computed **exactly**, with no
reinforcement learning. Any gap found is therefore a property of the reward, not
of a learning algorithm — which is precisely the separation the decomposition
claims can be made.

**Three results.**

1. **The plain-return reward has no optimum.** No risk term means unbounded
   leverage: 0.88, 8.1, 80, 801, 8010, 80098 along a ray. The mean-variance
   problem it stands in for is bounded. Bai et al. report this as the most
   common reward in the surveyed literature.
2. **With a risk term, `pi_r` is a third policy.** Distance 0.19 from the
   equilibrium at the best `kappa`, 3.23 from the pre-committed.
3. **The Sharpe ratio ranks policies in the wrong order.** The `kappa = phi`
   surrogate has Sharpe 1.1955 against the equilibrium's 1.1501 and objective
   1.3166 against 1.3236.

**Correction forced on the proposal.** `eps_surr` is **not sign-definite** — at
`kappa* = 0.808` it is `-0.0025`, the surrogate beating the equilibrium. So
"eps_surr = 0 if and only if" is false, and Theorem 1 had to be restated for the
policy distance.

---

## Stage 4 — Reinforcement learning, and the complete decomposition

**Built.** `agent.py` — affine-Gaussian policy trained by REINFORCE on the
additive reward. The policy class *contains* `pi_r` deliberately; otherwise
`eps_learn` would confound optimisation with approximation error. The agent
never sees `J`.

**Mistake caught here.** The first version peaked at iteration 500 then
degraded. Cause: the Gaussian score carries `1/sigma^2`, so gradient variance
diverges as exploration anneals. Fixed by absorbing that factor into the
learning rate, which is the Fisher preconditioning for this policy class.

**Result.** `phase1.py`, the complete decomposition:

    J(pi_pre) = 1.425799294   J(pi_eq) = 1.323643886
    J(pi_r)   = 1.316579788   J(pi_hat) = 1.307875869  (5 seeds, sd 0.0032)

    eps_incons = +0.102155409
    eps_surr   = +0.007064097
    eps_learn  = +0.008703920
    sum        = +0.117923426  = J(pi_pre) - J(pi_hat), closes to 0.0e+00

**Research result.** Which term dominates depends on `phi`, with a crossover:
`eps_incons` decays as `1/phi` while `eps_surr` grows. So "is reward design the
problem?" has no configuration-free answer, and two studies disagreeing may sit
on opposite sides of the crossover.

---

## Stage 5 — Lambda

**The actual research.** Theorem 1 of the proposal needed a functional `Lambda`;
it had none.

**Method.** Symbolic derivation with sympy, verified numerically at every step.

**R1. The surrogate optimum satisfies a backward recursion.** The observation
that unlocked everything: the additive objective is a *sum*, hence
time-consistent, hence subject to dynamic programming. With `d = M^-1 m`,
`g_t = kappa(s-1) - p_{t+1} s`:

    alpha_t = -d g_t / (kappa - p_{t+1})
    p_t     = [kappa p_{t+1} - (1-nu) g_t^2] / (kappa - p_{t+1})
    1 + q_t = (1 + q_{t+1})(s - nu g_t / (kappa - p_{t+1}))

Verified against the independent optimiser to 5e-14. **The multi-asset case
reduces exactly to the scalar one** — asset-specific quantities enter only
through `nu` and the direction `d`.

**R0. Concavity is automatic** (found during the final audit). `p_t <= 0` by
backward induction, so `kappa - p_{t+1} >= kappa > 0` always. No hypothesis
needed. 4000 random configurations, never violated.

**R2. At `s = 1`, `kappa* = phi(1 - nu)` recovers the equilibrium exactly.**
`alpha_t = 0` needs `g_t = 0`, and `p_T = 0` forces `kappa(s-1) = 0`. Proved in
full. Verified to 1e-16, any horizon, any asset count, `nu` from 0.001 to 0.98,
`phi` from 0.1 to 50.

This explains the earlier unexplained numerical value `kappa* = 0.808` in the
demo market: `phi(1 - nu) = 1 x (1 - 0.1916) = 0.8084`.

**R3. For `T = 1` the gap vanishes at every `s`.** The alpha mismatch multiplies
a deterministic `x_0` and merges with beta, so one `kappa` absorbs it. The
surrogate gap is genuinely multi-period.

**R4. For `s != 1`, `T >= 2`, the gap is irreducible and linear in `(s-1)`.**
Exponent 1.0000 everywhere tested. Reason, visible in R1: to first order
`alpha_t = -d(s-1)`, independent of `kappa`, so it cannot be tuned away.

**Mistake caught here.** The first scaling fit used `min_kappa eps_surr` and
returned exponent 0.52 with residual 0.48. Cause: `eps_surr` changes sign, so
minimising it finds the surrogate *beating* the equilibrium, and it crosses zero
where the policies still differ. Switched to the **policy distance**, which gave
a clean power law. This is the same error that forced the Theorem 1 restatement.

**R5. For `T = 2` the constant is closed form:**

    c = ||d|| sqrt((2 - nu)(1 + nu)) / (2 phi (1 - nu))

Verified to 0.11% worst case, exponent 1.0000, single- and multi-asset.

**Result.** `Lambda(kappa) = |kappa - phi(1-nu)| + c_T |s - 1|`, `gamma = 1`.

---

## Stage 6 — Paper, and the second audit

Wrote the paper. Then, before finalising, searched the literature and re-checked
every claim. Both steps changed the paper.

**Priority search.** The distinction between summing per-period mean-variance
tradeoffs and Li & Ng's terminal criterion is **standard** in portfolio choice —
Gârleanu & Pedersen (2013) and DeMiguel et al. (2015) do exactly the former. So
"these objectives differ" is not a contribution and the paper does not claim it
as one. What the search did *not* turn up is the quantitative map
`kappa* = phi(1-nu)`. That may be new; the search was not systematic and this is
flagged as the single largest risk.

**Audit, six checks. Three claims changed.**

1. **The Sharpe misranking is not cherry-picked.** 519 random (market, phi)
   pairs; Sharpe and `J` disagree on the ordering in **93.1%**. The paper had
   shown one table and implied nothing; it now reports the sweep. This is a much
   stronger result than was originally claimed.
2. **The crossover exists generally**, but its location is market-specific
   (phi between 4 and 10 across random markets, against 12 in the demo market).
   The paper said "near 12" without qualification.
3. **Negative `eps_learn` is structural.** It is measured in `J` while the agent
   optimises `J_r`; a worse maximiser of `J_r` can sit closer to the `J`-optimum,
   and such perturbations of `pi_r` were exhibited directly. The paper's earlier
   explanation — "the optimiser lands either side" — was wrong.
4. **`c_T` is superlinear in the horizon, not a power law**: log-log exponent
   1.16 with max residual 0.08 over `T = 2..10`. Earlier notes said "close to
   linear".
5. Theorem 4 survives extreme parameters (worst 1.1e-10 at `nu = 0.98`).
6. Concavity never failed, which produced R0.

---

## Stage 7 — Third pass: the continuous-time limit

Ran the priority search properly rather than shallowly. It did not turn up
`kappa* = phi(1-nu)`, but it turned up something better: a reason why.

**The correction is `O(Delta)`.** With rebalancing interval `Delta`,
`nu = SR^2 * Delta + O(Delta^2)`, verified numerically. So `kappa* -> phi` in
continuous time. The equilibrium mean-variance literature works in continuous
time, where the factor is identically zero — so its absence there is expected
rather than surprising. This is an argument for novelty, not a proof of it, and
the paper says so.

It also gives the practical boundary: 11% correction at annual rebalancing,
2.7% quarterly, under 0.25% weekly. A practitioner rebalancing daily may set
`kappa = phi` safely; one rebalancing annually should not.

**External validation.** The equilibrium recursion was derived here rather than
quoted, so it needed a check against something outside this work. Discretising a
year into `n` steps and comparing against Basak & Chabakauri's continuous-time
formula gives relative differences 0.76%, 0.25%, 0.06%, 0.01%, 0.00% for
`n = 4, 12, 52, 252, 2520`. The convergence rate is itself a check on the `nu ~
SR^2 Delta` scaling.

Both added to the paper as Section 4.2 and to the verification subsection.

---

## Stage 8 — Phase 3: the differential Sharpe ratio

Expected this to need new machinery. It did not.

Expanding the DSR increments algebraically shows it is a variance-penalised
return reward with `kappa_eff = A/(2B)`. Since `A`, `B` estimate the first two
moments of the realised return, `kappa_eff -> E[rho]/(2E[rho^2])` — a market
quantity with no preference parameter in it.

**False start.** The first run gave `B* = 3080`, nonsense. Cause: over a
4-period horizon with `eta = 0.05` the moment estimates never leave their
initial values, so `kappa_eff` was pure initialisation artefact. Rebuilt around
the stationary regime (long horizon, or `A_0, B_0` started at converged values).

Result: the DSR optimum sits in the variance-penalised family at `kappa = 21.56`
against a predicted 22.51, residual distance 0.019. Implied risk aversion 27.9.

**Overclaim caught.** A draft sentence said tuning `eta` "silently retunes the
investor's risk preference". Measured effect across `eta` in [0.02, 0.4]: about
2%. Softened.

---

## Stage 9 — Phase 5: out of model

The proposal called for a historical comparison; no market data is available
here, so the substitute is deliberate misspecification with the first two
moments held fixed.

**Result that contradicted the text I had already written.** I wrote "the
ranking is unchanged" before running it. Under AR(1) momentum the ranking
*reverses*, and the pre-committed policy collapses from 1.427 to 0.929. Fat
tails and volatility clustering leave it intact; dependence does not. Corrected,
and promoted from a caveat to a stated limitation of the whole analysis.

**Statistical power.** Asking how often a Sharpe comparison of `n` independent
replications selects the `J`-better policy: 0.022 at `n=10`, 0.000 at `n>=200`.
More data makes it more confidently wrong. This is the strongest version of the
paper's third gap.

---

## What is not done

- **The priority search.** Now done across the equilibrium mean-variance and
  dynamic-trading literatures; `kappa* = phi(1-nu)` did not appear, and
  Section 4.2 of the paper explains why it would not. Still not *systematic* —
  a formal search including the operations-research and stochastic-control
  literatures remains outstanding. If the result is known, the contribution
  reduces to the decomposition, the transfer to RL, and the numerical section.
- **Theorem 8's error terms** are confirmed numerically, not bounded. It is a
  series expansion, not yet a proof.
- **`c_T` for `T >= 3`** — open.
- **The differential Sharpe ratio** — not covered. It needs the state
  augmentation `z = (x, A, B)`, which is set up in the paper but not analysed.
- **Non-i.i.d. returns** — Proposition 1 extends to time-varying moments;
  Theorem 4 does not, since one `kappa` cannot satisfy `kappa = phi(1-nu_t)` for
  all `t`. The gap is expected to be strictly positive even at `s = 1` under
  time-varying moments, but this is untested.
- **Reading the sources.** The author has not yet read Hambly §4.3, Li & Ng
  (2000), or Basak & Chabakauri (2010) in full. These underpin Sections 2–4 and
  must be read before the paper is defended.

---

## Errors caught, collected

Recorded together because the pattern is the useful part: every one was found by
a check, not by inspection.

| # | Error | How it surfaced |
|---|---|---|
| 1 | Optimality check passed vacuously (Nelder–Mead didn't converge) | Comparing the optimiser's value against the closed form in both directions |
| 2 | Claimed the reversal is present on the mean path; it vanishes there | Computing the crossing point instead of asserting |
| 3 | REINFORCE peaked then degraded | Plotting the training curve rather than reading the final value |
| 4 | `eps_surr` treated as sign-definite | Scaling fit returning exponent 0.52 with a huge residual |
| 5 | Predicted constant `c` wrong by 12–59% | Comparing prediction against measurement across configurations |
| 6 | Figure 1 contradicted Table 2 (different `kappa`) | Reading the rendered page |
| 7 | Wrong explanation of negative `eps_learn` | Auditing rather than accepting the plausible story |
| 8 | "`c` grows linearly in `T`" | Fitting rather than eyeballing |
| 9 | Concavity stated as a hypothesis when it is automatic | Testing when it fails, and finding it never does |
| 10 | Proposal claimed `Lambda = 0` for the log-return reward | Actually running the consistency check it proposed |
| 11 | DSR `B* = 3080` from initialisation, not the market | Sanity-checking a number that looked absurd |
| 12 | Claimed `eta` strongly retunes risk preference (it is ~2%) | Measuring the sensitivity instead of asserting it |
| 13 | Wrote "ranking is unchanged out of model" before running it | Running it |
