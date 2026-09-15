# Findings

Two sections, kept apart on purpose. Anything under **New** needs a literature
check before it is claimed anywhere that matters.

## Reproduced (known results, used as correctness evidence)

- **Mean-variance is time-inconsistent; the Bellman equation fails.**
  Strotz (1955); Björk & Murgoci (2014). Demonstrated in `inconsistency.py`.
- **The pre-committed optimum is affine in wealth.** Li & Ng (2000).
  Implemented in `policies.py`, verified against direct optimisation to 2.6e-12.
- **The equilibrium control is wealth-independent.** Basak & Chabakauri (2010).
  `alpha_t = 0` identically, equivalently `A_t^2 = B_t`. Reproducing this is the
  main evidence that the backward recursion in `equilibrium.py` is right.

## New (unverified against the literature)

- **The preference reversal vanishes exactly on the expected wealth path.**
  The committed and re-solved controls agree at `x_1 = E[x_1]` to 1.8e-15 across
  markets, horizons, risk aversions and asset counts. Mechanism: `lambda` is
  pinned by `E[x_T]`.
  *Check against:* Li & Ng (2000), Basak & Chabakauri (2010). Plausibly known.

- **`eps_incons` scales exactly as `1/phi` and is independent of `x0`.**
  Verified to 9e-13 and 5e-15.
  *Check against:* the same two papers. Likely follows from the embedding.

- **The plain-return reward has no finite optimum.**
  The additive objective diverges along increasing leverage. The mean-variance
  problem it stands in for is bounded. Bai et al. (2024) report this as the most
  common reward in the surveyed literature.
  *Check against:* anything on unboundedness of return-maximising objectives.
  The fact is elementary; whether it has been stated about RL reward design in
  this literature is the question.

- **`eps_surr` is not sign-definite.**
  At `kappa* = 0.808` the surrogate optimum beats the equilibrium policy:
  `eps_surr = -0.0025`. So the additive surrogate is not a degraded equilibrium
  policy but a third object. **The proposal currently describes `eps_surr` as a
  shortfall and needs correcting.**

- **The Sharpe ratio ranks policies in the wrong order.**
  The `kappa = phi` surrogate has a higher Sharpe ratio than the equilibrium
  policy (1.1955 vs 1.1501) and a lower objective (1.3166 vs 1.3236).
  This is Gap 3 of the proposal, demonstrated.

- **Which error term dominates depends on `phi`, with a crossover near 12.**
  `eps_incons` decays as `1/phi`; `eps_surr` grows. Below the crossover
  time-inconsistency dominates, above it reward design does.
  *Consequence:* studies disagreeing about the importance of reward design may
  simply be on opposite sides of this crossover.

## Phase 2, settled: the structure of Lambda

For the i.i.d. market with variance-penalised reward `r = d - kappa d^2`.
Derived symbolically, verified numerically. See `lambda_functional.py`.

- **R1. The surrogate optimum satisfies a backward recursion.** The additive
  objective is a sum, hence time-consistent, hence subject to dynamic
  programming. With `d = M^-1 m`, `nu = m^T M^-1 m`,
  `g_t = kappa(s-1) - p_{t+1} s`:

      alpha_t = -d g_t / (kappa - p_{t+1})
      beta_t  =  d (1 + q_{t+1}) / (2(kappa - p_{t+1}))
      p_t     = [kappa p_{t+1} - (1-nu) g_t^2] / (kappa - p_{t+1})
      1 + q_t = (1 + q_{t+1})(s - nu g_t / (kappa - p_{t+1}))

  Verified against an independent optimiser to 5e-14. **The multi-asset case
  reduces exactly to the scalar one**: asset-specific quantities enter only
  through `nu` and the direction `d`.

- **R2. At `s = 1`, `kappa* = phi(1-nu)` recovers the equilibrium exactly.**
  Verified to 1e-16 in policy distance, any horizon, any number of assets.
  This explains the earlier unexplained numerical value `kappa* = 0.808`.

- **R3. For `T = 1` the gap vanishes at every `s`.** The alpha mismatch
  multiplies a deterministic `x_0` and merges with beta, so one `kappa` absorbs
  it. The surrogate gap is genuinely multi-period.

- **R4. For `s != 1`, `T >= 2`, the gap is irreducible and linear in `(s-1)`.**
  Exponent 1.0000 in every configuration tested, so `gamma = 1` rather than a
  quantity to be fitted. Reason: to first order `alpha_t = -d(s-1)`,
  independent of `kappa`, so it cannot be tuned away.

- **R0. Concavity is automatic.** For `kappa > 0`, `p_t <= 0` by backward
  induction, so `kappa - p_{t+1} >= kappa > 0` always. The one-step surrogate is
  concave without hypothesis. Verified over 4000 random configurations. This is
  now Lemma 2 of the paper; the code's guard is unreachable.

- **R5. For `T = 2` the constant is closed form:**

      c = ||d|| sqrt((2-nu)(1+nu)) / (2 phi (1-nu))

  Verified to 0.11% worst case, single- and multi-asset.

**Result:** `Lambda(r) = |kappa - phi(1-nu)| + c|s-1|`, with `gamma = 1`,
vanishing exactly when the surrogate optimum is the equilibrium policy.

**Theorem 1 of the proposal was misstated and has been corrected** in
`proposal.tex`: it now bounds the policy distance, not `eps_surr`. `eps_surr` is
a scalar difference, changes sign, and crosses zero where the policies still
differ, so an "if and only if" stated for it is false.

## Error found in the proposal's own consistency check

The proposal claimed `Lambda = 0` for the log-return (Kelly) reward, on the
reasoning that log-wealth is additive and therefore time-consistent. **That
reasoning is wrong and the claim is false.**

Time-consistency of the surrogate means dynamic programming is *valid* for it.
It says nothing about *which* objective it is consistent with. Log-return is
consistent with expected log-wealth, whose optimum is the Kelly policy;
`Lambda` measures distance from the mean-variance equilibrium. Those are
different objects.

Verified directly (s=1, T=3, phi=1, one asset):

    t    Kelly alpha   Kelly beta    eq alpha    eq beta
    0        1.53941     -0.21525     0.00000    0.92593
    1        1.27372      0.06661     0.00000    0.92593
    2        1.15704      0.17633     0.00000    0.92593

Kelly holds a *fraction* of wealth; the equilibrium holds a constant *amount*.
Different functional forms, not different parameters.

Consequence: time-consistency of the surrogate is **necessary but nowhere near
sufficient** for small `Lambda`. Corrected in `proposal.tex`. The paper never
made the claim, so it needs no change.

## Phase 3: the differential Sharpe ratio

**It is a variance-penalised reward in disguise.** Expanding the increments,

    B dA - (1/2) A dB = (eta/2)(2 B rho - A rho^2 - A B)

so up to a positive state-dependent scale and a constant,

    r^DSR  ~  rho - kappa_eff rho^2,     kappa_eff = A/(2B).

- `A_t`, `B_t` converge to the true first and second moments of the realised
  return. At T=200: `A* = 0.021636` vs `E[rho] = 0.021676`; `B* = 0.000479` vs
  `E[rho^2] = 0.000482`. So `kappa_eff -> E[rho]/(2 E[rho^2]) = 22.51`, which
  contains **no preference parameter**.
- Optimising the DSR directly and finding the closest member of the
  variance-penalised family gives `kappa = 21.56` against the predicted 22.51 —
  4% agreement, residual policy distance 0.019.
- Inverting Theorem 4: `phi_implied = kappa_eff/(1-nu) = 27.9`. The practitioner
  believes they are asking for risk-adjusted return; they are asking for
  mean-variance at a risk aversion the market sets.
- Decay `eta` over [0.02, 0.4] moves it only 30.4 to 31.0, so it is not a
  hyperparameter artefact. (An earlier draft claimed tuning `eta` "silently
  retunes risk preference" — the effect is ~2%, so that was overstated.)

So the DSR is **covered by Theorem 4 rather than excluded from it**, with
`|kappa_eff - phi(1-nu)|` as Lambda's first term — except that term cannot be
driven to zero by choice.

## Phase 5: out of model, and the power of a backtest

**Robust to higher moments, NOT to dependence.** Under processes matched to the
same first two moments:

| process | pre-committed | equilibrium | surrogate k* | surrogate k=phi |
|---|---|---|---|---|
| Gaussian | 1.4267 | 1.3236 | 1.3261 | 1.3166 |
| Student-t(4) | 1.4252 | 1.3238 | 1.3263 | 1.3166 |
| vol clustering | 1.4037 | 1.3234 | 1.3258 | 1.3163 |
| AR(1) momentum | **0.9285** | 1.1886 | 1.1865 | **1.2270** |

Fat tails and volatility clustering leave the ordering intact — they perturb
higher moments while preserving independence, and the analysis uses only the
first two. **Serial correlation reverses the ordering**, and the pre-committed
policy collapses from 1.427 to 0.929: its wealth slope is calibrated to a drift
that momentum makes wrong and it cannot revise.

*This is a real limitation.* Every closed form assumes independence across
periods. The results should not be read as applying to markets with
predictability.

**A backtest gets it wrong, and more data makes it worse.**

| replications | P(Sharpe picks the J-better policy) |
|---|---|
| 10 | 0.022 |
| 50 | 0.003 |
| 200 | 0.000 |
| 1000+ | 0.000 |

More data increases confidence in the wrong answer, because it estimates the
wrong quantity with increasing precision. This is the sharpest form of the
claim: the error is not hard to see in a backtest, it is invisible in a way
additional data aggravates.

## Open

- The constant `c` for `T >= 3`. Computable numerically. Its growth in the
  horizon is **superlinear and not a power law**: a log-log fit over
  `T = 2..10` gives exponent 1.16 with max log residual 0.08. (An earlier note
  here said "close to linear"; that was wrong.) The `T = 2` derivation is a
  first-order perturbation of both recursions and generalises in principle, but
  the algebra grows quickly.
- **Dependent returns.** The most important remaining item, promoted from a
  caveat by the Phase 5 result. Proposition 1 extends to time-varying
  `(m_t, M_t, s_t)` with `nu_t` replacing `nu`; Theorem 4 then needs `s_t = 1`
  for all `t` and a single `kappa = phi(1-nu_t)` simultaneously, generally
  infeasible. Establishing the size of the resulting gap is open.
- Reward classes beyond the variance-penalised family and the DSR.

## Third pass: the continuous-time limit

Two results that materially change the paper's positioning.

- **The `(1-nu)` correction is a discrete-time effect.** With rebalancing
  interval `Delta`, `m = O(Delta)` and `M = Sigma*Delta + O(Delta^2)`, so

      nu = m^T M^-1 m = SR^2 * Delta + O(Delta^2)

  Verified: `nu` matches `SR^2 * Delta` to three significant figures across
  annual through daily. So `kappa* -> phi` as `Delta -> 0`.

  Consequence for **priority**: the equilibrium mean-variance literature
  (Basak & Chabakauri, Björk-Murgoci-Zhou, Wang & Zhou) works in continuous
  time, where this factor is identically zero. Its absence there is expected.
  This is an argument for novelty, not a proof of it.

  Consequence for **practice**: 11% correction at annual rebalancing, 2.7% at
  quarterly, under 0.25% at weekly. Matters for coarse rebalancing only, and
  `nu ~ SR^2 Delta` says exactly where the boundary is.

- **External validation of the equilibrium recursion.** It was derived here
  rather than quoted, so it needed independent confirmation. Discretising one
  year into `n` steps and comparing against Basak & Chabakauri's continuous-time
  formula `u = (1/gamma)((alpha-r)/sigma^2) e^{-r(T-t)}` with `gamma = 2 phi`:
  relative differences 0.76%, 0.25%, 0.06%, 0.01%, 0.00% for
  `n = 4, 12, 52, 252, 2520`. Convergence at the rate implied by `nu ~ SR^2
  Delta` is itself a check on that scaling.

## Audit, second pass

Six checks run over the whole result set after the paper draft. Three claims
changed.

- **The Sharpe misranking is not a constructed example.** Over 519 random
  (market, phi) pairs the Sharpe ratio and `J` rank `pi_eq` against `pi_r` in
  opposite orders in **93.1%** of cases. The paper previously showed one table
  and implied nothing about generality; it now reports the sweep.

- **A crossover between `eps_incons` and `eps_surr` exists generally**, but its
  location is market-specific: between phi = 4 and 10 across random markets
  against 12 in the demo market. The paper said "near 12" without qualification.

- **Negative `eps_learn` is structural, not seed noise.** It is measured in `J`
  while the agent optimises `J_r`; a worse maximiser of `J_r` can sit closer to
  the `J`-optimum. Verified by exhibiting such perturbations of `pi_r` at every
  phi tested. The paper's earlier explanation ("stochastic optimiser lands either
  side") was wrong.

- Theorem 4 survives extreme parameters: `nu` from 0.001 to 0.98, `phi` from 0.1
  to 50, `T` up to 7. Worst distance 1.1e-10, at the conditioning limit.

- Concavity never failed in 400 configurations, which led to R0 above.
