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

## Open

- The constant `c` for `T >= 3`. Computable numerically. Its growth in the
  horizon is **superlinear and not a power law**: a log-log fit over
  `T = 2..10` gives exponent 1.16 with max log residual 0.08. (An earlier note
  here said "close to linear"; that was wrong.) The `T = 2` derivation is a
  first-order perturbation of both recursions and generalises in principle, but
  the algebra grows quickly.
- Reward classes beyond the variance-penalised family, in particular the
  differential Sharpe ratio, which needs the state augmentation of Step 2.
- Non-i.i.d. returns.

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
