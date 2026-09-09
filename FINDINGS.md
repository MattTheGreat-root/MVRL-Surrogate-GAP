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

## Open

- The functional `Lambda` of Theorem 1 in the proposal. Currently `eps_surr` is
  computed numerically per configuration with no closed form and no bound.
  This is the actual research.
