# Current findings — 22 September 2026

This file supersedes earlier claims. Historical versions are preserved in archive/2026-09-21-before-audit and the original delivered ZIP. PROVED means a mathematical derivation under the stated assumptions, not independent peer review or established publication priority. NUMERICALLY VERIFIED means verified in the specified finite experiment. PRELIMINARY means evidence limited by sampling/model scope. OPEN means unresolved. No current conjecture is used as an established premise.

| Result | Status | Evidence / limitation |
|---|---|---|
| Li–Ng precommitted policy and equilibrium reference | PROVED, known theory; numerically verified implementation | Existing verifier; primary literature ledger |
| Raw quadratic increment Bellman recursion | PROVED | notes/PROOFS.md §2; exact inner optimisation |
| Zero-interest κ=φ(1−ν) calibration | PROVED; not claimed novel | Also the fixed point of the adapted existing MVPI |
| General fixed-horizon mismatch coefficient | PROVED | notes/PROOFS.md §3; analytic remainder and global local-in-interest minimiser |
| c_T asymptotic T^(3/2) | PROVED as a coefficient asymptotic | Not a uniform joint T→∞, s→1 theorem |
| Exact terminal-unit reward redesign | PROVED | Independent moments, unconstrained dollars, equilibrium target |
| Common-penalty lower bound for changing moments | PROVED | Weighted dispersion; not automatically positive for all time variation |
| Observed-regime equilibrium and oracle hedge correction | PROVED | Joint return/next-regime law, scalar risky asset, solved continuation |
| DSR finite-horizon well-posedness and local identity | PROVED | Bounded returns, positive initial EWMA variance; no full fixed-penalty equivalence |
| Fixed-decay EWMA consistency | DISPROVED; correction PROVED | Limiting variance eta*sigma²/(2−eta)>0 for iid inputs |
| Exact sweeps and published reward adaptations | NUMERICALLY VERIFIED | 120 base configurations, 27 moment configurations, finite action trees |
| Five-seed learning comparison | PRELIMINARY | No learned mean–variance improvement from redesign at current budget |
| Moment estimation / frozen-policy stress | PRELIMINARY | Thirty estimation seeds; ten stress batches; AR(1) reverses design-vs-raw J |
| Publication priority of quantitative results | OPEN | Targeted primary-paper search; no broad “first” claim |
| Unrestricted/infinite-horizon DSR | OPEN | Finite grid is not a continuous/unbounded proof |
| General dependent multi-asset, constraints and transaction costs | OPEN | Scoped regime/long-only evidence only |
| Historical or PPO/DDPG algorithm replications | OPEN / not performed | Reward adaptations are not algorithm replications |

## Headline, with the right scope

A scalar penalty on squared wealth increments generally cannot recover the terminal mean–variance equilibrium at nonzero interest. Its best local policy mismatch has an explicit coefficient for every fixed horizon. Measuring excess gains in terminal units and applying date-specific moment calibration exactly recovers equilibrium under independent returns and unconstrained dollar holdings.

This is a concrete diagnostic and conditional design result. The zero-interest calibration alone is insufficient novelty: Zhang–Liu–Whiteson's existing MVPI reduces to the same fixed point. Equilibrium RL and associated time-consistent objectives also predate this project.

## Main exact comparison

Three risky assets, T=4, phi=1, s=1.02, wealth1; fixed equilibrium occupancy distance:

| Method | Distance | Terminal J | Terminal excess Sharpe |
|---|---:|---:|---:|
| Precommitted reference | 3.312087 | 1.425799 | 1.171951 |
| Equilibrium | 0 | 1.323644 | .982266 |
| Raw penalty κ=φ | .797127 | 1.316580 | .987285 |
| Scalar κ=φ(1−ν) | .195347 | 1.326115 | .987292 |
| Best scalar within raw family | .186435 | 1.326093 | .987292 |
| Terminal-unit design | numerical zero | 1.323644 | .982266 |
| Existing MVPI adaptation | .313586 | 1.325487 | .987293 |

Equilibrium alignment is not maximisation of initial J. Scalar and adapted MVPI can exceed equilibrium J. Sharpe is not annualised; the risk-free terminal payoff is subtracted correctly.

## Negative and limiting results

In the bounded 21-point long-only tree, design distance .041366 improves on raw .058485, but design J=1.107729 is slightly lower than raw J=1.107849 and scalar J=1.108217. The exact-recovery theorem does not extend through wealth-dependent constraints. Grid refinement changes small J differences, so broad ranking claims are not justified.

Five-seed learned J means: raw1.305628, scalar1.302556, design1.300671. Mean distances .8638, .8312, .8079 respectively. The learned design has better average alignment but substantial residual error and lower J. This is preliminary and not evidence of algorithm superiority.

Frozen iid-designed policies under AR(1) returns: design minus raw J=−.03813 (batch SE .00027), compared with +.00713 (SE .00018) under Gaussian iid. Neither frozen iid policy is asserted optimal under dependence.

## Corrections to the delivered proposal/paper

- Squared increments penalise second moment, not variance. Conditional variance rewards differ.
- Additive objectives admit Bellman optimisation on an adequate Markov state. Their mere additivity is not time inconsistency.
- Precommitment and equilibrium are different targets. Log growth is a valid separate objective.
- Analytic dollar holdings are unconstrained; simplex experiments require separate reference policies.
- Signed financial-value differences are not three nonnegative error terms. Report nonnegative surrogate regret separately.
- nu<1 needs positive-definite covariance; uniqueness requires nonzero mean. Single-period matching needs a positive feasible penalty and only matches at initial wealth.
- The global Lambda bound is false as κ→0. The local expansion is now proved.
- General-horizon coefficient is no longer open; old approximate linear-in-horizon wording is withdrawn.
- DSR state-dependent scaling cannot be discarded in a dynamic optimisation. Fixed-decay EWMA is not consistent.
- Volatility clustering is dependent; t(4) fourth-moment uncertainty was problematic. Revised stress uses t(8).
- The old 519-case / 93.1% claims had no delivered generator/output and are withdrawn.
- Sharpe now subtracts x0 times compounded risk-free growth. Learning rate is .02, not .03.
- Broad novelty statements are removed. Dai et al. explicitly learn equilibrium, while Bisi/Zhang explicitly analyse risk objectives/reward transformations.

See proposal.pdf for the research case, paper.pdf and notes/PROOFS.md for derivations, IMPLEMENTATION_REPORT.md for protocols, literature/LITERATURE_LEDGER.md for exact comparisons, and results/experiments.json for every current numerical claim.
