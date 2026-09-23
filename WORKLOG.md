> Current-state notice (22 September 2026): this is a historical diary. The latest dated entry and FINDINGS.md supersede earlier proof, novelty, numeric and completion claims.

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

## 21 September 2026 — Complete input audit
Read the whole delivered project and ZIP before changes; see notes/INITIAL_AUDIT.md for coverage and contradictions. Original documents, figures, and newer standalone files preserved under archive/2026-09-21-before-audit. No separate implementation report or raw experiment records were delivered. Current source of truth will be this package; parent duplicates will point here. Dependency installation is isolated in /private/tmp/mvrl-env. Research first: general-horizon proof, reward redesign, dependent benchmark, DSR correction; proposal comes after verified results.

## 21–22 September 2026 — Research completed at a stated scope; proposal rebuilt

**PROVED:** general fixed-horizon small-interest coefficient and remainder, including global penalty minimisation near zero interest. c_T grows as T^(3/2), not linearly. A terminal-unit excess-gain reward with date-specific second-moment correction exactly recovers the independent-return equilibrium. Common raw-penalty failure under varying moments has a weighted-dispersion lower bound, with equal-nu exceptions. Corrected positivity/zero-mean/one-period qualifications; withdrew the global Lambda inequality.

**PROVED:** observed scalar regime equilibrium includes a continuation covariance hedge. Added exact joint return/next-regime backward recursions, precommitment embedding, target occupancy evaluation and an equilibrium-informed reward correction. A new independent recovery test caught an erroneous (1−nu) factor in the linear hedge correction. It was removed from both proof and implementation before reporting; final coefficient is −2phi H u Cov(P,a_next|z).

**PROVED / NUMERICALLY VERIFIED:** DSR local algebra, positive augmented variance propagation, bounded finite-horizon well-posedness; fixed-eta EWMA inconsistency. Exhaustive finite trees demonstrate dynamic continuation and sensitivity to initial A,B and eta. No fixed-kappa dynamic equivalence, constant implied risk aversion or infinite-horizon fixed point is claimed.

**NUMERICALLY VERIFIED:** 120 independent parameter configurations (720 policy records), 27 mean/volatility/correlation configurations (81 records), 30 scaling cases, time-varying and four observed-regime markets, 64 bounded reward/grid/risk-aversion comparisons, 12 DSR initialisation/decay cases, four MVPI adaptations. Closed-form policies checked with independent optimisation, enumeration and one-step deviations.

**PRELIMINARY:** 15 matched-budget training runs (five seeds, three rewards, 4,000 iterations, batch512), 360 moment-estimation records and 160 frozen-policy stress records. Redesign gives exact alignment but no learned-J advantage at current budget; AR(1) stress reverses its J advantage over raw. Included these negative findings prominently.

**Literature:** inspected primary versions of Kolm–Ritter, Jiang–Xu–Liang, Yang et al., Moody–Saffell, Bisi et al., Zhang–Liu–Whiteson, Wang–Zhou, Dai–Dong–Jia, Li–Ng, Basak–Chabakauri (thesis chapter), and Ng–Harada–Russell. Exact metadata/locators and limitations in ledger. Björk–Murgoci remains abstract-only, Ritter2017 inaccessible. Some author versions differ from final publications; no journal section numbers invented. Moody/Ng PDFs were visually inspected because extraction was corrupt. Broad absence-of-objective-analysis and first-equilibrium-RL claims are false. The adapted MVPI fixed point gives the same zero-interest calibration, proved algebraically and verified numerically. Publication priority of the quantitative results remains OPEN.

**Implementation:** stable wealth-variance propagation, corrected excess Sharpe subtraction, integrated/rewrote standalone DSR and stress scripts, replaced stale module claims, retained archival originals. Saved all outputs, histories, protocols, environment and builders. New pytest suite: 25 passed; original verifier: 10 passed. Generated ten scientific figure pairs. Proposal now centres the problem/method/result and includes comparisons, established results, limitations and proposed extensions. Paper carries the full proof appendix. Both use shared content and saved data; PDFs rendered for visual verification. README/FINDINGS/implementation report and root copies reconciled.

**Still OPEN:** unrestricted/infinite-horizon DSR, constrained/cost/dependent multi-asset exact designs, estimated continuation hedges, systematic priority verification and full published algorithm/historical replications. A completed audit/revision is not completion of every possible research extension. Old diary statements below/above are historical; current FINDINGS.md governs.


## 22 September 2026 — Restore the original LaTeX presentation

User requested that both revised documents retain the original proposal/paper format and font. Inspected the archived original sources and title pages: 11pt A4 `article`, one-inch margins, UTF-8/T1 Computer Modern, original titles and Matin Tavakoli author blocks, standard numbered headings and centred page numbers. Preserved the report-style revision in `archive/2026-09-22-report-layout/`. Replaced the ReportLab renderer with genuine LaTeX source generation and compilation; converted display mathematics and proof appendix to native LaTeX, retained all ten figures as vector PDFs, kept updated evidence labels and literature limitations. The revised research data and experiments are unchanged. The paper's proof material is now in a conventional appendix. Compilation and visual QA are recorded after the build below.

Final presentation verification: compiled both documents with pdfLaTeX (TeX Live 2026), three passes each. Proposal: 12 pages; paper: 17 pages. Both retain the original 11pt A4 article, one-inch margins, Computer Modern body/math fonts, original title/author styling, and standard page numbering. Each has ten vector figures and twelve resolved source records. No overfull boxes, missing glyph warnings, or unresolved citations/references. Rendered and inspected all pages; corrected orphan headings and awkward final-section placement. Research-output SHA256 is unchanged. Saved `results/latex_verification.txt` and compiler logs; refreshed complete delivery and added a compact standalone `latex-original-format.zip` source bundle.
