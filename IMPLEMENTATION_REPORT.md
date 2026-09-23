# Implementation and verification report — 22 September 2026

## Deliverables and provenance

All existing research text/code/archive contents and every supplied figure were inspected before edits. The initial inventory is notes/INITIAL_AUDIT.md. Originals are preserved under archive/2026-09-21-before-audit. No historical dataset, raw experiment output, or standalone implementation report was present initially. This report documents the new run, not a reconstruction of unavailable old outputs.

## Model and algorithms

Main iid market: m=(.06,.04,.02); vol=(.18,.12,.08); correlation rows (1,.30,-.10), (.30,1,.05), (-.10,.05,1). x0=1. The analytic market uses risky dollar holdings without leverage/solvency constraints. Gaussian returns are used in training. Exact moment results do not require Gaussianity. nu=0.1943356810 approximately; use computed values, not this rounded value.

Raw reward: ΔX−κΔX². Scalar: κ=φ(1−nu). Best raw: exact quadratic minimisation in h=1/(2κ) using fixed equilibrium occupancy. Design: Y−φ(1−nu_t)Y² with Y=H_t P_t'u. Conditional-variance reward has the same independent-market optimum (proved; not a distinct learned method here).

Distances: sqrt(sum_t E_reference ||u_candidate(t,state)−u_target(t,state)||²). Main and dependent results use fixed target occupancy. The small-interest theorem and scaling experiments use original candidate occupancy. These are discrepancies, not globally symmetric policy norms. Bounded policies are maps from full return history to portfolio weights, compared on target wealth histories in dollar units.

Independent sweep: T in {1,2,4,8,16}, φ in {.25,.5,1,2,4,16}, s in {.98,1,1.02,1.08}: 120 configurations, six policies each. Market sensitivity: three mean scales (.5,1,1.5), three volatility scales (.75,1,1.25), three off-diagonal correlation scales (0,1,1.5): 27 configurations. All chosen covariance matrices are positive definite. Small-interest: T={2,3,4,6,10,16}, increments={−.001,−.0001,.0001,.001,.01}. Analytic calibration avoids arbitrary optimiser bounds.

Dependent market: two observed regimes, transition [[q,1−q],[1−q,q]], q={.5,.65,.8,.95}; conditional edge means [[.09,−.05],[.04,−.02]]; each edge has symmetric innovation ±.12. Initial regime 0; T=4, s=1.02, φ=1. Returns and next regime are jointly modelled. q=.5 does NOT make returns iid because edge means depend on the current regime. Hedge correction uses known equilibrium continuation. This is not an estimated or model-free algorithm.

Bounded comparison: one risky asset plus cash, T=3, wealth1, s=1.02; excess return −.18 or .30 with probability .5 each. Long-only weight grids {0,.1,...,1} and {0,.05,...,1}. All reward policies have identical history information and feasible actions. Exhaustive reward Bellman solves are exact for the finite grid; the equilibrium uses one-step deviations on that same grid. φ={.5,1,2,4}. No continuous-action convergence rate is asserted. Max |J_21−J_11| is .001218 (raw), .000878 (design), .003595 (DSR), which is material for small J differences. DSR follows Moody–Saffell's unscaled signal with return rho=X_next/X−1, eta=.1, A0=.01,B0=.01. Sensitivity: eta={.02,.1,.4}; (A0,B0)={(0,.001),(.01,.01),(.02,.002),(−.02,.002)}. Positive gross asset returns and long-only actions guarantee positive wealth. Zero costs; undiscounted rewards. Kolm–Ritter's κ=.1 maps to coefficient .05, but its currency units do not transfer. Only a diagnostic.

MVPI: Zhang et al. Algorithm 1 adapted from infinite-horizon discounted occupancy to a uniformly selected date in a finite horizon. Exact inner quadratic solves; y equals E[ΔX] under uniform-time occupancy. Iterate to residual <1e−12; report monotonic objective trace. Numerical convergence is a fixed-point result, not a proof of global mean-volatility optimality at nonzero interest.

Learning: existing affine Gaussian REINFORCE-style routine with Adam, normalised advantages and scalar-rescaled scores; learning rate .02; 4,000 iterations; batch512; seeds0–4; initial coefficients zero; exploration from 0.5 to 0.03 (verify defaults in agent.py). Record every200 iterations. The same seed schedules and sample budgets are used for raw/scalar/design. Final deterministic mean policies are evaluated by exact moments; no test-set Monte Carlo noise. Scores are scaled updates, not a full Fisher preconditioner. Reward design has moment knowledge; training does not receive the equilibrium policy. Hyperparameters were not retuned per reward. Five seeds support preliminary diagnostics only. Surrogate regrets use each reward's own units.

Estimation: samples N={50,200,1000,10000}, seeds1000–1029; sample means and unbiased covariance; policies evaluated in the true market. Stress tests: ten seeds900–909 and 50,000 common paths per seed for Gaussian, iid t(8), a common-volatility GARCH-type process and stationary AR(1) coefficient .3. Mean and marginal covariance match; dependence changes conditional opportunities. All policies are frozen from the iid model; iid “pre” and “equilibrium” labels denote their origins, not dependent optimality. t(8) has finite fourth moment, unlike old t(4). Batch standard errors are descriptive Monte Carlo precision, not real-market uncertainty.

## Validation

Original verifier: ten checks. New pytest suite: 25 checks, including an independent optimiser, exhaustive return-tree moments, one-step equilibrium deviations, analytic expansion across horizons, zero-mean and one-period exceptions, time-varying lower bound, DSR continuation counterexample, hedge correction and MVPI monotonicity. An independent two-period finite-action enumeration validates the bounded Bellman reward calculation. The hedge test caught a missing factor; the final formula and output were corrected before reporting.

Scripts: python -m mvrl.verify; python -m pytest -q; python -m mvrl.experiments; PYTHONPATH=. python scripts/make_figures.py; python scripts/make_literature_ledger.py; python scripts/write_research_documents.py; python scripts/render_documents.py. Exact environment versions are recorded in results/environment.txt. Main research dependencies are NumPy/SciPy; tests require pytest; figures require matplotlib; PDFs require pdflatex (TeX Live/MiKTeX) or tectonic; the original Computer Modern article typography has been restored. LaTeX compilation and page rendering are verified separately from numerical experiments.

## Status and limitations

The central unfinished local proof, time-varying extension, finite observed-regime reference, DSR correction and missing reproducible experiments are completed at the stated scope. Unrestricted DSR, general constraints/costs/dependence, systematic publication-priority review, full published algorithm/historical replications and improved learning guarantees remain OPEN. No publication or external submission has been made.
