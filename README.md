# Codes and implementations for the paper on reward design in portfolio reinforcement learning

This repository contains the **codes, mathematical implementations, numerical experiments, tests, figures, literature evidence, and reproducible documents** for the research paper *Reward Design That Targets the Intended Portfolio Policy*. It studies a basic but often overlooked question in reinforcement learning for asset allocation:

> Does the reward optimised by an RL agent actually induce the portfolio policy intended by the investor's terminal objective?

The project compares additive per-period rewards with terminal mean–variance portfolio objectives. It separates three issues that are often mixed together: the choice of financial target, the policy induced by a reward, and the error introduced by learning. The repository includes exact independent- and dependent-market benchmarks, finite constrained problems, adapted published reward formulas, policy-gradient experiments, analytical proofs, saved outputs, and the complete paper and proposal.

The current package was audited and revised on **23 September 2026**. Its claims are deliberately scoped. It does not claim universal trading improvement, superiority of one RL algorithm, or established publication priority.

## Research question

A reinforcement-learning agent maximises the reward supplied to it. In portfolio applications, that reward is commonly a sum of returns, quadratic wealth-increment terms, log growth, or a differential Sharpe signal. These quantities need not induce the same policy as the investor's intended terminal objective.

The main benchmark in this project is

```text
X_{t+1} = s_t X_t + P_t' u_t
J(pi)   = E[X_T] - phi Var(X_T),    phi > 0,
```

where `X_t` is wealth, `s_t` is the deterministic risk-free gross return, `P_t` is the risky excess-return vector, and `u_t` is the vector of risky dollar holdings. The project distinguishes the initial-date precommitted mean–variance optimum from the time-consistent equilibrium policy. The redesigned reward targets the equilibrium policy; it is not intended to reproduce the precommitted policy.

## Main result

For independent returns and unconstrained risky dollar holdings, the commonly used raw quadratic increment reward

```text
r_t = Delta X_t - kappa (Delta X_t)^2
```

cannot generally recover the terminal mean–variance equilibrium with one scalar `kappa` when the risk-free return differs from one. Its best local policy mismatch has an explicit coefficient for every fixed horizon.

The project gives a constructive correction. Define the excess gain in terminal units by

```text
Y_t = H_t P_t' u_t,
H_t = product of future risk-free gross returns,
```

and use the date-specific reward

```text
r_design,t = Y_t - phi (1 - nu_t) Y_t^2,
nu_t       = m_t' (Sigma_t + m_t m_t')^{-1} m_t.
```

Under the stated independent-return and unconstrained-action assumptions, this reward recovers the equilibrium policy exactly. At zero interest, the scalar calibration reduces to `kappa = phi(1-nu)`. That zero-interest fixed point is not claimed as novel: the adapted existing MVPI method reaches the same calibration.

## What is established

- The Li–Ng precommitted and consistent-planning equilibrium references are implemented and independently checked.
- The raw quadratic-increment Bellman recursion and zero-interest calibration are proved.
- For every fixed horizon, the locally minimal nonzero-interest mismatch and its coefficient are proved.
- The terminal-unit, date-specific reward exactly recovers equilibrium under independent returns and unconstrained dollar holdings.
- A common scalar penalty has an explicit weighted-dispersion lower bound when moments vary over time.
- In a finite observed-regime model, dependence creates an intertemporal hedge term; an oracle hedge correction recovers equilibrium.
- Differential Sharpe reward optimisation is well posed on the bounded finite tree, but its local quadratic expansion does not make it dynamically equivalent to a fixed variance penalty.
- The exact results are exercised over 120 horizon/risk-aversion/rate configurations and 27 moment configurations.
- A separate 25-test suite covers analytical identities, exhaustive tree calculations, deviations, edge cases, dependent returns, DSR, and MVPI monotonicity.

## Important negative results and boundaries

- Squared increments penalise a second moment, not a variance.
- Additivity alone does not imply time inconsistency; an additive reward has a Bellman formulation on an adequate state.
- Exact reward-policy recovery does not automatically survive wealth-dependent constraints.
- Better policy alignment does not necessarily mean a larger initial-date `J`, because equilibrium and precommitment are different targets.
- A higher Sharpe ratio need not rank policies in the same order as terminal mean–variance utility.
- The five-seed learning experiment is preliminary. At its current budget, the redesigned reward improves average alignment but does not improve learned mean–variance performance.
- Frozen policies designed for independent returns can reverse their ranking under serial dependence.
- Published rewards are adapted to common benchmark environments here; their original datasets and PPO, DDPG, SARSA, RRL, or other training pipelines are not replicated.
- Transaction costs, unrestricted DSR, continuous constrained problems, and general dependent multi-asset markets remain open.

The authoritative status and corrections register is [`FINDINGS.md`](FINDINGS.md). Proof assumptions and exceptions are recorded in [`notes/PROOFS.md`](notes/PROOFS.md).

## Repository contents

### Research code

| Path | Purpose |
|---|---|
| `mvrl/market.py` | Independent multi-period market and moment definitions |
| `mvrl/policies.py` | Precommitted policy, myopic policy, and affine policy container |
| `mvrl/equilibrium.py` | Time-consistent equilibrium by backward recursion |
| `mvrl/evaluate.py` | Exact policy evaluation and Monte Carlo support |
| `mvrl/surrogate.py` | Exact and independently optimised surrogate policies |
| `mvrl/design.py` | Terminal-unit reward construction and calibration |
| `mvrl/dependent.py` | Observed-regime dependent-return model and hedge correction |
| `mvrl/bounded.py` | Exhaustive long-only finite-tree benchmark |
| `mvrl/mvpi.py` | Finite-horizon adaptation of mean–volatility policy iteration |
| `mvrl/dsr.py` | Differential Sharpe calculations used by current experiments |
| `mvrl/agent.py` | Affine Gaussian REINFORCE-style learner |
| `mvrl/experiments.py` | Complete reproducible experiment driver |
| `mvrl/verify.py` | Ten legacy correctness checks |
| `mvrl/gap.py` | Independent-market reward-gap comparison |

The earlier standalone phase scripts remain available where useful, while superseded versions are retained under `archive/` for provenance.

### Evidence and documents

| Path | Purpose |
|---|---|
| `proposal.pdf`, `proposal.md`, `proposal.tex` | Results-based research proposal with comparative tables and figures |
| `paper.pdf`, `paper.md`, `paper.tex` | Current 17-page paper draft and source forms |
| `figures/` | Ten reproducible figures in both vector PDF and PNG formats |
| `results/experiments.json` | Saved benchmark values, sweeps, seeds, and training histories |
| `results/tests.txt` | Recorded result of the 25-test suite |
| `results/legacy_verification.txt` | Recorded result of the original ten checks |
| `literature/LITERATURE_LEDGER.md` | Source-by-source comparison with exact version locators |
| `literature/ledger.json` | Machine-readable literature ledger |
| `literature/references.bib` | Bibliography used by the documents |
| `IMPLEMENTATION_REPORT.md` | Models, parameters, seeds, budgets, adaptations, and limitations |
| `WORKLOG.md` | Append-only research and correction history |
| `archive/` | Preserved pre-audit and superseded document versions |

## Installation

Python 3.10 or newer is required. No GPU, accelerator, API key, or cloud service is needed.

```bash
git clone https://github.com/MattTheGreat-root/MVRL-Surrogate-GAP.git
cd MVRL-Surrogate-GAP

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

`requirements.txt` contains the numerical, plotting, and document-generation dependencies. `requirements-dev.txt` includes those requirements and adds the test and PDF-inspection tools.

## Quick verification

Run the original analytical and numerical checks first:

```bash
python -m mvrl.verify
```

The command must finish with ten `PASS` results and `overall PASS`. Then run the expanded research tests:

```bash
python -m pytest -q
```

Expected result:

```text
25 passed
```

Finally, validate the saved numerical outputs and rendered documents:

```bash
python scripts/check_outputs.py
```

The check confirms the saved experiment counts and central identities, then inspects the proposal and paper PDFs. The current package contains a 12-page proposal and a 17-page paper.

## Reproducing the complete project

The following sequence rebuilds the principal outputs from the repository root:

```bash
# 1. Baseline implementation checks
python -m mvrl.verify
python -m pytest -q

# 2. Complete numerical experiment package
python -m mvrl.experiments

# 3. Figures and evidence tables
PYTHONPATH=. python scripts/make_figures.py
python scripts/make_literature_ledger.py

# 4. Markdown and LaTeX research documents
PYTHONPATH=. python scripts/write_research_documents.py
python scripts/render_documents.py

# 5. Final cross-file and PDF verification
python scripts/check_outputs.py
```

The experiment driver includes exhaustive finite trees, parameter sweeps, estimation studies, stress simulations, and 15 learning runs. It is intentionally not executed during import or ordinary unit tests. Saved results are provided so readers can inspect the claims without rerunning the longer learning experiments.

No network connection is required to reproduce the supplied numerical results. Downloading literature sources again is optional; source versions and hashes are already recorded, and some publisher hosts may block automated downloads.

## Rebuilding the paper and proposal

The root PDFs are compiled from genuine LaTeX with 11-point A4 article formatting, one-inch margins, T1-encoded Computer Modern typography, vector figures, and complete proof material.

Install `pdflatex` through TeX Live/MacTeX/MiKTeX, or install `tectonic`, and run:

```bash
python scripts/render_documents.py
```

To choose a compiler explicitly:

```bash
python scripts/render_documents.py --engine /path/to/pdflatex
```

To regenerate source files without compiling PDFs:

```bash
python scripts/render_documents.py --sources-only
```

The document text and structured results originate in `documents.json`; the typeset proof material is in `latex/proofs.tex`. The previous report-style layout is preserved in `archive/2026-09-22-report-layout/`.

## Reference benchmark

The primary independent benchmark uses three risky assets, four periods, `phi=1`, `s=1.02`, and initial wealth one. Policy distance is evaluated using the fixed equilibrium occupancy.

| Policy | Distance to equilibrium | Terminal `J` | Terminal excess Sharpe |
|---|---:|---:|---:|
| Precommitted reference | 3.312087 | 1.425799 | 1.171951 |
| Equilibrium reference | 0.000000 | 1.323644 | 0.982266 |
| Raw reward, `kappa=phi` | 0.797127 | 1.316580 | 0.987285 |
| Scalar zero-interest calibration | 0.195347 | 1.326115 | 0.987292 |
| Best scalar in the raw family | 0.186435 | 1.326093 | 0.987292 |
| Terminal-unit design | numerical zero | 1.323644 | 0.982266 |

The table illustrates why the target must be declared explicitly. The precommitted policy maximises initial `J`, the equilibrium policy is the time-consistent reference, and the reward design is evaluated by whether it induces that equilibrium. A scalar surrogate can have a slightly larger initial `J` than equilibrium while still being misaligned with the equilibrium target.

## Learning experiment

The learning study uses the same affine Gaussian policy class, sample budget, and seed schedule for the raw, scalar-calibrated, and terminal-unit rewards. Each configuration uses 4,000 iterations, batches of 512, five seeds, Adam with learning rate `0.02`, normalised advantages, and a common exploration schedule. Deterministic mean policies are evaluated by exact moments, so final reported `J` values do not contain test-set Monte Carlo noise.

| Learned reward | Mean distance ± SD | Mean `J` ± SD | Mean surrogate regret |
|---|---:|---:|---:|
| Raw | 0.8638 ± 0.1890 | 1.305628 ± 0.001303 | 0.003584 |
| Scalar | 0.8312 ± 0.0978 | 1.302556 ± 0.003415 | 0.005381 |
| Terminal-unit design | 0.8079 ± 0.0807 | 1.300671 ± 0.003543 | 0.005640 |

These five-seed results diagnose learning error but do not establish algorithmic superiority. The exact redesigned reward is aligned, yet the learned policy retains substantial error at the current budget.

## Dependent and constrained benchmarks

The observed-regime experiment shows why a purely local moment correction is insufficient under dependence: current returns covary with the continuation value. A reward containing the known continuation hedge recovers the finite-regime equilibrium exactly, but this is an oracle construction rather than a model-free learning procedure.

The bounded benchmark uses one risky asset plus cash, a three-period binary return tree, and long-only portfolio-weight grids. The redesigned reward improves policy alignment in the reported configuration but does not recover the constrained equilibrium exactly and does not dominate every alternative in terminal `J`. This is expected: wealth-dependent feasible dollar holdings break the unconstrained proof.

## Reproducibility conventions

- Exact calculations and Monte Carlo or learning results are labelled separately.
- `PROVED` means derived under the documented assumptions, not independently peer reviewed.
- `NUMERICALLY VERIFIED` means checked in the specified finite experiment.
- `PRELIMINARY` denotes limited seeds, sampling, or model scope.
- `OPEN` denotes an unresolved question.
- Policy distances use the declared reference occupancy and are discrepancies, not globally symmetric norms.
- Sharpe values use terminal excess gain over compounded risk-free wealth and are not annualised.
- Published reward comparisons reproduce reward shapes on common benchmarks, not the papers' complete algorithms or historical results.

Exact environment information from the audited run is stored in `results/environment.txt`. File hashes and delivery integrity information are stored in `results/delivery_manifest.json`.

## Suggested reading order

1. `proposal.pdf` for the motivation, comparisons, and visual overview.
2. `FINDINGS.md` for the authoritative claim-status table and corrections.
3. `paper.pdf` for the full argument and proof appendix.
4. `notes/PROOFS.md` for derivations and edge cases.
5. `IMPLEMENTATION_REPORT.md` for experimental protocols and parameter choices.
6. `literature/LITERATURE_LEDGER.md` for prior-work positioning.
7. `results/experiments.json` for the complete saved numerical record.

## Current status

The repository is a reproducible research package and working paper, not a finished production trading system. It does not connect to a broker, download market prices for the experiments, or place trades. The exact results are mathematical benchmark results under stated models; they should not be interpreted as investment advice or expected live-market performance.

The most important remaining work is:

1. general dependent, multi-asset reward design without an oracle continuation;
2. continuous constrained portfolios, leverage limits, and transaction costs;
3. unrestricted and infinite-horizon DSR analysis;
4. larger and properly tuned learning comparisons with stronger uncertainty reporting;
5. broader systematic novelty and publication-priority review;
6. historical replications of complete published algorithms rather than reward-only adaptations.

## License and contribution notes

No standalone open-source license is currently included. Treat the repository as all-rights-reserved research material unless the author adds a license. Before proposing a change, read `CONTRIBUTING.md`, `FINDINGS.md`, and the latest entry in `WORKLOG.md`. New numerical claims should include a generator, saved output, an explicit status label, and a test or independent check.

For academic use, cite the accompanying paper once its final bibliographic record is available. Until then, cite the repository and commit hash used for the analysis.
