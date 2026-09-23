# Portfolio reward design: current research package

Revised **22 September 2026**. Start with **proposal.pdf**, then FINDINGS.md. The project now has scoped proofs, exact independent/dependent benchmarks, published reward adaptations, a separate learning-error experiment, saved outputs and ten reproducible figure pairs. It does **not** claim universal financial improvement or established publication priority.

## Main result

The raw squared-increment reward has an irreducible policy mismatch at nonzero interest. A terminal-unit excess-gain reward recovers the mean–variance equilibrium under independent returns and unconstrained risky dollar holdings. The fixed-horizon mismatch coefficient is proved. Existing MVPI already yields the zero-interest calibration; this is explicitly treated as prior art.

## Reading map

- `proposal.pdf`, `proposal.md`, `proposal.tex`: results-based proposal with ten figures and comparative tables.
- `paper.pdf`, `paper.md`, `paper.tex`: revised draft including proof appendix.
- `FINDINGS.md`: authoritative result/status and corrections register.
- `notes/INITIAL_AUDIT.md`: original state and complete read-before-edit inventory.
- `notes/PROOFS.md`: derivations, assumptions, exceptions and known-results separation.
- `literature/LITERATURE_LEDGER.md`, `ledger.json`, `references.bib`: twelve exact source records; eleven primary versions inspected, one abstract-only. Source PDF versions and hashes recorded.
- `IMPLEMENTATION_REPORT.md`: exact models, seeds, budgets, adaptation choices and limitations.
- `results/experiments.json`: all benchmark outputs, seed results and training histories.
- `results/tests.txt`, `legacy_verification.txt`: 25 new tests and 10 original checks.
- `figures/`: ten PDF/PNG figure pairs, regenerated from data.
- `WORKLOG.md`: append-only historical diary; its latest dated entry supersedes earlier conclusions.
- `archive/2026-09-21-before-audit/`: delivered documents and standalone scripts preserved before editing.

## Reproduce

Python 3.10+; install `requirements.txt` for research/figures/documents and `requirements-dev.txt` for tests. Run from this directory:

```sh
python -m mvrl.verify
python -m pytest -q
python -m mvrl.experiments
PYTHONPATH=. python scripts/make_figures.py
python scripts/make_literature_ledger.py
PYTHONPATH=. python scripts/write_research_documents.py
python scripts/render_documents.py
python scripts/check_outputs.py
```

`experiments` includes exhaustive finite trees and 15 training runs; it is not run during import or ordinary tests. Runtime/version evidence is in `results/environment.txt`. Fetching source PDFs is optional: `literature/download_sources.py`; some public hosts block downloads, with access limitations recorded in the ledger. No network is needed to reproduce experiments from the saved package.

PDFs are compiled from genuine LaTeX using the original article formatting: 11pt A4, one-inch margins, T1-encoded Computer Modern, the original titles/author blocks, and standard mathematical typography. Install `pdflatex` (TeX Live/MiKTeX) or `tectonic`, then run `python scripts/render_documents.py`; use `--engine /path/to/compiler` if needed. To generate sources without compiling, add `--sources-only`. Direct compilation is also supported: run `pdflatex proposal.tex` and `pdflatex paper.tex` three times each from the package directory. Figures are embedded as vector PDFs.

Shared prose/data remain in `documents.json` and its writer; typeset proofs are in `latex/proofs.tex` and correspond to `notes/PROOFS.md`. `scripts/render_documents.py` creates complete standalone `.tex` files. The root delivery sources use adjusted figure paths. The root `latex-original-format.zip` is a small standalone source bundle suitable for upload to Overleaf (select `proposal.tex` or `paper.tex`, compiler pdfLaTeX). The previous report-style PDFs and renderer are preserved in `archive/2026-09-22-report-layout/`; they are superseded.

## Boundaries

Independent analytic theory uses dollar holdings without leverage or solvency constraints. The long-only tree is a separate finite-action problem. Published rewards are **adapted**, not historical/PPO/DDPG replications. Dependent reward redesign uses a known continuation hedge. Five learning seeds provide preliminary diagnostics only and do not show improved learned J. Unrestricted DSR, transaction costs, general constraints/dependence and systematic novelty verification remain open.

Root-level duplicates are synchronised delivery copies; this directory is authoritative. The original root ZIP is historical, not the current results package.
