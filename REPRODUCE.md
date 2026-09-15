# Reproducing the results locally

Everything here runs on a laptop with numpy and scipy. No GPU, no
accelerator, no cloud. The RL training is a few thousand REINFORCE iterations on
a policy with a few dozen parameters.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10 or later. Developed against numpy 2.4 and scipy 1.17; earlier
versions of both should work, but if a check fails, version skew is the first
thing to rule out.

On Apple silicon nothing special is needed — numpy's accelerate backend is fine
and the problems are too small for it to matter either way.

## What to run, and what it should print

| command | time | what it checks |
|---|---|---|
| `python -m mvrl.verify` | ~4 s | ten correctness checks; **all must PASS** |
| `python -m mvrl.inconsistency` | <1 s | time-inconsistency demonstration |
| `python -m mvrl.decomposition` | <1 s | `eps_incons` and its sensitivity |
| `python -m mvrl.lambda_functional` | ~1 s | Theorem 4, Prop 6, Theorem 8 |
| `python -m mvrl.gap` | ~90 s | the surrogate gap; uses the independent optimiser |
| `python -m mvrl.phase1` | ~10 min | full decomposition, RL training, phi sweep |

Start with `verify`. If it passes, everything else is downstream of machinery
that is known good.

## Numbers to check against

These are the values in the paper. Deterministic quantities should match to the
digits shown; anything involving training will differ in the last places and is
marked.

**Reference policies** (`decomposition`, demo market, `phi = 1`)

    J(pi_pre)  = 1.425799294
    J(pi_eq)   = 1.323643886
    eps_incons = 0.102155409

**Theorem 4** (`lambda_functional`) — at `s = 1`, distance between the surrogate
optimum at `kappa = phi(1-nu)` and the equilibrium policy should be `~1e-16`,
for every horizon and asset count shown. If any row prints something like `1e-3`,
something is wrong.

**Theorem 8** (`lambda_functional`) — predicted and measured `c` agreeing to
better than `0.2%`, fitted exponent `1.0000`.

**The surrogate gap** (`gap`)

    kappa*                = 0.808485      (= phi(1-nu), nu = 0.191559)
    J(pi_r) at kappa*     = 1.326118461
    distance to pi_eq     = 0.192068

**Sharpe misranking** (`gap`) — the `kappa = phi` row must show a *higher*
Sharpe (1.1955) and a *lower* `J` (1.316580) than the equilibrium row (1.1501,
1.323644). This is the reversal.

**Full decomposition** (`phase1`) — *training-dependent, expect the last two or
three digits to differ*

    J(pi_hat)  ~ 1.3079      (5 seeds, sd ~0.003)
    eps_learn  ~ 0.0087
    sum        = J(pi_pre) - J(pi_hat)   must close to ~1e-16

The identity closing is not training-dependent and must hold exactly.

**Crossover** (`phase1`) — `eps_incons` and `eps_surr` should cross between
`phi = 8` and `phi = 16`. The exact location is market-specific; the crossing
existing is the claim.

## If something disagrees

A mismatch in a deterministic quantity is worth chasing, not explaining away.
The order to check:

1. numpy/scipy versions.
2. Whether `verify` passes — if not, the reference machinery is the problem and
   nothing downstream means anything.
3. Whether the disagreement is in a training-dependent number. Those move with
   seeds and platform; the deterministic ones should not.

If a deterministic number differs in more than the last two digits, that is a
finding and should go in `FINDINGS.md`, not be worked around.

## What running this does not settle

The compute is not where the remaining risk is. Re-running confirms the
implementation does what the paper says; it cannot tell you whether
`kappa* = phi(1-nu)` is already in the literature, whether Theorem 8's error
terms are properly controlled, or what `c_T` is for `T >= 3`. Those are reading
and mathematics.
