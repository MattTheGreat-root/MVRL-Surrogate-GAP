# Working on this repository

Notes to self on how to keep the repo honest as the research proceeds.

## Branch per research question, not per file

One branch per thing you are trying to find out, named for the question:

    phase2/dsr-augmentation
    phase2/lambda-functional
    phase3/transaction-costs

Merge when the question has an answer, including "no". A branch that dies is
still a result; record it in `FINDINGS.md` before deleting it.

## Commit messages state the finding, not the edit

    bad:   update surrogate.py
    good:  surrogate optimum is not the equilibrium at any kappa

    bad:   fix bug in agent
    good:  drop 1/sigma^2 from the score; REINFORCE was diverging as sigma annealed

The log then reads as a research diary, which is the only version history that
is any use six months later.

## Never commit a number you have not verified

Every quantity that ends up in the README or the paper should be produced by
code in `verify.py` or by a module that runs end to end. If a number appears in
prose but nowhere in the code, it will eventually be wrong and nobody will
notice.

Add a check to `verify.py` whenever you rely on a fact. The suite is the
contract: it has caught two errors already (a vacuous Nelder-Mead test, and an
overclaim about where the reversal vanishes).

## Keep known and new results visually separate

`FINDINGS.md` has two sections: **Reproduced** and **New**. Anything in the
second section must have a note saying which literature you checked before
claiming it. Move items from New to Reproduced without embarrassment when you
find them in a paper; that is what the checking is for.

## Tag what you send to people

    git tag -a proposal-v1 -m "state of the code when the proposal was submitted"

If a supervisor reads the repo at some point, you want to be able to reconstruct
exactly what they saw.

## Keep the proposal in the repo

`proposal.tex` and the code drift apart otherwise. When a result changes what
the proposal claims, change both in the same commit. The sign of `eps_surr` is
the current example: the code shows it is not sign-definite, and the proposal
still describes it as a shortfall.

## Slow things stay out of the default path

`phase1.py` takes about ten minutes. Anything that slow should be runnable on
demand and never on import, so `verify.py` stays fast enough that you actually
run it.
