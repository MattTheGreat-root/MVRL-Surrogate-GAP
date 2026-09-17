"""Known reference-value gap. Full signed decomposition: mvrl.experiments.
The surrogate and learning J-differences can each be negative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .equilibrium import solve_equilibrium
from .evaluate import exact_affine_objective
from .market import Market
from .policies import MyopicPolicy, li_ng_precommitted


@dataclass
class DecompositionReport:
    J_precommitted: float
    J_equilibrium: float
    eps_incons: float
    J_myopic_mc: float | None = None


def report(market: Market, phi: float, x0: float) -> DecompositionReport:
    precommitted = li_ng_precommitted(market, phi, x0)
    equilibrium = solve_equilibrium(market, phi).policy

    j_pre = exact_affine_objective(market, precommitted, phi, x0).objective
    j_eq = exact_affine_objective(market, equilibrium, phi, x0).objective

    return DecompositionReport(
        J_precommitted=j_pre,
        J_equilibrium=j_eq,
        eps_incons=j_pre - j_eq,
    )


def main():
    from .lambda_functional import demo_market
    r=report(demo_market(1.02,4),1.,1.)
    print(r)
    print('Full reward/learning decomposition and nonnegative surrogate regret: results/experiments.json')

if __name__ == '__main__': main()
