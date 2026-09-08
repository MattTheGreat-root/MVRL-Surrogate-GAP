"""
A policy-gradient agent trained on the additive surrogate.

This supplies the last term of the decomposition,

    eps_learn = J(pi_r) - J(pi_hat),

and with it the first end-to-end measurement the project needs.  Every other
quantity is already exact, so whatever this agent does can be attributed to the
learning procedure rather than to the reward.

Design choices, and why
-----------------------
The policy class is the affine-Gaussian family

    pi(u | x) = Normal( alpha_t x + beta_t,  sigma_t^2 I ),

which contains the exact surrogate optimum ``pi_r`` computed in
:mod:`mvrl.surrogate`.  This is deliberate.  If the agent's policy class did not
contain ``pi_r``, any shortfall would confound approximation error with
optimisation error, and ``eps_learn`` would no longer mean what the
decomposition says it means.  Restricting to a class where the target is
representable isolates the term.

The algorithm is REINFORCE with a per-period baseline and Adam.  It is a genuine
stochastic approximation to the additive objective ``E[sum_t r_t]``, which is
the property that matters here; the proposal's argument concerns what that
objective's optimum is, not which gradient estimator finds it.  A more elaborate
method would obscure the measurement without changing the question.

Exploration noise is annealed geometrically.  With fixed noise the agent
converges to a biased point, because the reward is quadratic in the action and
Gaussian exploration therefore contributes a variance term to the expected
reward; this is a real effect and is checked in :mod:`mvrl.verify`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .evaluate import exact_affine_objective
from .market import Market
from .policies import AffinePolicy
from .surrogate import Reward


@dataclass
class TrainingHistory:
    """Objective values recorded during training, for diagnostics."""

    iterations: list[int] = field(default_factory=list)
    surrogate: list[float] = field(default_factory=list)
    true_objective: list[float] = field(default_factory=list)
    sigma: list[float] = field(default_factory=list)


@dataclass
class TrainingResult:
    policy: AffinePolicy
    history: TrainingHistory
    final_sigma: float


def _per_period_rewards(
    market: Market,
    reward: Reward,
    wealth: np.ndarray,
    actions: np.ndarray,
    excess: np.ndarray,
    t: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Realised reward and next wealth for one period, vectorised over paths."""
    gain = np.einsum("pi,pi->p", excess[:, t, :], actions)
    next_wealth = market.s(t) * wealth + gain
    increment = next_wealth - wealth
    r = increment - reward.kappa * (increment - reward.target) ** 2
    return r, next_wealth


def reinforce(
    market: Market,
    reward: Reward,
    x0: float,
    iterations: int = 4000,
    batch_size: int = 512,
    learning_rate: float = 0.02,
    sigma_initial: float = 0.5,
    sigma_final: float = 0.03,
    phi_for_diagnostics: float | None = None,
    seed: int = 0,
    record_every: int = 100,
) -> TrainingResult:
    """Train an affine-Gaussian policy on the additive reward by REINFORCE.

    ``phi_for_diagnostics`` only affects what is recorded in the history; it
    plays no part in training, since the agent never sees the mean-variance
    objective.  That separation is the whole point of the experiment.
    """
    rng = np.random.default_rng(seed)
    T, n = market.horizon, market.n_assets

    alpha = np.zeros((T, n))
    beta = np.zeros((T, n))

    # Adam state
    m_alpha = np.zeros_like(alpha)
    v_alpha = np.zeros_like(alpha)
    m_beta = np.zeros_like(beta)
    v_beta = np.zeros_like(beta)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    decay = (sigma_final / sigma_initial) ** (1.0 / max(iterations - 1, 1))
    sigma = sigma_initial
    history = TrainingHistory()

    for step in range(iterations):
        excess = market.sample_excess(batch_size, rng)

        wealth = np.full(batch_size, float(x0))
        states = np.zeros((T, batch_size))
        actions = np.zeros((T, batch_size, n))
        noises = np.zeros((T, batch_size, n))
        rewards = np.zeros((T, batch_size))

        for t in range(T):
            states[t] = wealth
            mean_action = wealth[:, None] * alpha[t][None, :] + beta[t][None, :]
            noise = rng.normal(scale=sigma, size=(batch_size, n))
            action = mean_action + noise
            actions[t] = action
            noises[t] = noise
            rewards[t], wealth = _per_period_rewards(
                market, reward, wealth, action, excess, t
            )

        # return-to-go with a per-period baseline
        returns = np.cumsum(rewards[::-1], axis=0)[::-1]
        advantages = returns - returns.mean(axis=1, keepdims=True)
        scale = advantages.std(axis=1, keepdims=True)
        advantages = advantages / np.where(scale > 1e-12, scale, 1.0)

        # The score for a Gaussian mean is noise / sigma^2.  The 1/sigma^2
        # factor is dropped here and absorbed into the learning rate: it is the
        # Fisher preconditioning for this policy class, and keeping it makes the
        # gradient variance explode as sigma anneals.  Without this the run
        # peaks early and then degrades, which is a property of the estimator
        # rather than of the objective being estimated.
        score = noises                                   # (T, B, n)
        weighted = score * advantages[:, :, None]        # (T, B, n)

        grad_alpha = (weighted * states[:, :, None]).mean(axis=1)
        grad_beta = weighted.mean(axis=1)

        current_step_size = learning_rate * (sigma / sigma_initial) ** 0.5

        # Adam ascent
        for param, grad, m_state, v_state in (
            (alpha, grad_alpha, m_alpha, v_alpha),
            (beta, grad_beta, m_beta, v_beta),
        ):
            m_state *= beta1
            m_state += (1 - beta1) * grad
            v_state *= beta2
            v_state += (1 - beta2) * grad * grad
            m_hat = m_state / (1 - beta1 ** (step + 1))
            v_hat = v_state / (1 - beta2 ** (step + 1))
            param += current_step_size * m_hat / (np.sqrt(v_hat) + eps)

        sigma *= decay

        if step % record_every == 0 or step == iterations - 1:
            from .surrogate import additive_objective

            current = AffinePolicy(alpha=alpha.copy(), beta=beta.copy())
            history.iterations.append(step)
            history.surrogate.append(
                additive_objective(market, current, reward, x0)
            )
            history.sigma.append(sigma)
            if phi_for_diagnostics is not None:
                history.true_objective.append(
                    exact_affine_objective(
                        market, current, phi_for_diagnostics, x0
                    ).objective
                )

    return TrainingResult(
        policy=AffinePolicy(alpha=alpha, beta=beta),
        history=history,
        final_sigma=sigma,
    )
