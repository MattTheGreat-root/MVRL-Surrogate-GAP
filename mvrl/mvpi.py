"""Finite-horizon, uniform-time adaptation of Zhang et al. MVPI Algorithm 1.

The paper uses discounted infinite-horizon occupancy. Here R is a wealth
increment at a uniformly selected date. The inner quadratic MDP is exact;
convergence below establishes a fixed point, not a global MV optimum.
"""
import numpy as np
from .surrogate import surrogate_backward,additive_objective,Reward
from .policies import AffinePolicy
from .evaluate import exact_affine_objective

def solve(market,phi,x0=1.,tol=1e-12,max_iter=10000):
    unit=surrogate_backward(market,phi);y=0.;trace=[]
    for i in range(max_iter):
        p=AffinePolicy(unit.alpha,unit.beta*(1+2*phi*y))
        ev=exact_affine_objective(market,p,phi,x0)
        ynew=(ev.mean-x0)/market.horizon
        score=additive_objective(market,p,Reward('varpen',phi),x0)/market.horizon+phi*ynew*ynew
        trace.append(float(score))
        if abs(ynew-y)<tol:return p,dict(iterations=i+1,residual=float(abs(ynew-y)),mean_volatility=float(score),trace=trace)
        y=ynew
    raise RuntimeError('MVPI finite-horizon adaptation did not converge')
