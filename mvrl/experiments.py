"""Reproduce all revised numerical claims and figure data. No network required."""
import json,sys,platform
from pathlib import Path
import numpy as np
from .lambda_functional import demo_market
from .surrogate import surrogate_backward,Reward,additive_objective
from .equilibrium import solve_equilibrium
from .evaluate import exact_affine_objective
from .policies import li_ng_precommitted,AffinePolicy
from .design import (designed_policy,designed_objective,exact_raw_calibration,common_distance,horizon_constant,IndependentMarket)
from . import dependent as dep
from .bounded import solve_tree,policy_distance as tree_distance
from .agent import reinforce

ROOT=Path(__file__).resolve().parents[1]

def metrics(mk,p,eq,phi,x0=1.):
    ev=exact_affine_objective(mk,p,phi,x0);riskfree=x0*np.prod([mk.s(t) for t in range(mk.horizon)])
    return dict(J=float(ev.objective),mean=float(ev.mean),variance=float(ev.variance),distance=common_distance(mk,p,eq,x0),sharpe=float((ev.mean-riskfree)/np.sqrt(ev.variance)) if ev.variance>1e-15 else None)

def main():
    rows=[];scaling=[];learning=[];bounded=[];dependent=[];estimation=[];robustness=[]
    for T in [1,2,4,8,16]:
      for phi in [.25,.5,1.,2.,4.,16.]:
       for s in [.98,1.,1.02,1.08]:
        mk=demo_market(s,T);eq=solve_equilibrium(mk,phi).policy
        opt,k,_=exact_raw_calibration(mk,phi,1.)
        policies={'pre':li_ng_precommitted(mk,phi,1.),'equilibrium':eq,'raw':surrogate_backward(mk,phi),'scalar':surrogate_backward(mk,phi*(1-mk.nu(0))),'best_raw':opt,'design':designed_policy(mk,phi)}
        for name,p in policies.items():rows.append(dict(T=T,phi=phi,s=s,name=name,kappa_best=k,**metrics(mk,p,eq,phi)))
    print('independent sweeps complete',flush=True)
    for T in [2,3,4,6,10,16]:
     for e in [-.001,-.0001,.0001,.001,.01]:
      mk=demo_market(1+e,T);_,k,dist=exact_raw_calibration(mk,1,1,'candidate');c=horizon_constant(mk,1)
      scaling.append(dict(T=T,e=e,c=c,distance=dist,prediction=c*abs(e),kappa=k))
    varying=IndependentMarket([[.02],[.08],[.03],[.06]],[[[.02]],[[.04]],[[.01]],[[.03]]],[1.,1.,1.,1.])
    eq=solve_equilibrium(varying,1).policy
    opt,k,_=exact_raw_calibration(varying,1,1)
    tv=[dict(name=name,**metrics(varying,p,eq,1)) for name,p in [('raw',surrogate_backward(varying,1)),('best_raw',opt),('design',designed_policy(varying,1))]]
    for q in [.5,.65,.8,.95]:
     mk=dep.demo_market(q);eq,hedge=dep.equilibrium(mk,1)
     for name,p in [('pre',dep.precommitted(mk,1)),('equilibrium',eq),('raw',dep.surrogate(mk,1)),('local_calibration',dep.local_calibration(mk,1)),('hedge_design',dep.hedge_reward_policy(mk,1))]:
      dependent.append(dict(persistence=q,name=name,distance=dep.distance(mk,p,eq),hedge_max=float(abs(hedge).max()),**dep.evaluate(mk,p,1)))
    print('dependent reference sweep complete',flush=True)
    for phi in [.5,1.,2.,4.]:
     for grid in [11,21]:
      eq=solve_tree('equilibrium',phi=phi,grid_size=grid)
      for name in ['return','log','dsr','published_quadratic','raw','scalar','design','equilibrium']:
       out=eq if name=='equilibrium' else solve_tree(name,phi=phi,grid_size=grid)
       bounded.append(dict(name=name,phi=phi,grid=grid,eta=.1,A0=.01,B0=.01,initial_weight=out['policy'][()],distance=tree_distance(out['policy'],eq['policy']),**{k:v for k,v in out.items() if k!='policy'}))
    dsr=[]
    for eta in [.02,.1,.4]:
     for A,B in [(0.,.001),(.01,.01),(.02,.002),(-.02,.002)]:
      out=solve_tree('dsr',grid_size=21,eta=eta,A0=A,B0=B)
      eq=solve_tree('equilibrium',grid_size=21)
      dsr.append(dict(eta=eta,A0=A,B0=B,initial_weight=out['policy'][()],distance=tree_distance(out['policy'],eq['policy']),**{k:v for k,v in out.items() if k!='policy'}))
    print('bounded reward and DSR comparisons complete',flush=True)
    mk=demo_market(1.02,4);phi=1.;eq=solve_equilibrium(mk,phi).policy;jpre=exact_affine_objective(mk,li_ng_precommitted(mk,phi,1),phi,1).objective;jeq=exact_affine_objective(mk,eq,phi,1).objective
    for name,k in [('raw',phi),('scalar',phi*(1-mk.nu(0))),('design',phi)]:
     reward=Reward('design' if name=='design' else 'varpen',kappa=k)
     exact=designed_policy(mk,phi) if name=='design' else surrogate_backward(mk,k)
     obj=lambda p:designed_objective(mk,p,phi,1) if name=='design' else additive_objective(mk,p,reward,1)
     jr=exact_affine_objective(mk,exact,phi,1).objective
     for seed in range(5):
      fit=reinforce(mk,reward,1.,iterations=4000,batch_size=512,seed=seed,phi_for_diagnostics=phi,record_every=200)
      mt=metrics(mk,fit.policy,eq,phi)
      learning.append(dict(name=name,seed=seed,iterations=4000,**mt,surrogate_regret=obj(exact)-obj(fit.policy),incons=jpre-jeq,surr=jeq-jr,learn=jr-mt['J'],total=jpre-mt['J'],history=fit.history.__dict__))
     print('learning',name,'complete',flush=True)
    for N in [50,200,1000,10000]:
     for seed in range(30):
      rng=np.random.default_rng(1000+seed);sample=rng.multivariate_normal(mk.mu_excess,mk.cov_excess,N);m=sample.mean(0);cov=np.cov(sample,rowvar=False)
      estimated=type(mk)(m,cov,mk.riskless,mk.horizon)
      for name,p in [('raw',surrogate_backward(estimated,phi)),('scalar',surrogate_backward(estimated,phi*(1-estimated.nu(0)))),('design',designed_policy(estimated,phi))]:estimation.append(dict(N=N,seed=seed,name=name,**metrics(mk,p,eq,phi)))
    # Matched first and second moments. Repeated independent batches provide
    # uncertainty for J differences. t(8), not t(4), gives finite fourth moment.
    policies={'pre':li_ng_precommitted(mk,phi,1),'equilibrium':eq,'raw':surrogate_backward(mk,phi),'design':designed_policy(mk,phi)}
    from .phase5 import sample_misspecified,evaluate_paths
    for kind in ['gaussian','student-t','vol-clustering','momentum']:
     for seed in range(10):
      paths=sample_misspecified(mk,50000,np.random.default_rng(seed+900),kind)
      for name,p in policies.items():
       w=evaluate_paths(mk,p,paths,1.)
       robustness.append(dict(kind=kind,seed=seed,name=name,J=float(w.mean()-phi*w.var(ddof=1)),mean=float(w.mean()),variance=float(w.var(ddof=1))))
    market_sensitivity=[]
    from .market import Market
    base=demo_market(1.02,4)
    for mean_scale in [.5,1.,1.5]:
     for vol_scale in [.75,1.,1.25]:
      for corr_scale in [0.,1.,1.5]:
       cov=base.cov_excess.copy();diag=np.diag(np.diag(cov));cov=vol_scale**2*(diag+corr_scale*(cov-diag))
       mk=Market(base.mu_excess*mean_scale,cov,1.02,4);eq=solve_equilibrium(mk,1).policy
       for name,p in [('raw',surrogate_backward(mk,1)),('scalar',surrogate_backward(mk,1-mk.nu(0))),('design',designed_policy(mk,1))]:
        market_sensitivity.append(dict(mean_scale=mean_scale,vol_scale=vol_scale,corr_scale=corr_scale,name=name,**metrics(mk,p,eq,1)))
    from .mvpi import solve as solve_mvpi
    mvpi=[]
    for s in [.98,1.,1.02,1.08]:
     mk=demo_market(s,4);eq=solve_equilibrium(mk,1).policy;p,meta=solve_mvpi(mk,1)
     mvpi.append(dict(s=s,**metrics(mk,p,eq,1),**meta))
    result=dict(mvpi=mvpi,market_sensitivity=market_sensitivity,metadata=dict(seed_protocol='explicit per section',python=sys.version,numpy=np.__version__,platform=platform.platform(),date='2026-09-21',occupancy='fixed equilibrium except scaling',bounded='exact finite action tree; adapted rewards, no historical or algorithm reproduction'),independent=rows,scaling=scaling,time_varying=tv,dependent=dependent,bounded=bounded,dsr=dsr,learning=learning,estimation=estimation,robustness=robustness)
    (ROOT/'results'/'experiments.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print('saved results/experiments.json',flush=True)

if __name__=='__main__':main()
