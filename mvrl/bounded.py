"""Exact finite-tree published REWARD adaptations, not algorithm replications.

One risky asset, strictly positive gross returns, cash, no costs. All policies
use the identical finite long-only weight grid and full return history. DSR
optimisation includes A,B in its recursion. Terminal MV equilibrium solved on
the SAME grid. Published data/performance and PPO/DDPG are not reproduced.
"""
import numpy as np
from .dsr import dsr_step


def solve_tree(kind,phi=1.,T=3,grid_size=11,eta=.1,A0=.01,B0=.01,s=1.02,returns=(-.18,.30),probs=(.5,.5)):
    actions=np.linspace(0,1,grid_size);r=np.array(returns);p=np.array(probs)
    m=p@r;M=p@(r*r);nu=m*m/M
    def rec(t,x,A,B):
        if t==T:return 0.,x,x*x,{},[]
        best=None
        for w in actions:
            us=w*x;ys=x*(s+r*w)
            vals=[];means=[];seconds=[];maps=[]
            for j,y in enumerate(ys):
                rho=y/x-1
                D,aa,bb=dsr_step(A,B,rho,eta)
                val,mean,second,mp,_=rec(t+1,y,aa,bb)
                if kind=='return':reward=y-x
                elif kind=='log':reward=np.log(y/x)
                elif kind=='dsr':reward=D
                elif kind=='published_quadratic':reward=(y-x)-.05*(y-x)**2
                elif kind=='raw':reward=(y-x)-phi*(y-x)**2
                elif kind=='scalar':reward=(y-x)-phi*(1-nu)*(y-x)**2
                elif kind=='design':
                    gain=s**(T-t-1)*r[j]*us;reward=gain-phi*(1-nu)*gain*gain
                elif kind=='equilibrium':reward=0.
                else:raise ValueError(kind)
                vals.append(reward+val);means.append(mean);seconds.append(second);maps.append(mp)
            mean=p@means;second=p@seconds
            score=mean-phi*(second-mean*mean) if kind=='equilibrium' else p@vals
            if best is None or score>best[0]+1e-12:
                policy={():float(w)}
                for j,mp in enumerate(maps):policy.update({(j,)+key:value for key,value in mp.items()})
                best=(score,mean,second,policy,[])
        return best
    score,mean,second,policy,_=rec(0,1.,A0,B0)
    variance=second-mean*mean
    return dict(reward=float(score),mean=float(mean),variance=float(variance),J=float(mean-phi*variance),sharpe=float((mean-s**T)/np.sqrt(variance)) if variance>1e-15 else None,policy=policy)


def policy_distance(candidate,target,T=3,s=1.02,returns=(-.18,.30),probs=(.5,.5)):
    def rec(hist,x,prob):
        if len(hist)==T:return 0.
        wc,wt=candidate[hist],target[hist]
        total=prob*x*x*(wc-wt)**2
        return total+sum(rec(hist+(j,),x*(s+returns[j]*wt),prob*probs[j]) for j in range(2))
    return float(np.sqrt(rec((),1.,1.)))
