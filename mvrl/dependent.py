"""Finite observed-regime benchmark with joint return/next-regime branches.

Exact conditional moments; current returns can predict the next regime. No
independence shortcut is used. One risky asset, deterministic risk-free rate.
"""
from dataclasses import dataclass
import numpy as np

@dataclass
class RegimeMarket:
    transition: np.ndarray
    edge_mean: np.ndarray
    innovation_sd: float
    horizon: int=4
    riskless: float=1.02
    initial_regime: int=0
    def __post_init__(self):
        self.transition=np.asarray(self.transition,float);self.edge_mean=np.asarray(self.edge_mean,float)
        if self.transition.shape!=self.edge_mean.shape or self.transition.shape[0]!=self.transition.shape[1]:raise ValueError('square regime arrays required')
        if np.any(self.transition<0) or not np.allclose(self.transition.sum(1),1):raise ValueError('transition probabilities')
        if self.innovation_sd<=0 or self.riskless<=0 or self.horizon<1:raise ValueError('positive sd, rate and horizon required')
    @property
    def regimes(self):return len(self.transition)
    def branches(self,z):
        nxt=np.repeat(np.arange(self.regimes),2)
        prob=np.repeat(self.transition[z]/2,2)
        ret=np.repeat(self.edge_mean[z],2)+np.tile([-self.innovation_sd,self.innovation_sd],self.regimes)
        return prob,nxt,ret
    def moments(self,z):
        p,_,r=self.branches(z);m=p@r;M=p@(r*r)
        return m,M,M-m*m

@dataclass
class RegimePolicy:
    alpha: np.ndarray
    beta: np.ndarray


def moments(market,policy):
    T,K=market.horizon,market.regimes
    A=np.ones((T+1,K));a=np.zeros_like(A);B=np.ones_like(A);b=np.zeros_like(A);c=np.zeros_like(A)
    for t in range(T-1,-1,-1):
        for z in range(K):
            prob,j,r=market.branches(z);g=market.riskless+r*policy.alpha[t,z];h=r*policy.beta[t,z]
            A[t,z]=prob@(A[t+1,j]*g)
            a[t,z]=prob@(A[t+1,j]*h+a[t+1,j])
            B[t,z]=prob@(B[t+1,j]*g*g)
            b[t,z]=prob@(2*B[t+1,j]*g*h+b[t+1,j]*g)
            c[t,z]=prob@(B[t+1,j]*h*h+b[t+1,j]*h+c[t+1,j])
    return A,a,B,b,c

def evaluate(market,policy,phi,x0=1.):
    A,a,B,b,c=moments(market,policy);z=market.initial_regime
    mean=A[0,z]*x0+a[0,z];var=B[0,z]*x0*x0+b[0,z]*x0+c[0,z]-mean*mean
    return dict(mean=float(mean),variance=float(var),J=float(mean-phi*var))

def equilibrium(market,phi):
    if phi<=0:raise ValueError('positive phi required')
    T,K=market.horizon,market.regimes
    alpha=np.zeros((T,K));beta=np.zeros_like(alpha)
    # Expected future gains, sufficient for the hedge formula.
    a=np.zeros((T+1,K));hedge=np.zeros_like(beta)
    for t in range(T-1,-1,-1):
        H=market.riskless**(T-t-1)
        for z in range(K):
            p,j,r=market.branches(z);m,M,var=market.moments(z)
            covariance=p@(r*a[t+1,j])-m*(p@a[t+1,j])
            hedge[t,z]=-covariance/(H*var)
            beta[t,z]=m/(2*phi*H*var)+hedge[t,z]
            a[t,z]=H*m*beta[t,z]+p@a[t+1,j]
    return RegimePolicy(alpha,beta),hedge

def local_calibration(market,phi):
    beta=np.array([[market.moments(z)[0]/(2*phi*market.riskless**(market.horizon-t-1)*market.moments(z)[2]) for z in range(market.regimes)] for t in range(market.horizon)])
    return RegimePolicy(np.zeros_like(beta),beta)

def surrogate(market,kappa):
    if kappa<=0:raise ValueError('positive kappa required')
    T,K=market.horizon,market.regimes;s=market.riskless;dr=s-1
    pp=np.zeros((T+1,K));qq=np.zeros_like(pp);a=np.zeros((T,K));b=np.zeros_like(a)
    for t in range(T-1,-1,-1):
        for z in range(K):
            prob,j,r=market.branches(z);m,M,_=market.moments(z)
            den=kappa*M-prob@(pp[t+1,j]*r*r)
            if den<=0:raise ValueError('nonconcave Bellman problem')
            g=kappa*dr*m-s*(prob@(pp[t+1,j]*r));L=m+prob@(qq[t+1,j]*r)
            a[t,z]=-g/den;b[t,z]=L/(2*den)
            pp[t,z]=-kappa*dr*dr+s*s*(prob@pp[t+1,j])+g*g/den
            qq[t,z]=dr+s*(prob@qq[t+1,j])-g*L/den
    return RegimePolicy(a,b)

def precommitted(market,phi,x0=1.):
    T,K=market.horizon,market.regimes;s=market.riskless
    a=np.ones((T+1,K));b=np.ones_like(a);al=np.zeros((T,K));unit=np.zeros_like(al)
    for t in range(T-1,-1,-1):
        for z in range(K):
            p,j,r=market.branches(z);ar=p@(a[t+1,j]*r);ar2=p@(a[t+1,j]*r*r);br=p@(b[t+1,j]*r)
            al[t,z]=-s*ar/ar2;unit[t,z]=br/(2*phi*ar2)
            a[t,z]=s*s*(p@a[t+1,j]-ar*ar/ar2)
            b[t,z]=s*(p@b[t+1,j]-ar*br/ar2)
    e0=evaluate(market,RegimePolicy(al,np.zeros_like(unit)),phi,x0)['mean']
    e1=evaluate(market,RegimePolicy(al,unit),phi,x0)['mean']
    lam=(1+2*phi*e0)/(1-2*phi*(e1-e0))
    return RegimePolicy(al,lam*unit)

def distance(market,candidate,target,x0=1.):
    """Fixed target occupancy; exact unnormalised per-regime moments."""
    K=market.regimes;pr=np.zeros(K);f=np.zeros(K);v=np.zeros(K);z=market.initial_regime
    pr[z]=1;f[z]=x0;v[z]=x0*x0;total=0.
    for t in range(market.horizon):
        da=candidate.alpha[t]-target.alpha[t];db=candidate.beta[t]-target.beta[t]
        total+=np.sum(da*da*v+2*da*db*f+db*db*pr)
        np_,nf,nv=np.zeros(K),np.zeros(K),np.zeros(K)
        for z in range(K):
            p,j,r=market.branches(z);g=market.riskless+r*target.alpha[t,z];h=r*target.beta[t,z]
            np.add.at(np_,j,p*pr[z]);np.add.at(nf,j,p*(g*f[z]+h*pr[z]));np.add.at(nv,j,p*(g*g*v[z]+2*g*h*f[z]+h*h*pr[z]))
        pr,f,v=np_,nf,nv
    return float(np.sqrt(max(0,total)))

def deviation_value(market,continuation,phi,t,z,x,u):
    A,a,B,b,c=moments(market,continuation)
    p,j,r=market.branches(z);y=market.riskless*x+r*u
    f=p@(A[t+1,j]*y+a[t+1,j]);g=p@(B[t+1,j]*y*y+b[t+1,j]*y+c[t+1,j])
    return float(f-phi*(g-f*f))

def demo_market(persistence=.8,horizon=4,riskless=1.02):
    q=np.array([[persistence,1-persistence],[1-persistence,persistence]])
    return RegimeMarket(q,np.array([[.09,-.05],[.04,-.02]]),.12,horizon,riskless)

def hedge_reward_policy(market,phi):
    """Solve the designed conditional reward, using known equilibrium continuation.

    Explicit model-based inverse-control oracle; not a learned estimator.
    """
    eq,_=equilibrium(market,phi);_,a,_,_,_=moments(market,eq)
    beta=np.zeros_like(eq.beta)
    for t in range(market.horizon):
        H=market.riskless**(market.horizon-t-1)
        for z in range(market.regimes):
            p,j,r=market.branches(z);m,M,var=market.moments(z);nu=m*m/M
            cov=p@(r*a[t+1,j])-m*(p@a[t+1,j])
            linear=H*m-2*phi*cov*H
            quadratic=phi*(1-nu)*H*H*M
            beta[t,z]=linear/(2*quadratic)
    return RegimePolicy(np.zeros_like(beta),beta)
