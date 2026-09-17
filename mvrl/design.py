"""Reward calibration with explicit scope; proofs in notes/PROOFS.md."""
from dataclasses import dataclass
import numpy as np
from .market import Market
from .policies import AffinePolicy
from .surrogate import surrogate_backward, policy_distance

@dataclass
class IndependentMarket:
    """Independent, not necessarily identically distributed, returns."""
    means: np.ndarray
    covariances: np.ndarray
    rates: np.ndarray
    def __post_init__(self):
        self.means=np.asarray(self.means,float)
        self.covariances=np.asarray(self.covariances,float)
        self.rates=np.asarray(self.rates,float)
        if self.means.ndim!=2: raise ValueError('means must be T by n')
        self.horizon,self.n_assets=self.means.shape
        if self.covariances.shape!=(self.horizon,self.n_assets,self.n_assets): raise ValueError('covariance shape')
        if self.rates.shape!=(self.horizon,) or np.any(self.rates<=0): raise ValueError('positive rates required')
        for c in self.covariances:
            if not np.allclose(c,c.T) or np.linalg.eigvalsh(c).min()<=0: raise ValueError('positive definite covariance required')
    def m(self,t):return self.means[t]
    def M(self,t):return self.covariances[t]+np.outer(self.m(t),self.m(t))
    def s(self,t):return float(self.rates[t])
    def nu(self,t):return float(self.m(t)@np.linalg.solve(self.M(t),self.m(t)))
    def sample_excess(self,n_paths,rng):
        return np.stack([rng.multivariate_normal(self.m(t),self.covariances[t],n_paths) for t in range(self.horizon)],axis=1)

def future_growth(market,t):return float(np.prod([market.s(j) for j in range(t+1,market.horizon)]))

def designed_policy(market,phi,estimated_means=None,estimated_covariances=None):
    if phi<=0:raise ValueError('positive phi required')
    beta=[]
    for t in range(market.horizon):
        m=market.m(t) if estimated_means is None else estimated_means[t]
        cov=market.M(t)-np.outer(market.m(t),market.m(t)) if estimated_covariances is None else estimated_covariances[t]
        beta.append(np.linalg.solve(cov,m)/(2*phi*future_growth(market,t)))
    return AffinePolicy(np.zeros_like(beta),np.array(beta))

def designed_objective(market,policy,phi,x0):
    """Exact E sum[Y-phi(1-nu)Y²], Y=H P'u, on independent returns."""
    mean,var=float(x0),0.
    total=0.
    for t in range(market.horizon):
        m,M,s=market.m(t),market.M(t),market.s(t)
        cov=M-np.outer(m,m);a,b=policy.alpha[t],policy.beta[t]
        u=a*mean+b;H=future_growth(market,t)
        total+=H*(m@u)-phi*(1-market.nu(t))*H**2*(u@M@u+var*(a@M@a))
        var=(s+m@a)**2*var+u@cov@u+var*(a@cov@a)
        mean=s*mean+m@u
    return float(total)

def common_distance(market,candidate,target,x0):
    """Fixed TARGET occupancy; nonnegative, not a global policy-space norm."""
    return policy_distance(market,target,candidate,x0)

def horizon_constant(market,phi):
    if phi<=0:raise ValueError('positive phi required')
    T=market.horizon;nu=market.nu(0)
    d=np.linalg.solve(market.M(0),market.m(0))
    return float(np.linalg.norm(d)/(2*phi*(1-nu))*np.sqrt(T*(T*T-1)/3+nu*(1-nu)*T*(T-1)/2))

def exact_raw_calibration(market,phi,x0,occupancy='target'):
    """Global quadratic minimisation in h=1/(2*kappa); h>=0 closure.

    At h=0 the reward policy is the infinite-penalty limit. Returns
    (policy, kappa, distance); kappa may be infinity, an infimum only.
    """
    from .equilibrium import solve_equilibrium
    target=solve_equilibrium(market,phi).policy
    unit=surrogate_backward(market,.5)
    # X=L+h Z; propagate E[L],E[Z],Var(L),Var(Z),Cov(L,Z).
    el,ez,vl,vz,cz=float(x0),0.,0.,0.,0.
    aa=bb=cc=0.
    for t in range(market.horizon):
        a,b=unit.alpha[t],unit.beta[t];te=target.beta[t]
        if occupancy=='target':
            v0=a*el-te;v1=b
            aa+=b@b;bb+=v0@b;cc+=v0@v0+(a@a)*vl
            m,M,s=market.m(t),market.M(t),market.s(t);cov=M-np.outer(m,m)
            vl=s*s*vl+te@cov@te;el=s*el+m@te
        elif occupancy=='candidate':
            v0=a*el-te;v1=a*ez+b
            aa+=v1@v1+(a@a)*vz
            bb+=v0@v1+(a@a)*cz
            cc+=v0@v0+(a@a)*vl
            m,M,s=market.m(t),market.M(t),market.s(t);cov=M-np.outer(m,m)
            g=s+m@a;j=m@b;va=a@cov@a;vb=b@cov@b;cab=a@cov@b
            vl,vz,cz=(g*g+va)*vl+va*el*el,(g*g+va)*vz+va*ez*ez+2*cab*ez+vb,(g*g+va)*cz+va*el*ez+cab*el
            el,ez=g*el,g*ez+j
        else:raise ValueError('occupancy must be target or candidate')
    h=max(0.,-bb/aa) if aa>0 else 0.
    policy=AffinePolicy(unit.alpha,unit.beta*h)
    distance=common_distance(market,policy,target,x0) if occupancy=='target' else policy_distance(market,policy,target,x0)
    return policy,float(1/(2*h)) if h>0 else float('inf'),distance
