import numpy as np
import pytest
from scipy.optimize import minimize, minimize_scalar
from mvrl.market import Market
from mvrl.lambda_functional import demo_market
from mvrl.surrogate import surrogate_backward,policy_distance,additive_objective,Reward
from mvrl.design import IndependentMarket,designed_policy,designed_objective,horizon_constant,exact_raw_calibration
from mvrl.equilibrium import solve_equilibrium
from mvrl.evaluate import exact_affine_objective
from mvrl.policies import AffinePolicy,li_ng_precommitted
from mvrl import dependent as dep
from mvrl.dsr import dsr_step,ewma_variance

@pytest.mark.parametrize('T',[2,3,5,10])
@pytest.mark.parametrize('phi',[.5,3.])
def test_general_horizon_expansion(T,phi):
    e=1e-5;mk=demo_market(1+e,T)
    _,_,dist=exact_raw_calibration(mk,phi,1.,'candidate')
    assert abs(dist/e/horizon_constant(mk,phi)-1)<.002

@pytest.mark.parametrize('occupancy',['target','candidate'])
def test_exact_calibration_global(occupancy):
    mk=demo_market(1.06,5);target=solve_equilibrium(mk,2).policy
    p,k,dist=exact_raw_calibration(mk,2,1,occupancy)
    f=lambda h: policy_distance(mk,target,surrogate_backward(mk,1/(2*h)),1) if occupancy=='target' else policy_distance(mk,surrogate_backward(mk,1/(2*h)),target,1)
    fit=minimize_scalar(f,bounds=(.001,10),method='bounded',options={'xatol':1e-12})
    assert abs(fit.fun-dist)<1e-8
    assert abs(fit.x-1/(2*k))<1e-6

@pytest.mark.parametrize('s',[.95,1.,1.07])
def test_design_matches_equilibrium(s):
    mk=demo_market(s,5);p=designed_policy(mk,2);eq=solve_equilibrium(mk,2).policy
    assert np.max(np.abs(p.beta-eq.beta))<1e-10
    rng=np.random.default_rng(5);base=designed_objective(mk,p,2,1)
    for _ in range(20):
        other=AffinePolicy(rng.normal(0,.1,p.alpha.shape),p.beta+rng.normal(0,.1,p.beta.shape))
        assert designed_objective(mk,other,2,1)<base


def test_time_varying_independent():
    mk=IndependentMarket([[.02],[.08],[.03]],[[[.02]],[[.04]],[[.01]]],[1.01,1.03,.99])
    p=designed_policy(mk,1);eq=solve_equilibrium(mk,1).policy
    assert np.allclose(p.beta,eq.beta,atol=1e-12)


def test_known_value_identities():
    for T in [1,3,8]:
        mk=demo_market(1.02,T);nu=mk.nu(0);phi=2.;x=1.3
        for p,target in [(li_ng_precommitted(mk,phi,x),mk.riskless**T*x+((1-nu)**(-T)-1)/(4*phi)),(solve_equilibrium(mk,phi).policy,mk.riskless**T*x+T*nu/(4*phi*(1-nu)))]:
            assert abs(exact_affine_objective(mk,p,phi,x).objective-target)<1e-10


def test_zero_mean_and_one_period_exception():
    mk=Market([0.],[[.02]],1,3)
    assert np.all(surrogate_backward(mk,.01).beta==0)
    assert np.all(surrogate_backward(mk,30).beta==0)
    mk=Market([.1],[[.04]],.5,1);phi=100.;x0=1.
    assert (mk.s(0)-1)*x0+1/(2*phi*(1-mk.nu(0)))<0
    assert surrogate_backward(mk,1e8).act(0,x0)[0]>solve_equilibrium(mk,phi).policy.act(0,x0)[0]


def test_dependent_equilibrium_definition():
    mk=dep.demo_market();eq,hedge=dep.equilibrium(mk,1)
    assert np.max(abs(hedge))>.01
    for t in range(mk.horizon):
        for z in range(mk.regimes):
            for x in [.5,1.,2.]:
                opt=minimize_scalar(lambda u:-dep.deviation_value(mk,eq,1,t,z,x,u),bounds=(-10,10),method='bounded')
                assert abs(opt.x-eq.beta[t,z])<1e-5
                assert abs(-opt.fun-dep.deviation_value(mk,eq,1,t,z,x,eq.beta[t,z]))<1e-10


def test_dependent_precommit_independent_optimiser():
    mk=dep.demo_market(horizon=2);pre=dep.precommitted(mk,1);target=dep.evaluate(mk,pre,1)['J']
    def objective(v):return -dep.evaluate(mk,dep.RegimePolicy(v[:4].reshape(2,2),v[4:].reshape(2,2)),1)['J']
    fit=minimize(objective,np.zeros(8),method='BFGS',options={'gtol':1e-8})
    assert abs(-fit.fun-target)<1e-7


def test_dependent_moments_exhaustive_tree():
    mk=dep.demo_market(horizon=3);pol=dep.precommitted(mk,1)
    def rec(t,z,x,p):
        if t==mk.horizon:return [(p,x)]
        prob,j,r=mk.branches(z);u=pol.alpha[t,z]*x+pol.beta[t,z]
        return sum([rec(t+1,int(jj),mk.riskless*x+rr*u,p*pp) for pp,jj,rr in zip(prob,j,r)],[])
    paths=rec(0,0,1.,1.);mean=sum(p*x for p,x in paths);var=sum(p*(x-mean)**2 for p,x in paths)
    ev=dep.evaluate(mk,pol,1)
    assert abs(ev['mean']-mean)<1e-12 and abs(ev['variance']-var)<1e-12


def test_dsr_identity_positive_variance():
    A=.03;B=.04;rho=-.2;eta=.1
    D,aa,bb=dsr_step(A,B,rho,eta);V=B-A*A
    assert abs(D-B/V**1.5*(rho-A/(2*B)*rho*rho-A/2))<1e-12
    assert abs(bb-aa*aa-((1-eta)*V+eta*(1-eta)*(rho-A)**2))<1e-12
    assert ewma_variance(.04,.1,1000)>.002
    with pytest.raises(ValueError):dsr_step(0,0,.1)


def test_dsr_augmented_future_differs_from_myopic():
    from mvrl.bounded import solve_tree
    short=solve_tree('dsr',T=1,grid_size=11,eta=.4,A0=.02,B0=.002)
    long=solve_tree('dsr',T=3,grid_size=11,eta=.4,A0=.02,B0=.002)
    # Save an actual counterexample, not a claim of fixed-penalty equivalence.
    assert short['policy'][()]!=long['policy'][()]

def test_hedge_reward_exact_recovery():
    for q in [.5,.8,.95]:
        mk=dep.demo_market(q);eq,_=dep.equilibrium(mk,2)
        assert dep.distance(mk,dep.hedge_reward_policy(mk,2),eq)<1e-12


def test_tree_independent_enumeration():
    from itertools import product
    from mvrl.bounded import solve_tree
    vals=[]
    for ws in product([0,.5,1],repeat=3):
        value=0
        for j,r0 in enumerate([-.18,.30]):
            x=1.02+r0*ws[0]
            for r1 in [-.18,.30]:
                y=x*(1.02+r1*ws[j+1])
                value+=.25*((x-1)-(x-1)**2+(y-x)-(y-x)**2)
        vals.append(value)
    assert abs(solve_tree('raw',T=2,grid_size=3)['reward']-max(vals))<1e-12


def test_time_varying_scalar_lower_bound():
    mk=IndependentMarket([[.02],[.08],[.03]],[[[.02]],[[.04]],[[.01]]],[1.,1.,1.])
    a=np.array([np.linalg.norm(np.linalg.solve(mk.M(t),mk.m(t)))**2 for t in range(3)])
    b=np.array([1/(2*(1-mk.nu(t))) for t in range(3)])
    lower=np.sum(a*(b-np.sum(a*b)/np.sum(a))**2)
    _,_,dist=exact_raw_calibration(mk,1,1)
    assert lower>0 and abs(dist**2-lower)<1e-12


def test_mvpi_adaptation_and_zero_interest_equivalence():
    from mvrl.mvpi import solve
    for s in [1.,1.02]:
        mk=demo_market(s,4);p,meta=solve(mk,1)
        assert np.min(np.diff(meta['trace']))>-1e-12
        assert meta['residual']<1e-12
        if s==1:assert np.allclose(p.beta,solve_equilibrium(mk,1).policy.beta,atol=1e-10)
