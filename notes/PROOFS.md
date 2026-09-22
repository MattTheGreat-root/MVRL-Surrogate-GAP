# Completed and corrected mathematical results

Status: **proved**, with the assumptions stated below; independent numerical checks are recorded separately. These are project derivations, not a claim of literature priority.

Assume a finite horizon, deterministic positive gross risk-free returns, unconstrained risky dollar holdings, square-integrable independent excess-return vectors, and positive-definite covariance. Policies have finite second moments. Set M=Sigma+mm', d=M^{-1}m, nu=m'd in [0,1). Nondegenerate matching statements require m != 0.

## 1. Second moment is not variance
E[d_t-kappa*d_t²]=E[d_t]-kappa Var(d_t)-kappa E[d_t]². Calling this simply a variance penalty hides an additional squared-mean penalty. A conditional-variance reward m'u-phi*u'Sigma*u has a different optimum. Much of the zero-interest calibration follows immediately from this distinction and Sherman–Morrison; it must not be sold as an unexplained failure of Bellman optimisation.

## 2. Existing recursion and exactness
Let W_t(x)=p_t x²+q_t x+c_t and g_t=kappa(s_t−1)−p_next*s_t. Completing the square yields

    alpha_t = −d_t*g_t/(kappa−p_next)
    beta_t  = d_t*(1+q_next)/[2(kappa−p_next)]
    p_t = [kappa*p_next−(1−nu_t)*g_t²]/(kappa−p_next)
    1+q_t = (1+q_next)*(s_t−nu_t*g_t/(kappa−p_next)),
    p_T=q_T=0.

The constant c_t does not affect actions. For the quadratic increment reward, Bellman maximisation has Hessian -2(kappa-p_next)M. The recursion gives p<=0 by induction. Thus it maximises over all admissible Markov actions, not just an affine search class. At s=1 the optimum is d/(2kappa); the equilibrium is Sigma^{-1}m/(2phi)=d/[2phi(1-nu)]. Equality iff kappa=phi(1-nu) when m!=0. If m=0 both are zero for every kappa.

For T=1, equality refers to the initial action, not equality of functions on all wealth values. It requires 1/[2phi(1-nu)]+(s-1)x0>0; otherwise no positive finite kappa matches (m!=0).

## 3. General horizon small-interest theorem, including the remainder
Let m,Sigma be constant, T>=2 fixed, s=1+e, and b=1/[2phi(1-nu)]. Use the ORIGINAL one-sided occupancy discrepancy D²=sum E_{pi_k}||pi_k(t,X_t)-pi_eq(t,X_t)||². It is a discrepancy, not a symmetric metric.

Then, for sufficiently small |e|,

    min_{kappa>0} D = c_T |e| + O(e²),
    c_T = ||d|| b sqrt[T(T²-1)/3 + nu(1-nu)T(T-1)/2].

This proves the previous general-horizon conjecture and supplies its constant. For T=2 the bracket is (2-nu)(1+nu). As T tends to infinity with other parameters fixed, c_T ~ ||d|| b T^(3/2)/sqrt(3). This is not a uniform joint limit in e and T.

Proof. Divide p by kappa in the recursion. p/kappa and q, hence alpha, are independent of kappa; beta is linear in h=1/(2kappa). The controlled wealth on any common return path is X_t=L_t(e)x0+h Z_t(e). Therefore the squared discrepancy F(e,h) is exactly a quadratic polynomial in h. Its coefficients are analytic near e=0, since the finite backward recursions have nonzero denominators there, and moment propagation uses only finite second moments. At e=0, F(0,h)=T||d||²(h-b)². The h² coefficient is strictly positive; hence the global unconstrained minimiser h*(e) is analytic and stays positive near zero. It is also the global minimiser over h>0. This justifies minimisation and the uniform local Taylor expansion rather than assuming that a fitted local optimum is global.

Write kappa=phi(1-nu)(1+w e). Uniformly for bounded w, alpha_t=-d e+O(e²), beta_t=d b[1+((T-t-1)(1-nu)-w)e]+O(e²), and beta_eq,t=d b[1-(T-t-1)e]+O(e²). At e=0 the common policy is d b, giving E[X_t]=x0+t nu b and Var(X_t)=t b² nu(1-nu). The leading action difference divided by d e is -X_t+b[(T-t-1)(2-nu)-w]. Its means form an arithmetic progression with step -2b. The minimising w is (T-1)(1-nu)-x0/b. After centring, the means are b(T-1-2t). Summing their squares gives b² T(T²-1)/3; summing variances gives b² nu(1-nu)T(T-1)/2. Analyticity yields min F=c_T² e²+O(e³). Since c_T>0, its square root has remainder O(e²). QED.

The same leading coefficient holds under fixed equilibrium occupancy, because at e=0 the occupancies agree and only zeroth-order moments enter the leading quadratic term. Away from e=0 these are different discrepancies and must be labelled.

For m!=0, s!=1, T>=2 no exact matching along the surrogate occupancy is possible: equality at t=0 forces a nonzero equilibrium action, hence positive next-wealth variance. Every subsequent nonzero equilibrium action preserves positive variance. At the final date surrogate slope is -d(s-1) while equilibrium slope is zero, contradicting equality almost surely. This obstruction concerns the RAW increment family, not all additive rewards.

No global bound D<=C(|kappa-kappa0|+c|s-1|) with constant C over all kappa>0 follows: at s=1, D= sqrt(T)||d|| |1/(2kappa)-b| diverges as kappa tends to zero while the proposed right side stays bounded. A local bound on compact positive penalty sets is valid by smoothness.

## 4. Time-varying independent returns: exact reward redesign
Let H_t=product_{j=t+1}^{T-1} s_j (empty product 1). The equilibrium policy is

    u_eq,t = Sigma_t^{-1}m_t/(2phi H_t).

Proof: under future deterministic holdings, X_T=H_t(s_t x+P_t'u)+future gains. Independence removes action-dependent covariance with future gains. The one-step objective has u-dependent part H_t m_t'u-phi H_t² u'Sigma_t u, strictly concave, giving the displayed action. Backward induction establishes deterministic equilibrium holdings.

Define Y_t=H_t P_t'u_t and

    r_design,t = Y_t - phi(1-nu_t) Y_t².

Each conditional reward is strictly concave in u and independent of wealth. Exogenous independent returns and unconstrained actions mean pointwise maximisation at every time globally maximises the additive expectation, even over history-dependent policies. Its optimum is d_t/[2phi(1-nu_t)H_t]=u_eq,t. QED.

This removes the raw-reward interest obstruction by measuring excess gains in terminal units and applying time-dependent calibration. It reproduces EQUILIBRIUM, not the pre-committed optimum. An alternative conditional-variance reward Y_t-phi H_t² u'Sigma_t u has the same conditional maximiser. The redesign is model-based and requires moments and the intended target. It is not a universal improvement in J: a different policy may beat equilibrium at time zero.

At all s_t=1, a common raw kappa matches iff every nonzero-mean date has equal nu_t and kappa=phi(1-nu_t). For fixed reference occupancy (all policies here deterministic holdings), write a_t=||d_t||² and b_t=1/[2phi(1-nu_t)]. The squared distance is sum a_t(h-b_t)². Its exact minimum is sum a_t(b_t-b_bar)², with b_bar=sum a_t b_t/sum a_t. This is a positive lower bound iff the b_t vary over positive weights. Time variation alone does NOT imply an unavoidable mismatch (nu_t can remain constant).

## 5. Predictable Markov returns and the missing hedge
For a scalar excess return P and observed finite regime z, specify the JOINT conditional law of (P,z_next). Rates are deterministic. Let equilibrium terminal mean continuation be f_next(x,z')=H_t x+a_next(z'). Its conditional variance is independent of wealth. The current action-dependent variance is H_t² Var(P|z)u²+2H_t Cov(P,a_next(z_next)|z)u. Consequently

    u_eq(t,z)=m_z/[2phi H_t variance_z]
              -Cov(P,a_next(z_next)|z)/[H_t variance_z].

Backward propagation of conditional first/second moments establishes the formula. Positive conditional return variance makes it a strict global one-step maximum. A statewise substitution kappa=phi(1-nu_z) produces only the first term; a nonzero hedge covariance defeats that substitution even at zero interest.

An equilibrium-informed additive reward is

    r = Y-phi(1-nu_z)Y²
        -2phi Cov(P,a_next|z) H_t u,
    Y=H_t P u.

Conditional derivative gives exactly u_eq. Because exogenous transitions and this reward do not depend on wealth, its pointwise maximisers globally maximise the sum. This is an inverse-control construction using an already solved continuation, not a model-free shortcut or novelty claim. The useful test is the quantified failure of local calibration and the recovered hedge. Regime-switching MV itself is established literature.

## 6. DSR: what can actually be concluded
With A'=A+eta(rho-A), B'=B+eta(rho²-B), V=B-A²>0,

    V'=(1-eta)V+eta(1-eta)(rho-A)²>0,  0<eta<1.

For bounded returns |rho|<=R, finite T, and V0>0, V_t >= (1-eta)^t V0>0 and the DSR is bounded over the reachable finite-horizon state set. Thus an augmented state including market predictors, previous holdings if costs apply, wealth, A and B supports ordinary finite-horizon Bellman recursion. No infinite-horizon fixed-point result follows without additional assumptions.

The unscaled differential Sharpe signal is D=[B(rho-A)-A(rho²-B)/2]/V^(3/2). The legacy code multiplies this by eta, which leaves the policy optimum unchanged for one fixed eta, but changes reported reward units. Its local identity is D=(B/V^(3/2))[rho-(A/(2B))rho²-A/2]. This is only a one-step identity. A,B depend on previous actions, the positive scale is state-dependent, and the continuation value depends on their updates. It does NOT make the optimal dynamic policy a fixed-kappa quadratic policy. A<0 also gives a negative effective penalty. No constant implied terminal risk aversion has been proved.

For iid rho with finite variance sigma², a fixed eta EWMA satisfies limiting Var(A_t)=eta*sigma²/(2-eta)>0. It is not a consistent estimator. Its expectation tends to the mean; its realisation does not converge in probability to it. This disproves the legacy consistency claim. The corrected bounded-tree experiment solves the augmented reward exactly on a finite action set and tests eta/initialisation sensitivity without asserting an unrestricted optimum.

## 7. Known benchmark identities
For iid nu, deterministic risk-free growth S=product s_t and initial x0,

    J_pre=S*x0+[(1-nu)^(-T)-1]/(4phi),
    J_eq =S*x0+T*nu/[4phi(1-nu)].

The first follows from the Li–Ng embedding or completing the efficient-frontier square; the second follows by summing independent equilibrium gains in terminal units. Their difference proves the 1/phi scaling and independence of x0. These are consequences of known mean–variance solutions, not claimed new.

Signed telescoping J differences must be accompanied by nonnegative surrogate regret J_r(pi_r)-J_r(pi_hat). Sharpe means terminal EXCESS gain over x0*product s_t divided by terminal wealth standard deviation; it is not an annualised backtest Sharpe.

## 8. Existing MVPI transformation subsumes zero-interest scalar calibration
Zhang, Liu and Whiteson (2021), Algorithm 1, uses r_hat=(1+2phi*y)r-phi*r² with y equal to the occupancy mean reward. In our finite-horizon uniform-time adaptation, s=1 and iid returns give deterministic identical holdings u_n=d*h_n after each exact inner solve. Then y_n=m'u_n=nu*h_n, so h_{n+1}=1/(2phi)+nu*h_n. Because 0<=nu<1, the iteration converges to h*=1/[2phi(1-nu)], precisely the scalar-calibrated equilibrium. This is an elementary reduction of existing MVPI, not a new learning method. For nonzero interest, the implemented iteration remains a fixed-point method for a per-step mean-volatility objective; its numerical convergence does not establish a global optimum or terminal-MV equilibrium.
