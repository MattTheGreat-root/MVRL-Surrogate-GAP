"""Shared content for proposal/paper PDF, Markdown and XeLaTeX sources."""
from pathlib import Path
import json, numpy as np
ROOT=Path(__file__).resolve().parents[1]
r=json.loads((ROOT/'results/experiments.json').read_text());refs=json.loads((ROOT/'literature/ledger.json').read_text())
idx={x['key']:i+1 for i,x in enumerate(refs)}
def cite(*keys):return '['+', '.join(str(idx[k]) for k in keys)+']'
def p(s):return {'type':'p','text':s}
def h(s):return {'type':'h','text':s}
def eq(s):return {'type':'eq','text':s}
def fig(name,caption):return {'type':'fig','path':'figures/'+name+'.png','caption':caption}
def table(head,rows,widths=None):return {'type':'table','head':head,'rows':rows,'widths':widths}
def page(title,*blocks):return {'title':title,'blocks':list(blocks)}
def get(key,**kw):return [x for x in r[key] if all(x[k]==v for k,v in kw.items())]
main={x['name']:x for x in get('independent',T=4,phi=1.,s=1.02)}
num=lambda x:f'{x:.6f}'
status_rows=[['R1 · Raw reward recursion and zero-interest match','PROVED','Known control tools; zero-interest result also follows from MVPI.'],['R2 · Fixed-horizon mismatch coefficient','PROVED','Project derivation; literature priority not established.'],['R3 · Terminal-unit independent-return redesign','PROVED','Unconstrained dollar actions; known moments; equilibrium target.'],['R4 · Common-penalty lower bound under changing moments','PROVED','Zero interest; nonzero-mean dates; weighted dispersion formula.'],['R5 · Observed-regime hedge and oracle correction','PROVED','Scalar risky return, exogenous observed regimes; solved continuation required.'],['R6 · DSR algebra and finite-horizon well-posedness','PROVED','Bounded returns, positive initial EWMA variance; no infinite-horizon theorem.'],['R7 · Reward adaptations and parameter sweeps','NUMERICALLY VERIFIED','Exact recursions / exact finite action trees in specified markets.'],['R8 · Training and estimation evidence','PRELIMINARY','Five training seeds; thirty estimation seeds; no broad superiority claim.']]
proposal=[]
proposal.append(page('Reward design that targets the intended portfolio policy',
 p('Research proposal with established results · revised 22 September 2026'),
 h('The problem'),
 p('A financial RL agent optimises the reward supplied to it. A good Sharpe ratio or a well-trained policy does not establish that this reward represents the investor’s intended terminal financial objective. The first decision is therefore which policy is intended: the initial-date mean–variance optimum, or the time-consistent equilibrium followed by successive decision makers.'),
 h('The method'),
 p('Specify the financial objective and target policy; solve the reward-induced control problem; measure policy mismatch independently of learning; calibrate or redesign the reward; then evaluate the learned policy against both exact references.'),
 h('The strongest established result'),
 p('For an unconstrained independent-return market, one scalar penalty on squared wealth increments cannot generally recover the terminal mean–variance equilibrium at nonzero interest. We derive its exact local mismatch coefficient for every fixed horizon, and give a terminal-unit excess-gain reward that recovers the equilibrium exactly. These are scoped mathematical results, not a claim of universal financial or learning superiority.'),
 fig('01_policy_alignment','Figure 1 · NUMERICALLY VERIFIED. T=4, φ=1, s=1.02, three risky assets. Reward redesign eliminates exact policy mismatch. Scalar calibration attains a slightly larger initial J than equilibrium: better alignment and larger initial J are different goals.'),
 p('Practical output: an auditable test of whether a portfolio reward targets the specified policy, a constructive correction when assumptions permit it, and a diagnosis of whether reward design or learning is the main remaining error.')))
proposal.append(page('1 · What the closest literature actually does',
 p('The gap is not that financial RL ignores objectives or time inconsistency. Several important papers address them directly. The relevant question is how a specified reward’s optimal policy differs from an independently specified terminal-wealth target.'),
 table(['Paper / exact locator','Objective and method','Role in this project'],[
 ['Kolm–Ritter (2019), manuscript Eqs. 8–10','Approximate quadratic increment reward; fitted SARSA hedging.','Closest reward family; explicitly acknowledges covariance/approximation.'],
 ['Jiang et al. (2017), §4.2 Eqs. 21–22','Log-growth reward; direct gradient portfolio learning.','Alternative preference; not intrinsically a wrong growth reward.'],
 ['Yang et al. (2020), §3.3 Eq. 4; Table 2','Net value change; PPO/A2C/DDPG; Sharpe ensemble selection.','Return-reward adaptation; separates reward from evaluation criterion.'],
 ['Moody–Saffell (2001), §II-D Eqs. 14–15','Differential Sharpe; recurrent direct reinforcement.','Augmented-state DSR comparison, not fixed-penalty equivalence.'],
 ['Bisi et al. (2020), §§3–5; Zhang et al. (2021), Algorithm 1','Per-step mean-volatility; TRVO / MVPI-TD3.','Direct prior reward-transformation theory; MVPI adaptation added.'],
 ['Wang–Zhou (2020), §2 Eq. 11; Dai et al. (2023), working §4','Objective-specific MV RL; precommitment / log-MV equilibrium.','Positive precedents with analytic references; no first-MV-RL claim.']],[.30,.35,.35]),
 p('Concrete positioning: Kolm–Ritter optimise an explicitly approximate quadratic increment reward for hedging. Jiang et al. choose log growth. Yang et al. train on value increments and select agents by Sharpe. Our benchmark holds the market and feasible set fixed, solves these adapted reward problems, and compares their induced policies with a prescribed mean–variance equilibrium. This tests preference alignment; it does not refute each paper’s own stated objective. '+cite('kolm2019','jiang2017','yang2020')),
 p('The zero-interest calibration is not a sufficient novelty claim: the finite-horizon iid reduction of Zhang’s MVPI has the same fixed point. Basak–Chabakauri also discuss recovering time-consistent objectives with the same policy, and Björk–Murgoci’s abstract states an associated-control result. The candidate contribution is the explicit quantitative obstruction and its audited removal within a concrete reward family. Priority remains OPEN. '+cite('zhang2021','basak2010','bjork2014')),
 p('The source ledger records authors, exact titles, venue, identifiers, reward, algorithm, data, reported results, and version-specific locators for every comparison. Eleven primary versions were inspected; one additional source is explicitly abstract-only.')))
proposal.append(page('2 · Objective, benchmark and calibration procedure',
 p('Established analytic scope: finite horizon T, initial wealth x₀, deterministic positive risk-free gross returns sₜ, unconstrained risky dollar holdings uₜ, and independent excess-return vectors Pₜ with means mₜ and positive-definite covariances Σₜ. Gaussian sampling is used for training, but the exact affine moment results only require finite second moments. Long-only weights are treated separately.'),
 eq('Xₜ₊₁ = sₜ Xₜ + Pₜᵀuₜ      J(π) = E[X_T] − φ Var(X_T), φ>0\nMₜ = Σₜ + mₜmₜᵀ     dₜ = Mₜ⁻¹mₜ     νₜ = mₜᵀdₜ < 1\nHₜ = product of sⱼ for j>t      u_eq,t = Σₜ⁻¹mₜ / (2φHₜ)'),
 p('Li–Ng supply the precommitted reference; equilibrium control follows consistent planning. Our redesign targets equilibrium. If initial-date performance is the intended objective, use the precommitted benchmark and an appropriate terminal objective instead. '+cite('ling2000','basak2010','wang2020')),
 h('A reward-design rule with an explicit guarantee'),
 eq('Yₜ = Hₜ Pₜᵀuₜ\nr_design,t = Yₜ − φ(1−νₜ)Yₜ²'),
 p('PROVED: conditional expected reward is strictly concave in uₜ, wealth-independent, and maximised at u_eq,t. Independence and unconstrained actions make these pointwise maximisers globally optimal for the additive reward. Scaling gains into terminal units removes the interest-driven obstruction of the original wealth-increment reward. Time-varying moments require date-specific calibration.'),
 fig('10_existing_transformations','Figure 2 · NUMERICALLY VERIFIED. Existing MVPI adaptation agrees at zero interest but targets per-step mean-volatility away from it. With changing moments at zero interest, a common scalar penalty leaves distance 0.067715; the date-specific design gives zero.'),
 p('Implementation procedure: estimate or specify moments; declare the target and constraints; solve the exact reward problem where possible; report distance and terminal J; optimise reward parameters using a fixed reference occupancy; then train with the resulting reward and report its optimisation error separately.')))
proposal.append(page('3 · Theoretical contribution and its boundaries',
 h('R2 · The irreducible raw-reward mismatch — PROVED'),
 p('For iid returns set d=M⁻¹m, ν=mᵀd and b=1/[2φ(1−ν)]. Assume m≠0 and fixed T≥2. For r=ΔX−κ(ΔX)², let D be the square root of summed expected squared action differences along the reward policy. Then, locally around s=1:'),
 eq('min over κ>0 D(πκ,πeq) = c_T |s−1| + O((s−1)²)\nc_T = ||d|| b √[T(T²−1)/3 + ν(1−ν)T(T−1)/2]'),
 p('At zero interest exact matching requires κ=φ(1−ν). At nonzero interest and T≥2 the final reward-optimal policy has wealth slope −d(s−1), while equilibrium has zero slope and reachable wealth has positive variance. No scalar κ removes that difference. The statement concerns this reward family, not all additive rewards.'),
 p('The proof closes the earlier remainder and minimisation gaps: write h=1/(2κ); policy intercepts and controlled wealth are affine in h, so D² is exactly quadratic in h. Its coefficients are analytic in s−1 and its unique minimum remains interior near zero. This establishes the expansion after minimisation. At large fixed-parameter T, c_T grows as T^(3/2); this is not a joint uniform small-interest/large-horizon theorem.'),
 fig('02_horizon_theorem','Figure 3 · PROVED coefficient with NUMERICALLY VERIFIED finite-interest checks. Solid curves show exact calibration and dashed curves the local prediction. The earlier “approximately linear in horizon” claim is withdrawn.'),
 h('R4 · Changing moments and a common penalty — PROVED'),
 eq('At sₜ=1: min D² = Σₜ aₜ(bₜ−b̄)²\naₜ=||dₜ||², bₜ=1/[2φ(1−νₜ)], b̄=(Σₜaₜbₜ)/(Σₜaₜ).'),
 p('The lower bound is positive exactly when bₜ varies across positive-weight dates. Time variation by itself is insufficient. Degenerate zero means, the positivity condition for one-period matching, and the local nature of the theorem are retained in the proof appendix.')))
proposal.append(page('4 · Fair comparison with published reward designs',
 p('NUMERICALLY VERIFIED: a separate long-only benchmark has T=3, x₀=1, s=1.02, one risky excess return −.18 or .30 with equal probability, zero costs, and a common 21-point weight grid in [0,1]. Bellman recursion exhaustively solves each finite reward tree. DSR state includes A and B. A one-step-game recursion solves the equilibrium on the same grid. The 11-point grid is also run.'),
 fig('04_published_rewards','Figure 4 · Adaptations of published formulas, not replications of their data, algorithms or reported performance. κ/2=.05 is a unit-sensitive coefficient copied from the Kolm–Ritter example and is shown only as a sensitivity diagnostic.'),
 table(['Adapted reward','Distance','Terminal J','Excess Sharpe'],[[{'published_quadratic':'Published .05 diagnostic','return':'Value change','log':'Log growth','dsr':'DSR','raw':'Raw κ=φ','scalar':'Scalar calibration','design':'Terminal-unit design','equilibrium':'Grid equilibrium'}[x['name']],f"{x['distance']:.5f}",f"{x['J']:.6f}",f"{x['sharpe']:.5f}"] for x in get('bounded',phi=1.,grid=21)],[.43,.19,.19,.19]),
 p('The design improves alignment over raw and scalar rewards in this configuration, but has slightly lower J than either. It does not exactly recover the constrained equilibrium: wealth-dependent feasible dollar holdings invalidate the unconstrained proof. Return and log-growth policies pursue different preferences, so their terminal-MV rankings are not evidence that they fail at their own tasks.'),
 p('DSR uses η=.1, A₀=.01, B₀=.01 here. Sensitivity tests vary all three. Reward shapes are transferred; PPO/DDPG, RRL, trading costs, historical datasets and original train/test protocols are not reproduced. Algorithms cannot be ranked from this experiment.')))
proposal.append(page('5 · Exact values, dependent returns and DSR',
 table(['Independent benchmark','Policy distance','Terminal J','Excess Sharpe'],[[{'pre':'Precommitted reference','equilibrium':'Equilibrium','raw':'Raw κ=φ','scalar':'Scalar calibration','best_raw':'Best scalar in raw family','design':'Terminal-unit design'}[n],f"{main[n]['distance']:.6f}",f"{main[n]['J']:.6f}",f"{main[n]['sharpe']:.6f}"] for n in ['pre','equilibrium','raw','scalar','best_raw','design']],[.43,.19,.19,.19]),
 p('NUMERICALLY VERIFIED: T=4, φ=1, s=1.02. Distance falls from .797127 to .195347 after zero-interest scalar calibration and to numerical zero after redesign. Raw reward has higher excess Sharpe than equilibrium (.987285 vs .982266), but lower J (1.316580 vs 1.323644). Sharpe uses excess terminal gain over x₀s^T and is not annualised.'),
 h('R5 · Dependence introduces an intertemporal hedge — PROVED'),
 eq('u_eq(t,z) = m_z/(2φHₜ Var(P|z))\n              − Cov(P,a_next(z_next)|z)/(Hₜ Var(P|z))'),
 p('In the observed finite-regime model, a_next is the expected terminal continuation gain. A calibration using only current conditional moments omits this covariance. Adding −2φHₜu Cov(P,a_next|z) to the terminal-unit reward recovers equilibrium. This is an oracle construction using an already solved continuation, not a model-free learning result or a novel hedging principle. '+cite('basak2010')),
 fig('05_dependent_returns','Figure 5 · NUMERICALLY VERIFIED exact joint return/regime recursion. At persistence .8, raw distance is .365102, local calibration .317644, hedge-corrected reward numerical zero. Raw J exceeds equilibrium J here; that is consistent with the target being equilibrium.'),
 p('R6 — PROVED: bounded finite-horizon DSR is well posed with state augmentation and B₀−A₀²>0. Its local quadratic identity does not establish dynamic equivalence to a fixed variance penalty. Fixed-decay EWMA has positive limiting variance; the earlier consistency claim is false. Unrestricted or infinite-horizon DSR remains OPEN.')))
proposal.append(page('6 · Robustness and the limits of transfer',
 fig('03_risk_regimes','Figure 6 · NUMERICALLY VERIFIED. Common-scale φ×distance across risk aversion and interest. The full grid has 120 parameter configurations: five horizons, six risk aversions and four rates, each with six exact reference/reward policies. Maximum redesign discrepancy is 2.35×10⁻¹⁴.'),
 p('Additional exact checks vary mean, volatility and correlation over 27 markets. Redesign discrepancy remains below 5.50×10⁻¹⁵. These verify the scoped identity, not robustness of estimated models to arbitrary market misspecification.'),
 fig('08_estimation_robustness','Figure 7 · PRELIMINARY. Left: thirty independent moment-estimation samples at each size, mean ± sample SD. Right: frozen policies on ten batches of 50,000 paths, mean ± sample SD. GARCH and AR(1) are dependent stress tests; iid optimality is not asserted there.'),
 p('The design needs accurate moments. Estimation experiments evaluate policies built from estimated moments in the true market. Stress tests hold policies fixed: matched-moment iid t(8) preserves the moment-based result, but serial correlation can reverse performance. Under AR(1), design minus raw J is −.03813 (batch SE .00027); under Gaussian iid it is +.00713 (SE .00018). No universal robustness advantage is claimed.')))
proposal.append(page('7 · Separate learning error from reward-design error',
 eq('J(pre)−J(learned) = [J(pre)−J(eq)]\n                   + [J(eq)−J(reward optimum)]\n                   + [J(reward optimum)−J(learned)]'),
 p('The identity telescopes. Only the first term is guaranteed nonnegative in this benchmark. The last two are signed differences in the financial objective. Report a separate nonnegative surrogate regret: expected reward at its exact optimum minus expected reward of the learned policy. Policy distance always uses the declared reference occupancy.'),
 fig('06_learning_error','Figure 8 · PRELIMINARY. Five paired seeds per reward, 4,000 iterations and batches of 512, common affine policy family, Adam learning rate .02 and exploration schedule. This is the existing REINFORCE-style implementation with normalised advantages and scaled scores. No convergence theorem is asserted.'),
 table(['Learned reward','Mean distance ± SD','Mean J ± SD','Mean surrogate regret'],[[n,f"{np.mean([x['distance'] for x in get('learning',name=n)]):.4f} ± {np.std([x['distance'] for x in get('learning',name=n)],ddof=1):.4f}",f"{np.mean([x['J'] for x in get('learning',name=n)]):.6f} ± {np.std([x['J'] for x in get('learning',name=n)],ddof=1):.6f}",f"{np.mean([x['surrogate_regret'] for x in get('learning',name=n)]):.6f}"] for n in ['raw','scalar','design']],[.18,.28,.34,.20]),
 p('The exact redesign is aligned, but the learned redesign retains substantial distance. Its mean J is lower than the learned raw reward at this budget. Thus the current learning experiment does not support improved financial performance or broad algorithmic superiority. Reward regret is reported in each reward’s own units; comparisons of its scale across rewards require care.'),
 p('The fair next algorithm study must tune each reward on a separate validation budget, keep environment interactions and policy capacity comparable, add seeds, and report uncertainty and failures. The current experiment is a diagnosis, not a finished PPO/DDPG benchmark.')))
proposal.append(page('8 · What is established, and what remains proposed',
 table(['Result','Status','Boundary'],status_rows,[.37,.20,.43]),
 h('What can a researcher do now?'),
 p('Audit an additive financial reward against a specified equilibrium, compute its exact induced policy and mismatch without learning noise, choose the best scalar penalty in the raw family, and determine whether a structural redesign is required. Under independent known moments, implement the displayed reward to recover equilibrium exactly. Under observable dependence, identify the missing continuation hedge rather than assume local calibration suffices.'),
 h('Remaining research, with acceptance criteria'),
 p('OPEN — Publication priority: obtain the full discrete-control prior and compare the exact coefficient and redesign with earlier inverse-control and risk-sensitive RL results. Claim no “first” result before this is done. Existing MVPI equivalence must remain in the final positioning.'),
 p('OPEN — Constraints and costs: derive or numerically solve reward designs for wealth-dependent feasible sets, transaction costs and multiple dependent assets. Require a shared feasible benchmark and a quantified discretisation error before claiming exact recovery.'),
 p('OPEN — DSR: analyse unrestricted policies or an appropriately regularised infinite-horizon formulation. A bounded finite-grid calculation cannot establish a constant implied terminal risk aversion or a global DSR theorem.'),
 p('PROPOSED — Learning and practical validation: expand tuned policy-gradient/PPO/DDPG comparisons, implement estimated continuation hedges, and test with a preregistered historical protocol. Success means improved target alignment with controlled learning error; Sharpe and raw return remain secondary, separately reported outcomes.'),
 p('The former general-horizon conjecture is resolved locally. The broad global Lambda inequality is withdrawn.')))
proposal.append(page('9 · Evidence that changes the interpretation',
 fig('07_sharpe_dsr','Figure 9 · NUMERICALLY VERIFIED. The Sharpe/J ranking disagreement survives the corrected risk-free subtraction. DSR initial allocation varies materially with EWMA initialisation and decay; the experiment optimises full finite-horizon continuation.'),
 fig('09_decomposition_markets','Figure 10 · PRELIMINARY learning decomposition (left), NUMERICALLY VERIFIED moment sweep (right). Signed components must not be presented as three nonnegative losses. Ordered market curves compare each method’s distribution of distances, not paired x-axis cases.'),
 p('Main limitations: analytically simple markets and unconstrained dollar positions; moment and model knowledge; oracle continuation in the dependent redesign; small exact constrained trees; five training seeds; no transaction-cost or historical replication. These are deliberate boundaries of the evidence.'),
 p('The research contribution is therefore an exact, reproducible diagnostic and a conditional design result. It is stronger than an abstract framework, but narrower than a claim that a new reward generally improves financial RL.')))
refblocks=[]
for i,x in enumerate(refs,1):
 refblocks.append(p(f"[{i}] {x['authors']}. {x['title']}. {x['venue']} ({x['year']}). {x['url']}"))
 refblocks.append(p('Version/locator: '+x['version']+' '+x['locator']))
proposal.append(page('References and version-specific evidence · 1',*refblocks[:12]))
proposal.append(page('References and version-specific evidence · 2',*refblocks[12:]))
# Paper shares all empirical claims but places complete proof material before experiments.
proof=ROOT/'notes/PROOFS.md'
proofpages=[]
parts=proof.read_text().split('\n## ')[1:]
for section in parts:
 lines=section.splitlines();blocks=[];buf=[];code=[]
 def flush():
  if buf:blocks.append(p(' '.join(buf)));buf.clear()
  if code:blocks.append(eq('\n'.join(code)));code.clear()
 for line in lines[1:]:
  if not line.strip():flush()
  elif line.startswith('    '):
   if buf:flush()
   code.append(line[4:])
  else:
   if code:flush()
   buf.append(line.replace('**',''))
 flush();proofpages.append(page('Proof appendix · '+lines[0],*blocks))
proofpages=[page('Proof appendix · second moments and exact recursion',*proofpages[0]['blocks'],h(proofpages[1]['title']),*proofpages[1]['blocks']),*proofpages[2:6],page('Proof appendix · known benchmarks and MVPI',*proofpages[6]['blocks'],h(proofpages[7]['title']),*proofpages[7]['blocks'])]
paper=[page('From portfolio objectives to calibrated RL rewards',p('Research draft · 22 September 2026 · verified results and explicit open questions'),h('Abstract'),p('We study the policy induced by additive quadratic wealth-increment rewards relative to terminal mean–variance equilibrium in a controlled discrete-time market. We establish a fixed-horizon local coefficient for the irreducible mismatch of the scalar-penalty family, construct a terminal-unit reward that exactly recovers equilibrium under independent return moments, and derive the missing continuation hedge under observed Markov dependence. Exact and finite-grid comparisons adapt rewards from financial RL papers, while a separate five-seed learning study quantifies optimisation error. We find strong exact alignment improvements but no general improvement in initial mean–variance value, constrained performance or learned financial performance. Existing mean–variance policy iteration reproduces the zero-interest calibration, so this formula is not claimed as a new algorithm. Novelty of the quantitative results remains subject to further prior-art verification.'),h('Scope and reproducibility'),p('This draft replaces the earlier unqualified surrogate-gap narrative. The paper and proposal share saved experiment data, exact assumptions, figure captions and a version-specific literature ledger. Proofs below establish mathematical claims within the model; they do not certify publication priority. The raw experiment bundle contains all seeds, exact benchmark values and training histories.'),table(['Evaluation object','Meaning'],[['Precommitted policy','Initial-date maximiser of J.'],['Equilibrium policy','No profitable one-step deviation by any later self.'],['Reward optimum','Exact optimiser of the chosen additive criterion.'],['Learned policy','Finite-budget output, evaluated as a deterministic affine policy.']],[.30,.70])),*proposal[1:4],*proofpages,*proposal[4:]]
paper[0]['blocks'].append(next(b for b in proposal[0]['blocks'] if b['type']=='fig'))
# Additional standalone protocol ensures precise reproduction independent of narrative PDF.
protocol='''# Implementation and verification report — 22 September 2026

## Deliverables and provenance

All existing research text/code/archive contents and every supplied figure were inspected before edits. The initial inventory is notes/INITIAL_AUDIT.md. Originals are preserved under archive/2026-09-21-before-audit. No historical dataset, raw experiment output, or standalone implementation report was present initially. This report documents the new run, not a reconstruction of unavailable old outputs.

## Model and algorithms

Main iid market: m=(.06,.04,.02); vol=(.18,.12,.08); correlation rows (1,.30,-.10), (.30,1,.05), (-.10,.05,1). x0=1. The analytic market uses risky dollar holdings without leverage/solvency constraints. Gaussian returns are used in training. Exact moment results do not require Gaussianity. nu=0.1943363227 approximately; use computed values, not this rounded value.

Raw reward: ΔX−κΔX². Scalar: κ=φ(1−nu). Best raw: exact quadratic minimisation in h=1/(2κ) using fixed equilibrium occupancy. Design: Y−φ(1−nu_t)Y² with Y=H_t P_t'u. Conditional-variance reward has the same independent-market optimum (proved; not a distinct learned method here).

Distances: sqrt(sum_t E_reference ||u_candidate(t,state)−u_target(t,state)||²). Main and dependent results use fixed target occupancy. The small-interest theorem and scaling experiments use original candidate occupancy. These are discrepancies, not globally symmetric policy norms. Bounded policies are maps from full return history to portfolio weights, compared on target wealth histories in dollar units.

Independent sweep: T in {1,2,4,8,16}, φ in {.25,.5,1,2,4,16}, s in {.98,1,1.02,1.08}: 120 configurations, six policies each. Market sensitivity: three mean scales (.5,1,1.5), three volatility scales (.75,1,1.25), three off-diagonal correlation scales (0,1,1.5): 27 configurations. All chosen covariance matrices are positive definite. Small-interest: T={2,3,4,6,10,16}, increments={−.001,−.0001,.0001,.001,.01}. Analytic calibration avoids arbitrary optimiser bounds.

Dependent market: two observed regimes, transition [[q,1−q],[1−q,q]], q={.5,.65,.8,.95}; conditional edge means [[.09,−.05],[.04,−.02]]; each edge has symmetric innovation ±.12. Initial regime 0; T=4, s=1.02, φ=1. Returns and next regime are jointly modelled. q=.5 does NOT make returns iid because edge means depend on the current regime. Hedge correction uses known equilibrium continuation. This is not an estimated or model-free algorithm.

Bounded comparison: one risky asset plus cash, T=3, wealth1, s=1.02; excess return −.18 or .30 with probability .5 each. Long-only weight grids {0,.1,...,1} and {0,.05,...,1}. All reward policies have identical history information and feasible actions. Exhaustive reward Bellman solves are exact for the finite grid; the equilibrium uses one-step deviations on that same grid. φ={.5,1,2,4}. No continuous-action convergence rate is asserted. Max |J_21−J_11| is .001218 (raw), .000878 (design), .003595 (DSR), which is material for small J differences. DSR follows Moody–Saffell's unscaled signal with return rho=X_next/X−1, eta=.1, A0=.01,B0=.01. Sensitivity: eta={.02,.1,.4}; (A0,B0)={(0,.001),(.01,.01),(.02,.002),(−.02,.002)}. Positive gross asset returns and long-only actions guarantee positive wealth. Zero costs; undiscounted rewards. Kolm–Ritter's κ=.1 maps to coefficient .05, but its currency units do not transfer. Only a diagnostic.

MVPI: Zhang et al. Algorithm 1 adapted from infinite-horizon discounted occupancy to a uniformly selected date in a finite horizon. Exact inner quadratic solves; y equals E[ΔX] under uniform-time occupancy. Iterate to residual <1e−12; report monotonic objective trace. Numerical convergence is a fixed-point result, not a proof of global mean-volatility optimality at nonzero interest.

Learning: existing affine Gaussian REINFORCE-style routine with Adam, normalised advantages and scalar-rescaled scores; learning rate .02; 4,000 iterations; batch512; seeds0–4; initial coefficients zero; exploration from .3 to .03 (verify defaults in agent.py). Record every200 iterations. The same seed schedules and sample budgets are used for raw/scalar/design. Final deterministic mean policies are evaluated by exact moments; no test-set Monte Carlo noise. Scores are scaled updates, not a full Fisher preconditioner. Reward design has moment knowledge; training does not receive the equilibrium policy. Hyperparameters were not retuned per reward. Five seeds support preliminary diagnostics only. Surrogate regrets use each reward's own units.

Estimation: samples N={50,200,1000,10000}, seeds1000–1029; sample means and unbiased covariance; policies evaluated in the true market. Stress tests: ten seeds900–909 and 50,000 common paths per seed for Gaussian, iid t(8), a common-volatility GARCH-type process and stationary AR(1) coefficient .3. Mean and marginal covariance match; dependence changes conditional opportunities. All policies are frozen from the iid model; iid “pre” and “equilibrium” labels denote their origins, not dependent optimality. t(8) has finite fourth moment, unlike old t(4). Batch standard errors are descriptive Monte Carlo precision, not real-market uncertainty.

## Validation

Original verifier: ten checks. New pytest suite: 25 checks, including an independent optimiser, exhaustive return-tree moments, one-step equilibrium deviations, analytic expansion across horizons, zero-mean and one-period exceptions, time-varying lower bound, DSR continuation counterexample, hedge correction and MVPI monotonicity. An independent two-period finite-action enumeration validates the bounded Bellman reward calculation. The hedge test caught a missing factor; the final formula and output were corrected before reporting.

Scripts: python -m mvrl.verify; python -m pytest -q; python -m mvrl.experiments; PYTHONPATH=. python scripts/make_figures.py; python scripts/make_literature_ledger.py; python scripts/write_research_documents.py; python scripts/render_documents.py. Exact environment versions are recorded in results/environment.txt. Main research dependencies are NumPy/SciPy; tests require pytest; figures require matplotlib; PDFs require reportlab and bundled DejaVu fonts.

## Status and limitations

The central unfinished local proof, time-varying extension, finite observed-regime reference, DSR correction and missing reproducible experiments are completed at the stated scope. Unrestricted DSR, general constraints/costs/dependence, systematic publication-priority review, full published algorithm/historical replications and improved learning guarantees remain OPEN. No publication or external submission has been made.
'''
# Correct nu and exploration straight from current defaults, avoiding stale constants.
from mvrl.lambda_functional import demo_market
import inspect
from mvrl.agent import reinforce
protocol=protocol.replace('0.1943363227',f'{demo_market().nu(0):.10f}').replace('from .3 to .03',f"from {inspect.signature(reinforce).parameters['sigma_initial'].default} to {inspect.signature(reinforce).parameters['sigma_final'].default}")
(ROOT/'IMPLEMENTATION_REPORT.md').write_text(protocol)
(ROOT/'documents.json').write_text(json.dumps({'proposal':proposal,'paper':paper},indent=2,ensure_ascii=False))
# Markdown has the same full content and figure references.
for name,pages in [('proposal',proposal),('paper',paper)]:
 lines=[]
 for pg in pages:
  lines+=['# '+pg['title'],'']
  for b in pg['blocks']:
   if b['type']=='p':lines +=[b['text'],'']
   elif b['type']=='h':lines+=['## '+b['text'],'']
   elif b['type']=='eq':lines+=['```text',b['text'],'```','']
   elif b['type']=='fig':lines+=['!['+b['caption']+']('+b['path']+')',b['caption'],'']
   elif b['type']=='table':
    lines+=['| '+' | '.join(b['head'])+' |','|'+'---|'*len(b['head'])];lines+=['| '+' | '.join(row)+' |' for row in b['rows']];lines+=['']
 (ROOT/(name+'.md')).write_text('\n'.join(lines))
print('Shared proposal, paper and implementation report written.')
