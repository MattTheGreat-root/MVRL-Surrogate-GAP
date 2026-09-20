# Primary-paper comparison ledger

Audit date: 21–22 September 2026. Locators refer to the **version actually read**. A manuscript is not silently treated as the published version. Saved PDFs and SHA-256 values are in download_manifest.json. Numerical claims below are reports by the cited authors, not replications by this project.

| ID | Verified citation | Access / version | Role |
|---|---|---|---|
| kolm2019 | Petter N. Kolm; Gordon Ritter (2019). [Dynamic Replication and Hedging: A Reinforcement Learning Approach](https://doi.org/10.3905/jfds.2019.1.1.159). The Journal of Financial Data Science 1(1), 159–171. | 16-page author manuscript at smallake.kr; figures are missing in this manuscript. Text and equations read; journal PDF unavailable. | Closest raw-reward baseline. We isolate policy mismatch in a different allocation benchmark; do not infer failure of their hedger. Published κ/2=.05 is a units-sensitive diagnostic only. |
| jiang2017 | Zhengyao Jiang; Dixing Xu; Jinjun Liang (2017). [A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem](https://arxiv.org/abs/1706.10059). arXiv preprint 1706.10059. | arXiv v2, 16 July 2017; saved PDF. | Adapt log growth on an identical bounded action set. Different target preferences explain disagreement; their own growth objective is not mis-specified. |
| yang2020 | Hongyang Yang; Xiao-Yang Liu; Shan Zhong; Anwar Walid (2020). [Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy](https://doi.org/10.1145/3383455.3422540). Proceedings of the First ACM International Conference on AI in Finance (ICAIF), 8 pages. | Columbia-hosted author manuscript; saved PDF. Do not use the later 2025 arXiv upload as a 2020 identifier. | Adapt value increments with zero costs and undiscounted finite horizon. Measures preference alignment; does not reproduce PPO/DDPG or the historical result. |
| moody2001 | John Moody; Matthew Saffell (2001). [Learning to Trade via Direct Reinforcement](https://doi.org/10.1109/72.935097). IEEE Transactions on Neural Networks 12(4), 875–889. | IDSIA-hosted 17-page manuscript, visually read because text encoding is corrupted. | Adapt unscaled DSR on an augmented (wealth,A,B) tree. Original online gradient update is not identical to maximising an undiscounted sum of DSR signals. |
| bisi2020 | Lorenzo Bisi; Luca Sabbioni; Edoardo Vittori; Matteo Papini; Marcello Restelli (2020). [Risk-Averse Trust Region Optimization for Reward-Volatility Reduction](https://doi.org/10.24963/ijcai.2020/632). Proceedings of IJCAI-20, 4583–4589. | Published proceedings PDF, saved and read. | Direct prior art against broad reward-analysis novelty. We compare its per-step criterion with an independently specified terminal equilibrium; do not call its chosen short-term objective wrong. |
| zhang2021 | Shangtong Zhang; Bo Liu; Shimon Whiteson (2021). [Mean-Variance Policy Iteration for Risk-Averse Reinforcement Learning](https://doi.org/10.1609/aaai.v35i12.17302). Proceedings of AAAI 35(12), 10905–10913. | arXiv:2004.10888v6, 7 April 2022; differs in date from published 2021 proceedings. | Finite-horizon uniform-time adaptation added. At zero interest its iid fixed point equals our scalar calibration: that formula alone is not a defensible novelty claim. |
| wang2020 | Haoran Wang; Xun Yu Zhou (2020). [Continuous-Time Mean-Variance Portfolio Selection: A Reinforcement Learning Framework](https://doi.org/10.1111/mafi.12281). Mathematical Finance 30(4), 1273–1308. | arXiv:1904.11392, saved 39-page preprint; section references refer to this version. | Essential positive control: objective-faithful RL already exists. Our discrete reward-family audit asks a different question; no EMV replication claimed. |
| dai2023 | Min Dai; Yuchao Dong; Yanwei Jia (2023). [Learning Equilibrium Mean-Variance Strategy](https://doi.org/10.1111/mafi.12402). Mathematical Finance 33(4), 1166–1212. | NUS RMI Working Paper 2021-02, March 2021, 52 pages, read via web PDF. Local download blocked; journal section numbering not asserted. | Refutes any first-equilibrium-RL claim. Our target is discrete terminal wealth MV and an audit of common rewards, not their log-MV learning problem. |
| ling2000 | Duan Li; Wan-Lung Ng (2000). [Optimal Dynamic Portfolio Selection: Multiperiod Mean-Variance Formulation](https://doi.org/10.1111/1467-9965.00100). Mathematical Finance 10(3), 387–406. | Journal PDF mirrored at tianjiablog.wordpress.com; saved. | Our precommitment reference is established theory. Reward-design comparisons must specify whether this or equilibrium is the intended target. |
| basak2010 | Suleyman Basak; Georgy Chabakauri (2010). [Dynamic Mean-Variance Asset Allocation](https://doi.org/10.1093/rfs/hhq028). The Review of Financial Studies 23(8), 2970–3016. | Author thesis chapter 2, “Dynamic Mean-Variance Asset Allocation”, read; journal full text unavailable. Chapter numbering is not journal numbering. | Foundational benchmark and hedge precedent. Our finite-regime correction is a discrete audit construction, not the discovery of intertemporal hedging or inverse objectives. |
| ng1999 | Andrew Y. Ng; Daishi Harada; Stuart Russell (1999). [Policy Invariance under Reward Transformations: Theory and Application to Reward Shaping](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf). Proceedings of ICML, 278–287. | Author-hosted 10-page conference paper; visually read due to encoding. No DOI asserted. | Our redesign deliberately changes the raw reward optimum. Policy-invariant shaping cannot by itself correct a mismatched target. |
| bjork2014 | Tomas Björk; Agatha Murgoci (2014). [A Theory of Markovian Time-Inconsistent Stochastic Control in Discrete Time](https://doi.org/10.1007/s00780-014-0234-y). Finance and Stochastics 18(3), 545–592. | Publisher metadata and abstract only; full-text verification OPEN. Not counted as a paper fully read. | Requires conservative novelty: existence of an equivalent reward problem is prior art. Exact theorem-level comparison remains open. |

## kolm2019

| Requested field | Extraction |
|---|---|
| locator | “Automatic hedging in theory”, manuscript pp. 7–8, Eqs. (8)–(10), footnote 4; “Automatic hedging in practice”, pp. 9–14, Exhibits 1–5. |
| problem | Discrete option hedging with nonlinear trading costs. |
| objective | Terminal wealth mean–variance, then a simplified sum of increment variances. |
| reward | Δw − (κ/2)(Δw)², explicitly approximate. |
| algorithm | Batch fitted action-value learning with one-step SARSA targets and nonlinear regression. |
| market | Simulated GBM European call; strike/spot 100, 10 days, 5 decisions/day; κ=.1. |
| metric | Hedging P&L, costs, volatility; comparison to delta hedging. |
| result | Reports lower cost across 10,000 test paths; volatility difference not significant at 99%. |
| studies reward objective | Yes: explicitly discusses covariance and approximation. |
| time inconsistency | Does not formulate a subgame equilibrium in the inspected derivation. |
| known benchmark | BSM hedge in frictionless limit; not a solved frictional optimum. |
| project relation | Closest raw-reward baseline. We isolate policy mismatch in a different allocation benchmark; do not infer failure of their hedger. Published κ/2=.05 is a units-sensitive diagnostic only. |

## jiang2017

| Requested field | Extraction |
|---|---|
| locator | §§2.2–2.4, §4.2 Eqs. (21)–(22), §4.3; Tables 1–2; §6. |
| problem | Long-only cryptocurrency allocation with costs. |
| objective | Portfolio growth. |
| reward | Average log portfolio growth including transaction-cost factor. |
| algorithm | Direct gradient ascent; EIIE CNN/RNN/LSTM policies and portfolio-vector memory. |
| market | Poloniex, 11 noncash assets plus Bitcoin cash, 30-minute decisions; three 2016–2017 backtests. |
| metric | Final accumulated value, Sharpe, maximum drawdown. |
| result | EIIE variants report leading accumulated values in all three tests (Table 2). |
| studies reward objective | Yes: log reward corresponds to multiplicative growth. |
| time inconsistency | No terminal mean–variance equilibrium objective. |
| known benchmark | Trading comparators, no known true market optimum. |
| project relation | Adapt log growth on an identical bounded action set. Different target preferences explain disagreement; their own growth objective is not mis-specified. |

## yang2020

| Requested field | Extraction |
|---|---|
| locator | §3.3 Eq. (4); §5 ensemble selection; §6.1 and Fig. 4; §6.2.4 and Table 2. |
| problem | Automated constrained stock trading. |
| objective | Portfolio value growth with costs and turbulence control. |
| reward | Change in portfolio value net of transaction costs. |
| algorithm | PPO, A2C, DDPG; rolling Sharpe-based ensemble selection. |
| market | Dow 30 daily WRDS data; training 2009–2015, testing Jan. 2016–8 May 2020. |
| metric | Return, volatility, drawdown, Sharpe. |
| result | Table 2: ensemble Sharpe 1.30, DJIA .47, minimum variance .45. |
| studies reward objective | Defines value-change objective; no identity with terminal MV. |
| time inconsistency | No equilibrium mean–variance treatment. |
| known benchmark | DJIA and minimum-variance comparator, no true dynamic optimum. |
| project relation | Adapt value increments with zero costs and undiscounted finite horizon. Measures preference alignment; does not reproduce PPO/DDPG or the historical result. |

## moody2001

| Requested field | Extraction |
|---|---|
| locator | §II-D, Eqs. (12)–(17), PDF pp. 3–4; §III-A Eq. (31); §IV-A, Fig. 5; §IV-C.2, Fig. 8. |
| problem | Trading and stock/T-bill allocation. |
| objective | Risk-adjusted trading performance. |
| reward | Differential Sharpe Eq. (14), EWMA updates Eq. (15); also differential downside-deviation reward. |
| algorithm | Recurrent reinforcement learning; Q-learning/advantage updating comparisons. |
| market | Synthetic prices; USD/GBP; monthly S&P500/T-bills with 1970–1994 test period. |
| metric | Profit and Sharpe; downside risk in FX example. |
| result | §IV-C.2 reports annualised Sharpe .83 for RRL, .63 for Q-trader, .34 buy-and-hold. |
| studies reward objective | Yes: derives DSR as first-order Sharpe change. |
| time inconsistency | No terminal MV equilibrium equivalence. |
| known benchmark | Simple market examples; no known optimal real-market policy. |
| project relation | Adapt unscaled DSR on an augmented (wealth,A,B) tree. Original online gradient update is not identical to maximising an undiscounted sum of DSR signals. |

## bisi2020

| Requested field | Extraction |
|---|---|
| locator | §3 Eqs. (3)–(6); §§4–5, Algorithm 1; §7.1–7.2, Fig. 1(a)–(d). |
| problem | Trading with short-term risk control. |
| objective | Expected reward minus occupancy-weighted per-step reward variance. |
| reward | Trading P&L minus turnover costs; policy-dependent centred quadratic transformation. |
| algorithm | Trust Region Volatility Optimization (TRVO), extending TRPO. |
| market | Daily S&P500 through 2019; minute USD/EUR and USD/JPY, train 2017/test 2018. |
| metric | Mean-volatility and mean-variance frontiers. |
| result | Fig. 1 reports broader/frontier improvements over exponential rewards; S&P results are in-sample. |
| studies reward objective | Yes: reward-volatility Bellman/gradient theory and return-variance bound. |
| time inconsistency | Distinguishes risk measures; does not target the terminal-MV subgame equilibrium here. |
| known benchmark | No closed-form optimum for historical tasks. |
| project relation | Direct prior art against broad reward-analysis novelty. We compare its per-step criterion with an independently specified terminal equilibrium; do not call its chosen short-term objective wrong. |

## zhang2021

| Requested field | Extraction |
|---|---|
| locator | “Mean-Variance RL”, Eq. (2); “Mean-Variance Policy Iteration”, Eqs. (3)–(4), Algorithm 1, Propositions 1–2; “Experiments”, Table 1. |
| problem | Scalable risk-averse RL. |
| objective | Discounted-occupancy per-step mean–variance. |
| reward | Transformed reward r − λr² + 2λry; y updated by policy evaluation. |
| algorithm | MVPI, instantiated with TD3; offline density-ratio variant. |
| market | Eight noisy-action MuJoCo tasks and an offline benchmark; not financial prices. |
| metric | Risk-aware score, mean, variance, Sharpe-like ratio. |
| result | Reports strong risk-aware results versus TD3 and risk-averse baselines; not universal dominance on every metric. |
| studies reward objective | Yes, explicitly distinguishes per-step and total-return variance. |
| time inconsistency | No financial subgame-equilibrium target. |
| known benchmark | Small illustrative MDP; no closed-form MuJoCo optimum. |
| project relation | Finite-horizon uniform-time adaptation added. At zero interest its iid fixed point equals our scalar calibration: that formula alone is not a defensible novelty claim. |

## wang2020

| Requested field | Extraction |
|---|---|
| locator | §2 Eq. (11); §3 Theorems 1 and 3; §4; §5.1 Table 1, §§5.2–5.3. |
| problem | Learn continuous-time mean–variance allocation. |
| objective | Precommitted terminal MV via a mean constraint and quadratic embedding. |
| reward | Terminal squared deviation plus entropy regularisation. |
| algorithm | Exploratory mean–variance (EMV), policy evaluation/improvement. |
| market | Simulated GBM parameter grid and nonstationary simulations. |
| metric | Terminal return, variance, Sharpe; algorithm comparisons. |
| result | §5.1 reports EMV best Sharpe in 23 of 28 tested cases. |
| studies reward objective | Yes: objective and optimal exploratory control derived. |
| time inconsistency | Uses precommitment; not an equilibrium solver. |
| known benchmark | Closed-form Gaussian exploratory policy and classical limit. |
| project relation | Essential positive control: objective-faithful RL already exists. Our discrete reward-family audit asks a different question; no EMV replication claimed. |

## dai2023

| Requested field | Extraction |
|---|---|
| locator | Working version §§2–3, §4.2 Theorem 2; §5 Algorithm 1; §6.1–6.2, Fig. 1. |
| problem | Learn equilibrium allocation under stochastic investment opportunities. |
| objective | Mean–variance of log returns plus entropy. |
| reward | Objective-specific equilibrium TD/policy updates, not arbitrary additive wealth increments. |
| algorithm | Exploratory equilibrium policy iteration and parameterised learning. |
| market | Simulated Gaussian mean-return model; perfect/noisy factor observations. |
| metric | Policy deviation from theoretical solution and convergence. |
| result | Fig. 1 shows learned policies close to the analytic target, with increased error under observation noise. |
| studies reward objective | Yes. |
| time inconsistency | Yes, explicitly central. |
| known benchmark | Semi-analytic equilibrium; closed form in Gaussian factor model. |
| project relation | Refutes any first-equilibrium-RL claim. Our target is discrete terminal wealth MV and an audit of common rewards, not their log-MV learning problem. |

## ling2000

| Requested field | Extraction |
|---|---|
| locator | §§3–4, Theorems 1–2 (embedding); §5 riskless-asset case. |
| problem | Multiperiod portfolio selection. |
| objective | Terminal mean–variance at the initial date. |
| reward | No RL reward; auxiliary quadratic terminal objective. |
| algorithm | Embedding and stochastic linear-quadratic control. |
| market | Independent, potentially time-varying return moments. |
| metric | Efficient frontier and terminal mean/variance. |
| result | Analytic optimal policies/frontiers via embedding. |
| studies reward objective | Analyses the objective transformation, not additive RL rewards. |
| time inconsistency | Supplies precommitted optimum, not subgame equilibrium. |
| known benchmark | Yes, core exact reference. |
| project relation | Our precommitment reference is established theory. Reward-design comparisons must specify whether this or equilibrium is the intended target. |

## basak2010

| Requested field | Extraction |
|---|---|
| locator | Thesis chapter 2 §2.2.2, Proposition 2.1, Eqs. (148)–(149); §2.2.2 Remark 1; §2.2.4; discrete-time extension in §2.4. |
| problem | Dynamic allocation under stochastic opportunities. |
| objective | Time-consistent terminal wealth mean–variance. |
| reward | No RL reward; equilibrium control and associated utility interpretation. |
| algorithm | Recursive equilibrium analysis and analytic specialisations. |
| market | Continuous-time incomplete-market models; discrete-time extensions. |
| metric | Policy, hedging demand, terminal moments. |
| result | Decomposes risky demand into myopic and intertemporal hedging components. |
| studies reward objective | Discusses objectives giving the same policy (Remark 1). |
| time inconsistency | Yes. |
| known benchmark | Analytic equilibria in tractable market models. |
| project relation | Foundational benchmark and hedge precedent. Our finite-regime correction is a discrete audit construction, not the discovery of intertemporal hedging or inverse objectives. |

## ng1999

| Requested field | Extraction |
|---|---|
| locator | §3 Theorem 1, Eq. (2), Corollary 2; §4 Figures 1–2; Appendix A. |
| problem | Reward shaping without changing optimal policies. |
| objective | Expected discounted reward. |
| reward | Potential difference γΦ(s′)−Φ(s). |
| algorithm | Policy-invariance theorem; SARSA gridworld examples. |
| market | Synthetic gridworlds, not markets. |
| metric | Policy invariance and learning steps. |
| result | Potential shaping preserves optimal policies under stated conditions and can accelerate learning. |
| studies reward objective | Yes. |
| time inconsistency | Not a mean–variance equilibrium treatment. |
| known benchmark | Known small MDPs. |
| project relation | Our redesign deliberately changes the raw reward optimum. Policy-invariant shaping cannot by itself correct a mismatched target. |

## bjork2014

| Requested field | Extraction |
|---|---|
| locator | Publisher abstract only; no theorem number or detailed comparison asserted. |
| problem | Markovian time-inconsistent control. |
| objective | General time-inconsistent criteria. |
| reward | Not extracted from full text. |
| algorithm | Game-theoretic extended dynamic programming (abstract). |
| market | General control setting (abstract). |
| metric | Equilibrium policies and values (abstract). |
| result | Abstract states an associated time-consistent problem can reproduce equilibrium control and value. |
| studies reward objective | Abstract supports existence of an equivalent control problem. |
| time inconsistency | Yes. |
| known benchmark | Not independently extracted. |
| project relation | Requires conservative novelty: existence of an equivalent reward problem is prior art. Exact theorem-level comparison remains open. |

## Search scope and unresolved access

Primary papers were located through author/institution repositories, arXiv, conference proceedings and publisher records. Search phrases included financial RL portfolio reward mean–variance, differential Sharpe, equilibrium mean–variance RL, reward-volatility, mean–variance policy iteration, and reward calibration/mismatch. This is a targeted relevance search, not an exhaustive systematic review or a proof of novelty. Search results and cited references led to the Bisi/Zhang additions, which materially weaken a novelty claim based only on scalar calibration.

Ritter (2017), “Machine Learning for Trading”, SSRN 3015609, DOI 10.2139/ssrn.3015609: author PDF returned 403. No unverified formulation or result is attributed to it; Kolm–Ritter's inspected manuscript is used directly instead. Björk–Murgoci full text and journal-version concordance for thesis/working-paper sources remain open.

The archived DeMiguel reference used the wrong year (2015 instead of 2014 for RFS volume 27). It is omitted from the revised argument because its primary paper was not needed/read. Broad survey-based assertions and unsupported “first” claims are removed. Basak–Chabakauri's prior associated-objective remark and Dai–Dong–Jia's equilibrium learner must be considered before any publication-priority claim.

## Adaptation rules

Published reward formulas are tested in a common controlled market, not their original data or algorithm. No result here replicates published backtest performance. Quadratic coefficient .05 comes from Kolm–Ritter's κ=.1 and Eq. (10); changing wealth units changes its meaning. It is a sensitivity diagnostic, not evidence of an inferior published hyperparameter. Main calibrated comparisons share the same φ and units. Zhang MVPI is adapted from discounted infinite-horizon occupancy to uniformly weighted finite-horizon increments; its fixed-point convergence is checked, but global optimality for the new nonlinear criterion is not claimed. DSR uses the full augmented finite-tree state; the paper's online local-gradient algorithm is not reproduced.
