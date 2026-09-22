"""Scientific figures from saved data; never fits values to a desired conclusion."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mvrl.lambda_functional import demo_market
from mvrl.surrogate import surrogate_backward
from mvrl.design import designed_policy
from mvrl.equilibrium import solve_equilibrium
ROOT=Path(__file__).resolve().parents[1];data=json.loads((ROOT/'results/experiments.json').read_text());out=ROOT/'figures';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'white','savefig.facecolor':'white','axes.labelcolor':'#263747','text.color':'#263747','axes.prop_cycle':plt.cycler(color=['#c7683e','#e0a23b','#16847e','#59677d','#a267a8'])})
colors={'raw':'#c7683e','scalar':'#e0a23b','design':'#16847e','equilibrium':'#263747','pre':'#a267a8','best_raw':'#59677d','return':'#a65c8d','log':'#6f77b5','dsr':'#82934a','published_quadratic':'#99816c','local_calibration':'#e0a23b','hedge_design':'#16847e'}
labels={'raw':'Raw: κ=φ','scalar':'Scalar calibration','design':'Terminal-unit design','equilibrium':'Equilibrium','pre':'Precommitment','best_raw':'Best raw κ','return':'Return','log':'Log growth','dsr':'DSR','published_quadratic':'Published κ/2=.05','local_calibration':'Local calibration','hedge_design':'With hedge correction'}
def save(fig,name):
 fig.savefig(out/(name+'.pdf'),bbox_inches='tight');fig.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
def select(key,**filters):return [r for r in data[key] if all(r[k]==v for k,v in filters.items())]
#1
fig,axs=plt.subplots(1,3,figsize=(12,3.6),layout='constrained');mk=demo_market(1.02,4);x=np.linspace(.5,2,100)
pols={'raw':surrogate_backward(mk,1),'scalar':surrogate_backward(mk,1-mk.nu(0)),'design':designed_policy(mk,1)}
for name,p in pols.items():axs[0].plot(x,p.alpha[0,0]*x+p.beta[0,0],label=labels[name],color=colors[name])
axs[0].set(xlabel='Current wealth',ylabel='Risky dollars: asset 1',title='A. What action does reward induce?');axs[0].legend(fontsize=8)
rows=select('independent',T=4,phi=1.,s=1.02);names=['raw','scalar','best_raw','design'];r={q['name']:q for q in rows};pos=np.arange(4)
axs[1].bar(pos,[r[n]['distance'] for n in names],color=[colors[n] for n in names]);axs[1].set(xticks=pos,xticklabels=['Raw','Scalar','Best κ','Design'],ylabel='Distance to equilibrium',title='B. Alignment improves')
for i,n in enumerate(names):axs[1].text(i,r[n]['distance']+.015,f"{r[n]['distance']:.3f}",ha='center',fontsize=9)
axs[2].bar(pos,[r[n]['J']-r['equilibrium']['J'] for n in names],color=[colors[n] for n in names]);axs[2].axhline(0,color='#263747',lw=.8);axs[2].set(xticks=pos,xticklabels=['Raw','Scalar','Best κ','Design'],ylabel='J(policy) − J(equilibrium)',title='C. Alignment ≠ larger initial J')
save(fig,'01_policy_alignment')
#2
fig,axs=plt.subplots(1,2,figsize=(10.5,3.7),layout='constrained')
for T in [2,4,10,16]:
 rows=sorted(select('scaling',T=T),key=lambda r:r['e']);rows=[r for r in rows if r['e']>0]
 axs[0].loglog([r['e'] for r in rows],[r['distance'] for r in rows],'o-',label=f'T={T}')
 axs[0].loglog([r['e'] for r in rows],[r['prediction'] for r in rows],'--',alpha=.6)
rows=select('scaling',e=.0001);ts=[r['T'] for r in rows];cs=[r['c'] for r in rows];axs[1].plot(ts,cs,'o-',label='Proved coefficient c(T)');axs[1].plot(ts,np.array(ts)**1.5*cs[-1]/ts[-1]**1.5,'--',label='T^(3/2) guide')
axs[0].set(xlabel='Interest increment |s−1|',ylabel='Best raw reward distance',title='A. Exact values and local prediction');axs[0].legend(fontsize=8);axs[1].set(xlabel='Horizon T',ylabel='c(T)',title='B. Horizon dependence is superlinear');axs[1].legend(fontsize=8);save(fig,'02_horizon_theorem')
#3
fig,axs=plt.subplots(1,3,figsize=(12,3.7),layout='constrained')
for ax,name in zip(axs,['raw','scalar','design']):
 phis=[.25,.5,1.,2.,4.,16.];ss=[.98,1.,1.02,1.08];arr=np.array([[select('independent',T=4,phi=p,s=s,name=name)[0]['distance']*p for s in ss] for p in phis])
 im=ax.imshow(arr,vmin=0,vmax=11,cmap='YlOrBr',aspect='auto');ax.set(xticks=range(4),xticklabels=ss,yticks=range(6),yticklabels=phis,xlabel='Gross risk-free return s',ylabel='Risk aversion φ',title=labels[name]);
 for i in range(6):
  for j in range(4):ax.text(j,i,f'{arr[i,j]:.2f}',ha='center',va='center',fontsize=8,color='black' if arr[i,j]<6 else 'white')
fig.colorbar(im,ax=axs,label='φ × policy distance (common scale)',shrink=.85);save(fig,'03_risk_regimes')
#4
fig,axs=plt.subplots(1,2,figsize=(12,4.7),layout='constrained');names=['return','log','dsr','published_quadratic','raw','scalar','design','equilibrium'];rows={r['name']:r for r in select('bounded',phi=1.,grid=21)};names=[n for n in names if n in rows];pos=np.arange(len(names))
for ax,key,title in zip(axs,['distance','J'],['A. Exact reward optima: policy alignment','B. Terminal mean–variance objective']):
 ax.barh(pos,[rows[n][key] for n in names],color=[colors[n] for n in names]) if key=='distance' else ax.scatter([rows[n][key] for n in names],pos,c=[colors[n] for n in names],s=55)
 ax.set(yticks=pos,yticklabels=[labels[n] for n in names],xlabel=key,title=title);ax.invert_yaxis()
 if key=='J':ax.set_xlim(.99,1.12)
fig.suptitle('Published reward adaptations • identical long-only action grid • not historical replications',fontsize=11);save(fig,'04_published_rewards')
#5
fig,axs=plt.subplots(1,2,figsize=(10.5,3.7),layout='constrained')
for name in ['raw','local_calibration','hedge_design']:
 rows=sorted(select('dependent',name=name),key=lambda r:r['persistence']);qs=[r['persistence'] for r in rows];axs[0].plot(qs,[r['distance'] for r in rows],'o-',label=labels[name],color=colors[name]);axs[1].plot(qs,[r['J']-select('dependent',name='equilibrium',persistence=r['persistence'])[0]['J'] for r in rows],'o-',color=colors[name])
axs[0].set(xlabel='Regime persistence',ylabel='Distance to regime equilibrium',title='A. Conditional moments miss a hedge');axs[0].legend(fontsize=8);axs[1].axhline(0,color='gray',lw=.8);axs[1].set(xlabel='Regime persistence',ylabel='J(policy) − J(equilibrium)',title='B. Matching equilibrium has a specific goal');save(fig,'05_dependent_returns')
#6
fig,axs=plt.subplots(1,3,figsize=(12,3.8),layout='constrained');names=['raw','scalar','design'];pos=np.arange(3)
for ax,key,title in zip(axs,['distance','J','surrogate_regret'],['A. Learned alignment','B. Learned financial objective','C. Optimisation error in own reward']):
 vals=[[r[key] for r in select('learning',name=n)] for n in names]
 if key=='J':
  for i,v in enumerate(vals):ax.errorbar(i,np.mean(v),yerr=np.std(v,ddof=1),fmt='o',color=colors[names[i]],capsize=5,ms=7)
 else:ax.bar(pos,[np.mean(v) for v in vals],yerr=[np.std(v,ddof=1) for v in vals],color=[colors[n] for n in names],capsize=4,alpha=.75)
 for i,v in enumerate(vals):ax.scatter(i+np.linspace(-.12,.12,len(v)),v,color='#263747',s=12,zorder=3)
 ax.set(xticks=pos,xticklabels=['Raw','Scalar','Design'],ylabel=key,title=title)
 if key=='J':ax.set_ylim(1.28,1.32)
fig.suptitle('Five matched seeds • 4,000 REINFORCE iterations • bars ± sample SD',fontsize=11);save(fig,'06_learning_error')
#7
fig,axs=plt.subplots(1,2,figsize=(10.5,3.8),layout='constrained');rows={r['name']:r for r in select('independent',T=4,phi=1.,s=1.02)}
for name in ['raw','scalar','design']:
 r=rows[name];axs[0].scatter(r['sharpe'],r['J'],s=80,color=colors[name]);axs[0].annotate(labels[name],(r['sharpe'],r['J']),xytext=(5,3 if name!='raw' else -13),textcoords='offset points',fontsize=8)
axs[0].set(xlabel='Terminal excess Sharpe (not annualised)',ylabel='Terminal J',title='A. Raw beats target on Sharpe, loses on J');axs[0].margins(x=.35,y=.2)
for eta in [.02,.1,.4]:
 rows=select('dsr',eta=eta);axs[1].plot(range(4),[r['initial_weight'] for r in rows],'o-',label=f'η={eta}')
axs[1].set(xticks=range(4),xticklabels=['0 / .001','.01 / .01','.02 / .002','−.02 / .002'],xlabel='Initial A / B',ylabel='Initial risky weight',title='B. DSR depends on its internal state');axs[1].legend(fontsize=8);save(fig,'07_sharpe_dsr')
#8
fig,axs=plt.subplots(1,2,figsize=(11,3.8),layout='constrained')
for name in ['raw','scalar','design']:
 Ns=[50,200,1000,10000];v=[[r['distance'] for r in select('estimation',name=name,N=N)] for N in Ns];axs[0].errorbar(Ns,[np.mean(q) for q in v],yerr=[np.std(q,ddof=1) for q in v],fmt='o-',label=labels[name],color=colors[name],capsize=3)
axs[0].set(xscale='log',xlabel='Moment estimation sample size',ylabel='Distance to true equilibrium',title='A. Exact calibration still needs estimates');axs[0].legend(fontsize=8)
kinds=['gaussian','student-t','vol-clustering','momentum'];names=['raw','design','pre']
for i,name in enumerate(names):
 vals=[[r['J'] for r in select('robustness',name=name,kind=k)] for k in kinds];axs[1].errorbar(np.arange(4)+(i-1)*.12,[np.mean(v) for v in vals],yerr=[np.std(v,ddof=1) for v in vals],fmt='o',label=labels[name],color=colors[name],capsize=3)
axs[1].set(xticks=range(4),xticklabels=['Gaussian','t(8)','GARCH','AR(1)'],ylabel='Terminal J',title='B. Frozen-policy stress tests');axs[1].legend(fontsize=8);save(fig,'08_estimation_robustness')
#9
fig,axs=plt.subplots(1,2,figsize=(10.5,3.8),layout='constrained')
for i,name in enumerate(['raw','scalar','design']):
 rows=select('learning',name=name);means=[np.mean([r[k] for r in rows]) for k in ['incons','surr','learn']];axs[0].bar(np.arange(3)+(i-1)*.24,means,width=.24,label=labels[name],color=colors[name])
axs[0].axhline(0,color='gray',lw=.8);axs[0].set(xticks=range(3),xticklabels=['Pre − equilibrium','Equilibrium − reward','Reward − learned'],ylabel='Signed terminal-J difference',title='A. Separate three sources of difference');axs[0].legend(fontsize=8)
rows=data.get('market_sensitivity',[])
for i,name in enumerate(['raw','scalar','design']):
 v=sorted([r['distance'] for r in rows if r['name']==name]);axs[1].plot(range(len(v)),v,label=labels[name],color=colors[name])
axs[1].set(xlabel='Ordered configuration (27 mean/vol/correlation cases)',ylabel='Distance to equilibrium',title='B. Robustness across market moments');axs[1].legend(fontsize=8);save(fig,'09_decomposition_markets')
# Ten figures are saved, including the MVPI comparison below.
fig,axs=plt.subplots(1,2,figsize=(10.5,3.8),layout='constrained')
for name in ['raw','scalar','design']:
 rows=sorted(select('independent',T=4,phi=1.,name=name),key=lambda r:r['s']);axs[0].plot([r['s'] for r in rows],[r['distance'] for r in rows],'o-',label=labels[name],color=colors[name])
rows=data['mvpi'];axs[0].plot([r['s'] for r in rows],[r['distance'] for r in rows],'s--',color='#6260a8',label='MVPI adaptation');axs[0].set(xlabel='Gross risk-free return s',ylabel='Distance to equilibrium',title='A. Existing transformation matches at s=1');axs[0].legend(fontsize=8)
rows=data['time_varying'];axs[1].bar(range(3),[r['distance'] for r in rows],color=[colors[r['name']] for r in rows]);axs[1].set(xticks=range(3),xticklabels=['Raw','Best scalar','Date-specific design'],ylabel='Distance to equilibrium',title='B. Changing moments defeat a common κ');save(fig,'10_existing_transformations')
