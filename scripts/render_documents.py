"""Build genuine LaTeX articles with the original 11pt Computer Modern layout.

Usage: python scripts/render_documents.py [--sources-only] [--engine ENGINE]
ENGINE may be pdflatex or tectonic (also detected on PATH).
Research prose/data remain in documents.json; mathematical proofs are in
latex/proofs.tex. No report renderer or system font substitution is used.
"""
from pathlib import Path
import argparse,json,re,shutil,subprocess
ROOT=Path(__file__).resolve().parents[1]

# Explicit mathematical replacements preserve notation while producing real math.
INLINE={
'T^(3/2)':r'T^{3/2}', '2.35×10⁻¹⁴':r'2.35\times10^{-14}', '5.50×10⁻¹⁵':r'5.50\times10^{-15}',
'−2φHₜu Cov(P,a_next|z)':r'-2\varphi H_tu\allowbreak\operatorname{Cov}(P,a_{t+1}\mid z)',
'B₀−A₀²>0':r'B_0-A_0^2>0', 'r=ΔX−κ(ΔX)²':r'r=\Delta X-\kappa(\Delta X)^2',
'b=1/[2φ(1−ν)]':r'b=1/[2\varphi(1-\nu)]', 'κ=φ(1−ν)':r'\kappa=\varphi(1-\nu)',
'−d(s−1)':r'-d(s-1)', 'd=M⁻¹m':r'd=M^{-1}m', 'ν=mᵀd':r'\nu=m^{\top}d',
'h=1/(2κ)':r'h=1/(2\kappa)', 'm≠0':r'm\ne0', 'T≥2':r'T\ge2',
'κ/2=.05':r'\kappa/2=.05', 'κ=φ':r'\kappa=\varphi', 'φ×distance':r'\varphi\times\text{distance}',
'x₀s^T':r'x_0s^T', 'a_next':r'a_{t+1}', 'u_eq,t':r'u_t^{\mathrm{eq}}',
's=1.02':r's=1.02', 's=1':r's=1', 's−1':r's-1', 'T=4':r'T=4', 'T=3':r'T=3',
'φ=1':r'\varphi=1', 'η=.1':r'\eta=.1', 'A₀=.01':r'A_0=.01', 'B₀=.01':r'B_0=.01',
'x₀=1':r'x_0=1', 'sₜ':r's_t', 'uₜ':r'u_t', 'x₀':r'x_0', 'Pₜ':r'P_t',
'mₜ':r'm_t', 'Σₜ':r'\Sigma_t', 'bₜ':r'b_t', 'c_T':r'c_T', 'D²':r'D^2',
'φ':r'\varphi', 'κ':r'\kappa', 'η':r'\eta', '±':r'\pm', '−':r'-',
}
def escape(s):
 d={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
 return ''.join(d.get(c,c) for c in s).replace('–','--').replace('—','---').replace('‑','-').replace('·',r'\textperiodcentered{}').replace('§',r'\S{}').replace('’',"'").replace('“','``').replace('”',"''")
def text(s):
 s=s.replace('policy-gradient/PPO/DDPG','policy-gradient, PPO and DDPG')
 # Protect URLs, explicit math and numbered citations before text escaping.
 pat='https://[^\\s]+|\\[(?:\\d+)(?:, \\d+)*\\]|'+ '|'.join(re.escape(k) for k in sorted(INLINE,key=len,reverse=True))
 out=[];start=0
 for m in re.finditer(pat,s):
  out.append(escape(s[start:m.start()]));v=m.group()
  if v.startswith('https://'):out.append(r'\url{'+v+'}')
  elif v.startswith('['):out.append(r'\cite{'+','.join('ref'+n.strip() for n in v[1:-1].split(','))+'}')
  else:out.append('$'+INLINE[v]+'$')
  start=m.end()
 out.append(escape(s[start:]));return ''.join(out)

EQUATIONS=[
 r'''\begin{aligned}
 X_{t+1}&=s_tX_t+P_t^{\top}u_t,& J(\pi)&=\E[X_T]-\varphi\Var(X_T),\quad\varphi>0,\\
 M_t&=\Sigma_t+m_tm_t^{\top},& d_t&=M_t^{-1}m_t,\quad\nu_t=m_t^{\top}d_t<1,\\
 H_t&=\prod_{j=t+1}^{T-1}s_j,&u_t^{\mathrm{eq}}&=\frac{\Sigma_t^{-1}m_t}{2\varphi H_t}.
 \end{aligned}''',
 r'''Y_t=H_tP_t^{\top}u_t,\qquad r_t^{\mathrm{design}}=Y_t-\varphi(1-\nu_t)Y_t^2.''',
 r'''\begin{aligned}
 \min_{\kappa>0}D(\pi^{\kappa},\pieq)&=c_T|s-1|+O((s-1)^2),\\
 c_T&=\|d\|b\sqrt{\frac{T(T^2-1)}{3}+\frac{\nu(1-\nu)T(T-1)}{2}}.
 \end{aligned}''',
 r'''\begin{gathered}
 s_t=1:\qquad\min_{\kappa>0}D^2=\sum_{t=0}^{T-1}a_t(b_t-\bar b)^2,\\
 a_t=\|d_t\|^2,\qquad b_t=\frac{1}{2\varphi(1-\nu_t)},\qquad
 \bar b=\frac{\sum_t a_tb_t}{\sum_t a_t}.
 \end{gathered}''',
 r'''u^{\mathrm{eq}}(t,z)=\frac{m_z}{2\varphi H_t\Var(P\mid z)}
 -\frac{\operatorname{Cov}(P,a_{t+1}(z_{t+1})\mid z)}{H_t\Var(P\mid z)}.''',
 r'''\begin{aligned}
 J(\pipre)-J(\pihat)
 &=\underbrace{J(\pipre)-J(\pieq)}_{\text{price of time consistency}}\\
 &\quad+\underbrace{J(\pieq)-J(\pir)}_{\text{signed reward-design difference}}\\
 &\quad+\underbrace{J(\pir)-J(\pihat)}_{\text{signed learning difference}}.
 \end{aligned}'''
]

def preamble(name):
 original=(ROOT/'archive/2026-09-21-before-audit'/f'{name}.tex').read_text()
 p=original.split(r'\title{')[0]
 p=p.replace(r'\newcommand{\pir}{\pi^{\kappa}}',r'\newcommand{\pir}{\pi^{r}}')
 # Retain the original class, font encoding, margins, math and theorem styles.
 p=p.replace(r'\usepackage[hidelinks]{hyperref}',r'\usepackage{longtable,array,url,placeins}'+'\n'+r'\usepackage[hidelinks]{hyperref}')
 if r'\usepackage{graphicx}' not in p:p+=r'\usepackage{graphicx}'+'\n'
 p+=r'''
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\setlength{\emergencystretch}{2em}
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.85}
\renewcommand{\textfraction}{0.1}
\renewcommand{\floatpagefraction}{0.7}
'''
 # Reuse each original title and its original vertical spacing exactly.
 p+=original[original.index(r'\title{'):original.index(r'\date{')]
 date='Research Proposal' if name=='proposal' else 'Working draft --- 22 September 2026'
 p+=r'\date{'+date+'}\n\\begin{document}\n\\maketitle\n'
 p+=r'\vspace{-1.5em}' if name=='proposal' else r'\vspace{-1em}'
 return p+'\n'

def blocks(bs):
 lines=[]
 for b in bs:
  typ=b['type']
  if typ=='p':lines+=[text(b['text']),'']
  elif typ=='h':
   h=b['text'].replace('R2 · ','').replace('R4 · ','').replace('R5 · ','')
   lines+=[r'\subsection{'+text(h)+'}']
  elif typ=='eq':
   prefixes=['Xₜ₊₁','Yₜ','min over','At sₜ','u_eq(t,z)','J(pre)']
   idx=next(i for i,p in enumerate(prefixes) if b['text'].startswith(p))
   lines +=[r'\begin{equation}',EQUATIONS[idx],r'\end{equation}']
  elif typ=='fig':
   cap=re.sub(r'^Figure \d+ · ','',b['caption'])
   path=str(Path(b['path']).with_suffix('.pdf'))
   lines +=[r'\begin{figure}[htbp]',r'\centering',r'\includegraphics[width=\linewidth]{'+path+'}',r'\caption{'+text(cap)+'}',r'\label{fig:'+Path(path).stem+'}',r'\end{figure}']
  elif typ=='table':
   n=len(b['head']);ws=b.get('widths') or [1/n]*n
   # Width expression distributes the total available width, including column gaps.
   spec='@{}'+''.join('L{'+r'\dimexpr '+f'{w:.3f}'+r'\linewidth-'+f'{w*2*(n-1):.3f}'+r'\tabcolsep\relax}' for w in ws)+'@{}'
   head=' & '.join(r'\textbf{'+text(v)+'}' for v in b['head'])+r'\\'
   lines +=[r'\begingroup\small',r'\setlength{\LTpre}{8pt}\setlength{\LTpost}{8pt}',r'\renewcommand{\arraystretch}{1.18}',r'\begin{longtable}{'+spec+'}',r'\toprule',head,r'\midrule\endfirsthead',r'\toprule',head,r'\midrule\endhead',r'\bottomrule\endfoot']
   lines +=[' & '.join(text(str(v)) for v in row)+r'\\[3pt]' for row in b['rows']]
   lines +=[r'\end{longtable}',r'\endgroup','']
 return '\n'.join(lines)

def references(pages,name):
 bs=[b['text'] for pg in pages if pg['title'].startswith('References') for b in pg['blocks']]
 # Keep original paper author-year citations; proposal uses numbered references.
 labels=['Kolm and Ritter(2019)','Jiang et al.(2017)','Yang et al.(2020)','Moody and Saffell(2001)','Bisi et al.(2020)','Zhang et al.(2021)','Wang and Zhou(2020)','Dai et al.(2023)','Li and Ng(2000)','Basak and Chabakauri(2010)','Ng et al.(1999)','Björk and Murgoci(2014)']
 out=[r'\begin{thebibliography}{99}']
 if name=='paper':out.append(r'\small')
 for i in range(0,len(bs),2):
  num=i//2+1
  opt='['+escape(labels[num-1])+']' if name=='paper' else ''
  out += [r'\bibitem'+opt+'{ref'+str(num)+'} '+text(re.sub(r'^\[\d+\] ','',bs[i])),r'\par\smallskip\noindent\emph{Source verification.} '+text(bs[i+1].replace('Version/locator: ','')),'']
 out+=[r'\end{thebibliography}'];return '\n'.join(out)

PROPOSAL_ABSTRACT='''Financial reinforcement learning optimises the reward supplied to it, which need not induce the investor's intended terminal mean-variance policy. This project specifies the target policy first, solves the control problem induced by the reward, and separates reward-design mismatch from learning error. For independent returns and unconstrained risky dollar holdings, we establish the exact local mismatch coefficient of the scalar quadratic-increment reward for every fixed horizon, and construct a terminal-unit reward that recovers the mean-variance equilibrium. Exact benchmarks, adaptations of published reward formulas, and separate learning experiments quantify the scope and limits of the result. Policy alignment improves in the analytic setting, but the evidence does not establish universal improvement in Sharpe, initial mean-variance value, or learned performance. Existing mean-variance policy iteration already recovers the zero-interest scalar calibration; publication priority of the quantitative results remains open. This proposal presents the established results and identifies the remaining research.'''

def build(name,pages):
 out=[preamble(name),r'\begin{abstract}']
 abstract=PROPOSAL_ABSTRACT if name=='proposal' else pages[0]['blocks'][2]['text']
 out +=[r'\noindent '+text(abstract),r'\end{abstract}']
 if name=='proposal':
  out +=[r'\section{Problem, method and established contribution}',blocks(pages[0]['blocks'][1:])]
 else:
  out +=[r'''\begin{center}
\fbox{\begin{minipage}{0.92\textwidth}\small
\textbf{Status.} Working draft. The scoped proofs are established; publication priority,
unrestricted DSR, constraints and costs, and full algorithm replications remain open.
Evidence labels distinguish proved, numerically verified and preliminary results.
\end{minipage}}
\end{center}''',r'\section{Introduction and scope}',blocks(pages[0]['blocks'][3:])]
 for pg in pages[1:]:
  if pg['title'].startswith(('References','Proof appendix')):continue
  title=re.sub(r'^\d+ · ','',pg['title'])
  if title=='Evidence that changes the interpretation':out.append(r'\clearpage')
  if pg['blocks'][0]['type']=='table':out.append(r'\ifdim\dimexpr\pagegoal-\pagetotal\relax<150pt\newpage\fi')
  out +=[r'\section{'+text(title)+'}',blocks(pg['blocks'])]
 if name=='paper':
  out +=[r'\clearpage',r'\appendix',(ROOT/'latex/proofs.tex').read_text()]
 out +=[r'\clearpage',references(pages,name),r'\end{document}']
 source='\n\n'.join(out)+'\n'
 if name=='paper':source=source.replace(r'\cite{',r'\citep{')
 (ROOT/f'{name}.tex').write_text(source)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--sources-only',action='store_true');parser.add_argument('--engine');args=parser.parse_args()
 for name,pages in json.loads((ROOT/'documents.json').read_text()).items():build(name,pages)
 if not args.sources_only:
  engine=args.engine or shutil.which('pdflatex') or shutil.which('tectonic')
  if not engine:raise SystemExit('LaTeX sources generated. Install pdflatex or tectonic, or pass --engine PATH, to compile PDFs.')
  builddir=ROOT/'tmp/latex';builddir.mkdir(parents=True,exist_ok=True)
  for name in ['proposal','paper']:
   if 'tectonic' in Path(engine).name:
    cmd=[engine,'--keep-logs','--outdir',str(builddir),name+'.tex'];runs=1
   else:cmd=[engine,'-interaction=nonstopmode','-halt-on-error','-output-directory='+str(builddir),name+'.tex'];runs=3
   for _ in range(runs):subprocess.run(cmd,cwd=ROOT,check=True)
   shutil.copy2(builddir/f'{name}.pdf',ROOT/f'{name}.pdf')
   shutil.copy2(builddir/f'{name}.log',ROOT/'results'/f'{name}_latex.log')
   print(name,'compiled with',engine)
