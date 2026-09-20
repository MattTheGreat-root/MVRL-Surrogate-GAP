"""Download exact public primary-paper versions used in the source ledger."""
from pathlib import Path
import urllib.request,hashlib,json
SOURCES={
 'zhang2021':'https://arxiv.org/pdf/2004.10888',
 'bisi2020':'https://www.ijcai.org/proceedings/2020/0632.pdf',
 'dai2021version':'https://rmi.nus.edu.sg/wp-content/uploads/2021/04/RMI-WPS-2021-02.pdf',
 'basak_thesis_version':'https://lbsresearch.london.edu/id/eprint/2464/1/Thesis_George_Chabakauri.pdf',
 'ritter2017':'https://cims.nyu.edu/~ritter/ritter2017machine.pdf',
 'kolm2019':'https://smallake.kr/wp-content/uploads/2020/11/ReinforcementOptionsJournalVersion.pdf',
 'jiang2017':'https://arxiv.org/pdf/1706.10059',
 'yang2020':'https://openfin.engineering.columbia.edu/sites/openfin.engineering.columbia.edu/files/content/publications/ensemble.pdf',
 'moody2001':'https://people.idsia.ch/~juergen/rnnaissance2003talks/MoodySaffellTNN01.pdf',
 'ling2000':'https://tianjiablog.wordpress.com/wp-content/uploads/2019/12/optimal-dynamic-portfolio-selection-multiperiod-mean-variance-formulation.pdf',
 'wang2020':'https://arxiv.org/pdf/1904.11392',
 'ng1999':'https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf',
}
if __name__=='__main__':
 import fitz
 root=Path(__file__).parent;rows=[]
 for key,url in SOURCES.items():
  p=root/(key+'.pdf')
  try:
   if not p.exists():
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
    with urllib.request.urlopen(req,timeout=60) as r:p.write_bytes(r.read())
   doc=fitz.open(p);(root/(key+'.txt')).write_text('\n'.join(f'\n=== PDF PAGE {i+1} ===\n'+page.get_text() for i,page in enumerate(doc)))
   rows.append(dict(key=key,url=url,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),pages=len(doc)));print(key,len(doc),'pages',flush=True)
  except Exception as e:print(key,type(e).__name__,str(e),flush=True)
 (root/'download_manifest.json').write_text(json.dumps(rows,indent=2))
