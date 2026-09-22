"""Verify saved evidence and document integrity; exits nonzero on disagreement."""
from pathlib import Path
import json,math,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
r=json.loads((ROOT/'results/experiments.json').read_text())
assert len(r['independent'])==720 and len(r['bounded'])==64 and len(r['learning'])==15
assert len(r['market_sensitivity'])==81 and len(r['mvpi'])==4
for x in r['learning']:
 assert abs(x['total']-x['incons']-x['surr']-x['learn'])<1e-12
 assert x['surrogate_regret']>=-1e-10
for group in ['independent','market_sensitivity']:
 assert max(x['distance'] for x in r[group] if x['name']=='design')<1e-10
assert max(x['distance'] for x in r['dependent'] if x['name']=='hedge_design')<1e-10
for x in r['mvpi']:
 assert x['residual']<1e-12 and min(np.diff(x['trace']))>-1e-12
for x in json.loads((ROOT/'literature/download_manifest.json').read_text()):
 assert hashlib.sha256((ROOT/'literature'/(x['key']+'.pdf')).read_bytes()).hexdigest()==x['sha256']
for name,pages in json.loads((ROOT/'documents.json').read_text()).items():
 for p in pages:
  for b in p['blocks']:
   if b['type']=='fig':assert (ROOT/b['path']).exists()
 import pymupdf
 d=pymupdf.open(ROOT/(name+'.pdf'))
 for i,page in enumerate(d):
  text=page.get_text()
  assert len(text)>400,(name,i+1,'orphan page')
  assert '\ufffd' not in text
  for block in page.get_text('dict')['blocks']:
   if 'lines' not in block:continue
   for line in block['lines']:
    for span in line['spans']:
     x0,y0,x1,y1=span['bbox']
     assert x0>=35 and x1<=page.rect.width-35,(name,i+1,span['text'],'horizontal clipping')
 print(name,len(d),'pages; text and margins verified')
print('Saved experiment counts, signed decomposition, regret, exact recovery, source hashes and document links PASS.')
