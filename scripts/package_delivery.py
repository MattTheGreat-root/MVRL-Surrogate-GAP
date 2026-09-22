"""Synchronise parent delivery copies; retain original ZIP and archived sources."""
from pathlib import Path
import shutil,zipfile,hashlib,json
ROOT=Path(__file__).resolve().parents[1];parent=ROOT.parent
for name in ['proposal.pdf','paper.pdf','WORKLOG.md','FINDINGS.md','IMPLEMENTATION_REPORT.md']:
 shutil.copy2(ROOT/name,parent/name)
for name in ['proposal.tex','paper.tex']:
 (parent/name).write_text((ROOT/name).read_text().replace('{figures/','{mvrl-surrogate-gap/figures/'))
for name in ['proposal.md','paper.md']:
 (parent/name).write_text((ROOT/name).read_text().replace('](figures/','](mvrl-surrogate-gap/figures/'))
for module in ['dsr','phase5']:
 (parent/(module+'.py')).write_text('''"""Compatibility entry point. Original standalone script is archived."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'mvrl-surrogate-gap'))
from mvrl.'''+module+''' import main
if __name__=='__main__': main()
''')
(parent/'README.md').write_text('''# Current research delivery — 22 September 2026

Start with **proposal.pdf** and **FINDINGS.md**. **paper.pdf** includes the proof appendix. Editable Markdown and compiled LaTeX sources are provided. Both PDFs use the original 11pt A4 article format, one-inch margins and Computer Modern fonts.

The authoritative source package is **mvrl-surrogate-gap/**. It contains the full code, 25 tests, 10 original verification checks, saved experiment outputs, ten figure pairs, exact primary-paper ledger, implementation report, source documents and reproduction scripts. Root copies are generated delivery copies.

**mvrl-surrogate-gap-revised-2026-09-22.zip** is the revised complete package. **mvrl-surrogate-gap.zip** is the original historical delivery and is preserved unchanged. Older scripts and documents remain in the package archive, with stale claims clearly superseded.

Main result: exact local mismatch of the scalar quadratic-increment reward and an independent-return reward redesign that recovers mean–variance equilibrium under stated assumptions. Existing MVPI recovers the zero-interest scalar calibration, so that formula alone is not claimed as novel. Learned performance superiority, broad constrained/dependent generalisation and publication priority remain unestablished.
''')
files=[p for p in ROOT.rglob('*') if p.is_file() and not any(x in p.parts for x in ['__pycache__','.pytest_cache','tmp']) and p.name!='.DS_Store' and p.name!='delivery_manifest.json']
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
(ROOT/'results/delivery_manifest.json').write_text(json.dumps(manifest,indent=2))
files.append(ROOT/'results/delivery_manifest.json')
archive=parent/'mvrl-surrogate-gap-revised-2026-09-22.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for p in files:z.write(p,str(Path(ROOT.name)/p.relative_to(ROOT)))
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
print('Delivery synced;',len(files),'files;',round(archive.stat().st_size/1e6,2),'MB; ZIP integrity PASS.')

# Small, self-contained source bundle for a standard LaTeX/Overleaf project.
with zipfile.ZipFile(parent/'latex-original-format.zip','w',zipfile.ZIP_DEFLATED) as z:
 for name in ['proposal.tex','paper.tex']:
  z.write(ROOT/name,name)
 for p in sorted((ROOT/'figures').glob('*.pdf')):
  z.write(p,str(p.relative_to(ROOT)))
 z.writestr('README.txt',"Both documents use the original 11pt A4 article layout, one-inch margins, and Computer Modern fonts.\n\nSelect proposal.tex or paper.tex as the main document and use pdfLaTeX. Compile three times to resolve citations. In Overleaf, upload this ZIP and choose the main document in project settings.\n\nEach .tex file is complete, with an inline bibliography and all proof text. The figures/ directory contains the required vector PDF figures. No Python, system fonts, external bibliography file, or network access is required with a standard TeX Live installation.\n")
print('Portable LaTeX source bundle created: latex-original-format.zip')
