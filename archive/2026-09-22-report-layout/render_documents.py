"""Render shared documents.json to PDFs and editable XeLaTeX sources."""
from pathlib import Path
import json,re,html
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle,Image,KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT
from PIL import Image as PILImage
ROOT=Path(__file__).resolve().parents[1]
# matplotlib ships these fonts on ordinary environments too.
import matplotlib
fontdir=Path(matplotlib.get_data_path())/'fonts/ttf'
for name,file in [('Body','DejaVuSans.ttf'),('Bold','DejaVuSans-Bold.ttf'),('Italic','DejaVuSans-Oblique.ttf')]:pdfmetrics.registerFont(TTFont(name,str(fontdir/file)))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='Bold',italic='Italic',boldItalic='Bold')
ink=colors.HexColor('#263747');teal=colors.HexColor('#137d78');muted=colors.HexColor('#586879')
styles={
'p':ParagraphStyle('p',fontName='Body',fontSize=9.6,leading=13.7,spaceAfter=8,textColor=ink),
'h':ParagraphStyle('h',fontName='Bold',fontSize=11.6,leading=16,spaceBefore=6,spaceAfter=6,textColor=teal,keepWithNext=True),
'title':ParagraphStyle('title',fontName='Bold',fontSize=20,leading=25,spaceAfter=16,textColor=ink,keepWithNext=True),
'caption':ParagraphStyle('caption',fontName='Body',fontSize=8,leading=10.8,textColor=muted,spaceAfter=12),
'cell':ParagraphStyle('cell',fontName='Body',fontSize=8,leading=10.8,textColor=ink),
'head':ParagraphStyle('head',fontName='Bold',fontSize=8,leading=10.8,textColor=colors.white),
'eq':ParagraphStyle('eq',fontName='Body',fontSize=9.5,leading=14,spaceBefore=4,spaceAfter=10,leftIndent=10,borderColor=teal,borderWidth=.7,borderPadding=7,backColor=colors.HexColor('#eff7f6'))}
def norm(s):return s.replace('–','-').replace('—','-').replace('‑','-')
def rich(s):
 s=html.escape(norm(s))
 s=re.sub(r'(https://[^\s]+)',r'<link href="\1" color="#137d78">\1</link>',s)
 return s.replace('\n','<br/>')
def footer(c,doc):
 c.saveState();w,h=doc.pagesize
 c.setStrokeColor(colors.HexColor('#d7e2e4'));c.line(44,h-34,w-44,h-34)
 c.setFillColor(teal);c.setFont('Bold',8);c.drawString(44,h-26,'PORTFOLIO REWARD DESIGN  /  RESEARCH REVISION')
 c.setFont('Body',7.5);c.setFillColor(muted);c.drawString(44,26,'22 September 2026  •  Assumptions and evidence status stated explicitly');c.drawRightString(w-44,26,str(doc.page));c.restoreState()
def tex_escape(s):
 d={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
 return ''.join(d.get(c,c) for c in norm(s))
def totex(name,pages):
 lines=[r'% Generated from documents.json. Compile with XeLaTeX; PDF delivered via ReportLab.',r'\documentclass[10pt,a4paper]{article}',r'\usepackage[margin=18mm]{geometry}',r'\usepackage{fontspec,graphicx,longtable,array,xcolor,hyperref}',r'\setmainfont{DejaVu Sans}',r'\setlength{\parindent}{0pt}',r'\setlength{\parskip}{6pt}',r'\hypersetup{colorlinks=true,urlcolor=teal}',r'\begin{document}']
 for i,pg in enumerate(pages):
  if i:lines.append(r'\clearpage')
  lines.append(r'\section*{'+tex_escape(pg['title'])+'}')
  for b in pg['blocks']:
   if b['type']=='p':lines.extend([tex_escape(b['text']),''])
   elif b['type']=='h':lines.append(r'\subsection*{'+tex_escape(b['text'])+'}')
   elif b['type']=='eq':lines.extend([r'\begin{quote}\small',tex_escape(b['text']).replace('\n',r'\\'+'\n'),r'\end{quote}'])
   elif b['type']=='fig':lines.extend([r'\begin{center}\includegraphics[width=\linewidth]{'+b['path']+'}',r'\end{center}',r'{\small '+tex_escape(b['caption'])+'}'])
   elif b['type']=='table':
    n=len(b['head']);ws=b.get('widths') or [1/n]*n;spec=''.join('p{'+f'{w*.91:.3f}'+r'\linewidth}' for w in ws)
    lines +=[r'\begin{longtable}{'+spec+'}',r'\hline',' & '.join(r'\textbf{'+tex_escape(t)+'}' for t in b['head'])+r'\\\hline']
    lines +=[' & '.join(tex_escape(v) for v in row)+r'\\\hline' for row in b['rows']];lines.append(r'\end{longtable}')
 lines.append(r'\end{document}');(ROOT/(name+'.tex')).write_text('\n'.join(lines))
for name,pages in json.loads((ROOT/'documents.json').read_text()).items():
 doc=SimpleDocTemplate(str(ROOT/(name+'.pdf')),pagesize=(595.276,841.89),rightMargin=44,leftMargin=44,topMargin=49,bottomMargin=44,title=pages[0]['title'],author='Research project',pageCompression=1)
 story=[];width=507.276
 for i,pg in enumerate(pages):
  if i:story.append(PageBreak())
  story.append(Paragraph(rich(pg['title']),styles['title']))
  for b in pg['blocks']:
   t=b['type']
   if t in ['p','h','eq']:story.append(Paragraph(rich(b['text']),styles[t]))
   elif t=='fig':
    path=ROOT/b['path'];im=PILImage.open(path);height=width*im.height/im.width
    image=Image(str(path),width=width,height=height)
    story.append(KeepTogether([image,Spacer(1,5),Paragraph(rich(b['caption']),styles['caption'])]))
   elif t=='table':
    n=len(b['head']);ws=b.get('widths') or [1/n]*n
    cells=[[Paragraph(rich(v),styles['head']) for v in b['head']]]+[[Paragraph(rich(str(v)),styles['cell']) for v in row] for row in b['rows']]
    tab=Table(cells,colWidths=[width*w for w in ws],repeatRows=1,hAlign='LEFT');tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),teal),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f1f5f6'),colors.white]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),.6,teal)]));story.extend([tab,Spacer(1,10)])
 doc.build(story,onFirstPage=footer,onLaterPages=footer);totex(name,pages);print(name,'rendered')
