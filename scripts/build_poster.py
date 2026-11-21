"""Build a vector A1 coursework poster from verified aggregate evidence."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
from xml.sax.saxutils import escape

from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import mm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables/presentation"
W, H = 594 * mm, 841 * mm
INK, MUTED, TEAL, GOLD, PURPLE = "#15323C", "#4E626B", "#007C83", "#B57326", "#76528B"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary_path = ROOT / "evidence/executed_summary.json"
    sensitivity_path = ROOT / "evidence/supplement_20261121_01/sensitivity.json"
    summary = json.loads(summary_path.read_text())
    sensitivity = json.loads(sensitivity_path.read_text())
    metrics = summary["metrics"]
    pdfmetrics.registerFont(TTFont("Poster", "/System/Library/Fonts/Supplemental/Arial.ttf"))
    pdfmetrics.registerFont(TTFont("PosterBold", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"))
    target = OUT / "Team-FraudML-Poster.pdf"
    c = canvas.Canvas(str(target), pagesize=(W, H), pageCompression=1)
    c.setTitle("FraudML | Payment fraud ranking: repair, history and time stability")
    c.setAuthor("Team FraudML | DSC5001")
    boxes = []

    def rect(x, top, width, height, fill, radius=0):
        c.setFillColor(HexColor(fill))
        if radius:
            c.roundRect(x, H-top-height, width, height, radius, stroke=0, fill=1)
        else:
            c.rect(x, H-top-height, width, height, stroke=0, fill=1)

    def text(content, x, top, width, size=23, color=INK, bold=False, max_height=None, leading=None):
        style = ParagraphStyle("p", fontName="PosterBold" if bold else "Poster", fontSize=size,
                               leading=leading or size*1.3, textColor=HexColor(color), alignment=TA_LEFT)
        paragraph = Paragraph(content, style)
        _, height = paragraph.wrap(width, 4000)
        if max_height is not None and height > max_height + .1:
            raise ValueError(f"Poster text overflow {height:.1f}>{max_height}: {content[:80]}")
        paragraph.drawOn(c, x, H-top-height)
        boxes.append({"top":top,"bottom":top+height,"x":x,"width":width,"text":content})
        if top < 0 or top+height > H-18 or x+width > W-18:
            raise ValueError("Poster content outside page")
        return top+height

    def line(x1, top1, x2, top2, color="#C9D5D8", width=1):
        c.setStrokeColor(HexColor(color)); c.setLineWidth(width)
        c.line(x1,H-top1,x2,H-top2)

    def panel(x, top, width, height, number, title):
        rect(x,top,width,height,"#F1F6F6",12)
        text(number,x+23,top+21,55,22,TEAL,True)
        text(title,x+78,top+19,width-100,28,INK,True,max_height=42)

    rect(0,0,W,H,"#FFFFFF")
    rect(0,0,W,321,INK)
    text("DSC5001  /  STATISTICAL MACHINE LEARNING I",48,35,W-96,21,"#9EE0D8",True)
    text("Payment fraud ranking",48,83,W-96,65,"#FFFFFF",True)
    text("What survives when the data changes?",48,165,W-96,45,"#FFFFFF",True)
    text("Team FraudML  |  21 November 2026  |  Retrospective study",50,230,W-100,21,"#D2E4E8")
    authors=[("Wang Yashi","60926887"),("Li Yihui","60451042"),("Wang Shihong","60995230"),("Wu Jiaqi","61034244"),("Chen Situ","60791221")]
    for i,(name,uid) in enumerate(authors):
        x=50+i*(W-100)/5
        text(f"{name} <font color='#9EE0D8'>| {uid}</font>",x,279,(W-100)/5-8,17.5,"#FFFFFF")

    rect(48,345,W-96,116,"#E1F1EE",10)
    text("THE QUESTION",70,362,245,19,TEAL,True)
    text("With the same review capacity, which changes improve fraud ranking - and how stable is that improvement?",315,359,W-390,30,INK,True,max_height=89)

    stats=[("590,540","labeled transactions"),("394 / 182","raw columns / relative days"),("21 + 3","core fits + legacy reruns"),("7 / 4","fixed candidates / evaluation periods")]
    for i,(value,label) in enumerate(stats):
        x=53+i*(W-106)/4
        text(value,x,489,(W-106)/4-16,40,TEAL,True)
        text(label,x,545,(W-106)/4-16,20,MUTED,max_height=58)
    text("IEEE-CIS / Vesta transaction table [1]  |  20,663 fraud labels (3.50%)  |  Identity features excluded",53,603,W-106,21,MUTED)

    panel(48,652,W-96,323,"01","Controlled comparisons over time")
    left=72; col=(W-168)/2
    text("TIME DESIGN",left,722,col,19,TEAL,True)
    text("Two 28-day development evaluations: days [61,89) and [89,117). Later validation/test share one fitted model snapshot.",left,755,col,23,max_height=100)
    text("Train-only transforms; equal timestamps stay together. History uses strictly earlier inputs, never labels. Already viewed test: retrospective, not a new blind test.",left,862,col,21,MUTED,max_height=92)
    x=left+col+26
    text("SEVEN FIXED CANDIDATES",x,722,col,19,TEAL,True)
    text("<b>LR0</b> original numeric pipeline; <b>LR1</b> train-fitted 0.1-99.9% clipping; <b>LR2</b> same clipping + stronger L2 (C=0.1).",x,755,col,22,max_height=100)
    text("<b>H0</b> base LightGBM [2]; <b>H1</b> cumulative history; <b>H2</b> correct missing-entity grouping; <b>H3</b> add 24h counts and elapsed time.",x,865,col,22,max_height=91)

    gap=25; cw=(W-96-gap)/2
    panel(48,1000,cw,536,"02","Repairing the LR failure")
    panel(48+cw+gap,1000,cw,536,"03","Does history help consistently?")
    text("Test: 88,581 transactions, 3,083 fraud labels.",72,1068,cw-48,21,MUTED)
    text("Same 0.5% capacity = 443 reviews",72,1101,cw-48,24,INK,True)
    barx=238; chartw=cw-410
    text("Hits",48+cw-135,1136,70,17,MUTED,True)
    text("AP",48+cw-65,1136,50,17,MUTED,True)
    labels=[("LR0",INK),("LR1",TEAL),("LR2",TEAL),("LGB-H0",GOLD),("LGB-H1",PURPLE),("LGB-H2",PURPLE),("LGB-H3",PURPLE)]
    for i,(model,color) in enumerate(labels):
        y=1171+i*33
        row=metrics["test"][model]["period_rates"]["0.005"]
        text(model,72,y-2,150,19,INK,model in {"LR2","LGB-H0"})
        rect(barx,y,chartw,17,"#DCE7E8")
        if row["hits"]:
            rect(barx,y,chartw*row["hits"]/443,17,color)
        text(str(row["hits"]),48+cw-135,y-4,65,19,INK,True)
        text(f"{row['average_precision']:.3f}",48+cw-66,y-4,57,17,MUTED)
    text("Amount: 8 hits. Random expectation: 15.42.",72,1410,cw-48,19,MUTED)
    text("Clipping repairs much of the failure. Stronger L2 alone is not a consistent extra gain across periods. Margin ranking alone did not fix LR0.",72,1450,cw-48,21,INK,max_height=75)

    hx=48+cw+gap
    text("H3 minus H0: change in fraud hits",hx+24,1068,cw-48,23,INK,True)
    text("0.5% period budget; paired 3-day-block 95% intervals",hx+24,1106,cw-48,18,MUTED)
    x0=hx+205; span=355; lo=-30; hi=12
    def px(v):return x0+(v-lo)/(hi-lo)*span
    for tick in [-30,-20,-10,0,10]:
        line(px(tick),1150,px(tick),1357,"#BCCED1" if tick else INK,1)
        text(str(tick),px(tick)-12,1366,40,16,MUTED)
    for i,(period,label) in enumerate([("dev_F1","Dev F1"),("dev_F2","Dev F2"),("validation","Validation"),("test","Test")]):
        row=summary["paired_analysis"]["periods"][period]["contrasts"]["LGB-H3 minus LGB-H0"]["period_rate"]["hits"]
        y=1170+i*55
        text(label,hx+24,y-14,160,21)
        line(px(row["lower"]),y,px(row["upper"]),y,PURPLE,4)
        c.setFillColor(HexColor(PURPLE)); c.circle(px(row["point_difference"]),H-y,6,stroke=0,fill=1)
        text(f"{row['point_difference']:+.0f} [{row['lower']:.0f}, {row['upper']:.0f}]",hx+590,y-12,cw-610,17,MUTED)
    text("No stable positive gain: effects change sign.",hx+24,1410,cw-48,22,INK,True)
    text("All four 3-day intervals cross zero. A post-hoc 7-day block gives test [-19,-1]; 1/3-day blocks cross zero. Inference is sensitive, not proof of equivalence.",hx+24,1450,cw-48,21,INK,max_height=75)

    panel(48,1561,W-96,284,"04","Daily capacity changes the question")
    text("Daily 16 reviews vs global ranking - both use 496 slots",72,1629,850,24,INK,True)
    cols=[72,300,490,700]
    for label,x in zip(["Test model","Daily hits","Global hits","Difference"],cols):
        text(label,x,1675,180,18,MUTED,True)
    for i,model in enumerate(["LR0","LR2","LGB-H0","LGB-H3"]):
        row=metrics["test"][model]["daily"]["fixed_16"]
        vals=[model,str(row["hits"]),str(row["matched_global"]["hits"]),f"{row['hits']-row['matched_global']['hits']:+d}"]
        for value,x in zip(vals,cols):text(value,x,1707+i*29,180,21,INK,i==0)
    text("LR0 paired 3-day 95% interval: [-38, 341.02] hits; crosses zero.",72,1823,850,15.5,MUTED,max_height=21)
    line(930,1632,930,1818)
    text("THE DATE-COMPOSITION CHECK",965,1630,W-1040,18,TEAL,True)
    text("Omitting days 171-172 changes LR0's daily advantage from <b>+301 to -8 hits</b>.",965,1667,W-1040,26,max_height=106)
    text("Post hoc, fixed scores; budgets rematched. All 332 continuous-window omissions reported, not only this example.",965,1771,W-1040,19,MUTED,max_height=65)

    panel(48,1870,W-96,287,"05","What we learned - and what the data cannot establish")
    sections=[("REPAIR, THEN COMPARE","Raw-feature audit: V323 train max 49, test max 1,411. LR0's top 443 contain 442 cases from two days. Distribution shifts and score contributions support a diagnosis, not a unique cause."),
              ("MORE FEATURES CAN FAIL","History is not a guaranteed upgrade. Retain negative results and complete ablations; do not choose the test winner after inspection."),
              ("ROWS ARE NOT MONTHS","590k rows support this ranking study; only 31 test day buckets limit small-effect and long-term claims. No real review costs, label arrival times or verified deployment benefit.")]
    sw=(W-144)/3
    for i,(title,body) in enumerate(sections):
        xx=72+i*sw
        text(title,xx,1940,sw-24,20,TEAL,True)
        text(body,xx,1980,sw-28,22,max_height=154)

    text("REPRODUCIBILITY",52,2184,250,18,TEAL,True)
    text("10 correctness tests; independent score/queue checks; 394-column quality audit; 1/3/7-day block sensitivity. Clean installation + synthetic end-to-end passed; no full real-data refit in that environment.",300,2182,W-352,19,MUTED,max_height=61)
    line(48,2255,W-48,2255)
    text("Sources: verified study_20261121_01; executed_summary.json; supplement_20261121_01. Intervals condition on fixed models and observed dates; no training uncertainty or multiplicity correction. Scores are uncalibrated; daily review is end-of-day batch evaluation.",52,2270,W-104,16,MUTED,max_height=44)
    text("[1] Kaggle / Vesta, IEEE-CIS Fraud Detection (2019).  [2] Ke et al., LightGBM, NeurIPS (2017).  [3] Saito &amp; Rehmsmeier, precision-recall evaluation, PLOS ONE (2015). Metrics: AP plus capacity-limited hits, precision and recall.",52,2320,W-104,16,MUTED,max_height=45)
    c.showPage();c.save()
    source={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [summary_path,sensitivity_path]}
    manifest={"created_at_utc":datetime.now(timezone.utc).isoformat(),"page_mm":[594,841],"orientation":"portrait","language":"en","source_sha256":source,"code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"output_sha256":hashlib.sha256(target.read_bytes()).hexdigest(),"text_boxes":boxes}
    (OUT/"poster_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    print(target)


if __name__=="__main__":main()
