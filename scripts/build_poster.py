"""Build a dense, evidence-led vector A1 poster; no training or new analysis."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentation"
W, H = 594 * mm, 841 * mm
INK, MUTED, TEAL, ORANGE, PURPLE = "#17343D", "#536871", "#007F80", "#BE592F", "#78568F"
PALE, RULE = "#EEF5F4", "#CDDAD9"
MODELS = ["LR0", "LR1", "LR2", "LGB-H0", "LGB-H1", "LGB-H2", "LGB-H3"]
PERIODS = ["dev_F1", "dev_F2", "validation", "test"]


def main():
    sources = [ROOT / p for p in ["evidence/executed_summary.json",
        "evidence/supplement_20261121_01/sensitivity.json",
        "evidence/supplement_20261121_01/quality_audit.json",
        "evidence/supplement_20261121_01/verification.json", "configs/study_plan.json"]]
    summary, sensitivity, quality, supplement_qa, plan = [json.loads(p.read_text()) for p in sources]
    metrics = summary["metrics"]
    omission = next(r for r in sensitivity["periods"]["test"]["omissions"] if r["omitted_days"] == [171, 172])
    fonts = Path(os.environ.get("FRAUDML_POSTER_FONT_DIR", "/System/Library/Fonts/Supplemental"))
    pdfmetrics.registerFont(TTFont("Poster", str(fonts / "Arial.ttf")))
    pdfmetrics.registerFont(TTFont("PosterBold", str(fonts / "Arial Bold.ttf")))
    pdfmetrics.registerFontFamily("Poster", normal="Poster", bold="PosterBold")
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "Team-FraudML-Poster.pdf"
    c = canvas.Canvas(str(target), pagesize=(W, H), pageCompression=1)
    c.setTitle("FraudML | Payment fraud ranking: temporal stability, repair and history")
    c.setAuthor("Team FraudML | DSC5001")
    boxes, sections, plotted = [], [], {}
    margin, gap = 44, 28
    cw = (W - 2 * margin - 2 * gap) / 3
    xs = [margin + i * (cw + gap) for i in range(3)]

    def rect(x, y, w, h, color):
        c.setFillColor(HexColor(color)); c.rect(x, H-y-h, w, h, stroke=0, fill=1)

    def line(x1, y1, x2, y2, color=RULE, width=1):
        c.setStrokeColor(HexColor(color)); c.setLineWidth(width)
        c.line(x1, H-y1, x2, H-y2)

    def para(value, x, y, w, size=20, color=INK, bold=False, limit=None, leading=None):
        style = ParagraphStyle("poster", fontName="PosterBold" if bold else "Poster",
            fontSize=size, leading=leading or size*1.27, textColor=HexColor(color))
        p = Paragraph(str(value), style)
        _, h = p.wrap(w, H)
        if limit is not None and h > limit + .1:
            raise ValueError(f"Text exceeds {limit}: {value}")
        if x < 20 or y < 0 or x+w > W-20 or y+h > H-15:
            raise ValueError(f"Text outside page at {y:.1f}: {value}")
        p.drawOn(c, x, H-y-h)
        boxes.append({"x":x, "top":y, "width":w, "bottom":y+h, "text":str(value), "font_size":size})
        return y+h

    def section(col, y, number, title):
        x = xs[col]
        line(x, y, x+cw, y, TEAL, 2)
        para(number, x, y+12, 34, 19, TEAL, True)
        bottom = para(title, x+39, y+10, cw-39, 24, INK, True)
        sections.append({"number":number, "title":title, "column":col, "top":y})
        return bottom+9

    def body(col, y, value, size=20, color=INK):
        return para(value, xs[col], y, cw, size-1, color, leading=(size-1)*1.17)+7

    def table(col, y, headers, rows, widths, size=18, row_height=34, highlights=None):
        x = xs[col]
        rect(x, y, cw, 34, INK)
        for i, label in enumerate(headers):
            para(label, x+sum(widths[:i])+8, y+7, widths[i]-14, size-1, "#FFFFFF", True, limit=24)
        y += 34
        for ridx, row in enumerate(rows):
            if ridx % 2 == 0: rect(x,y,cw,row_height,PALE)
            if highlights and ridx in highlights: rect(x,y,cw,row_height,"#FBECE4")
            height = row_height
            # Measure all cells first to support wrapped ablation descriptions.
            for i, value in enumerate(row):
                p=Paragraph(str(value), ParagraphStyle("cell",fontName="Poster",fontSize=size,leading=size*1.2))
                _,h=p.wrap(widths[i]-14,H); height=max(height,h+14)
            if height>row_height and (ridx%2==0 or highlights and ridx in highlights):
                rect(x,y,cw,height,"#FBECE4" if highlights and ridx in highlights else PALE)
            for i,value in enumerate(row):
                para(value,x+sum(widths[:i])+8,y+7,widths[i]-14,size,INK,i==0,limit=height-10,leading=size*1.2)
            y+=height
        line(x,y,x+cw,y)
        return y+9

    rect(0,0,W,H,"#FFFFFF")
    rect(0,0,W,238,INK)
    para("DSC5001  /  STATISTICAL MACHINE LEARNING I",44,25,W-88,19,"#A4DECF",True)
    para("Payment fraud ranking",41,60,W-82,66,"#FFFFFF",True)
    para("Temporal stability, numerical repair and the value of transaction history",44,145,W-88,29,"#FFFFFF",True)
    para("Team FraudML  |  Retrospective study  |  21 November 2026",44,194,W-88,20,"#D7E5E7")
    authors=[("Wang Yashi","60926887"),("Li Yihui","60451042"),("Wang Shihong","60995230"),("Wu Jiaqi","61034244"),("Chen Situ","60791221")]
    for i,(name,uid) in enumerate(authors):
        para(f"<b>{name}</b>  |  {uid}",44+i*(W-88)/5,253,(W-88)/5-8,17)
    stats=[("590,540","transactions / 3.50% fraud"),("21 + 3","core fits / legacy reruns"),("7 x 4","candidates / evaluation periods"),("394","columns audited"),("332","date-omission scenarios")]
    for i,(value,label) in enumerate(stats):
        x=44+i*(W-88)/5
        para(value,x,293,(W-88)/5-10,36,TEAL,True)
        para(label,x,340,(W-88)/5-12,17,MUTED)
    rect(44,383,W-88,69,PALE)
    para("MAIN FINDING",59,396,190,18,TEAL,True)
    para("Clipping repairs severe LR degradation; more history does not deliver stable gains.<br/>At the same review capacity, the dates receiving extreme scores can change the conclusion.",252,393,W-314,23,INK,True,limit=59)
    for x in [xs[1]-gap/2,xs[2]-gap/2]: line(x,478,x,2162,RULE)

    # Column 1: questions, controlled design and implementation.
    y=section(0,470,"01","Question and data")
    y=body(0,y,"Under the <b>same limited review capacity</b>, do numerical repair and richer history improve fraud ranking across time? We separate preprocessing effects, feature increments and date composition.")
    y=body(0,y,"IEEE-CIS / Vesta [1]: <b>590,540 labeled transactions</b>, 20,663 fraud labels, 394 raw columns and 182 relative day buckets. Identity is audit-only. ID, target and split time are excluded from model inputs.")

    y=section(0,y+5,"02","Time design and fit budget")
    split_rows=[]
    for period in PERIODS:
        snap=period if period.startswith("dev") else "original_train"
        split=summary["splits"][snap]
        ev=split["evaluations"][period]
        split_rows.append([{"dev_F1":"Dev F1","dev_F2":"Dev F2","validation":"Valid.","test":"Test"}[period],f"{split['train']['rows']:,}",f"{ev['rows']:,}",str(ev['observed_days'])])
    y=table(0,y,["Period","Train n","Eval. n","Days"],split_rows,[100,143,143,cw-386],size=18)
    y=body(0,y,"Evaluate days <b>[61,89)</b> and <b>[89,117)</b>. Overlapping expanding training windows are not independent replications. Validation/test share one 413,378-row fitted snapshot.",19)
    y=body(0,y,"<b>7 candidates x 3 snapshots = 21 core fits</b>, plus 3 legacy reruns. All 9 LR fits converged. Parameters and candidates are fixed; no large search or test-based winner selection.",19)

    y=section(0,y+5,"03","Seven controlled ablations")
    model_rows=[
        ["LR0","Original numeric pipeline; L2, C=1."],
        ["LR1","LR0 + train-fitted 0.1-99.9% clipping."],
        ["LR2","LR1 + stronger L2 penalty, C=0.1."],
        ["LGB-H0","Base LightGBM [2]; no added history."],
        ["LGB-H1","+4 cumulative fields; shared missing-entity group."],
        ["LGB-H2","Same history; missing entities keep history missing."],
        ["LGB-H3","+4 fields: card1/addr1 24h counts and elapsed time."]]
    y=table(0,y,["Candidate","Change under test"],model_rows,[112,cw-112],size=19,row_height=36)
    y=body(0,y,"LR: balanced classes, L-BFGS, max_iter=1000; rank by decision_function. LightGBM: 300 trees, learning rate 0.05, 31 leaves, min_child_samples=20; balanced classes, seed 42.",18,MUTED)

    y=section(0,y+5,"04","Leakage and boundary controls")
    for label,desc in [
        ("TRAIN ONLY","Fit clipping, imputation, encoding and scaling on training data. Clip amount, dist1/2 and C/D/V numeric fields."),
        ("STRICT PAST","History uses card1 and addr1 inputs at times &lt;t, never labels; 24h windows are [t-86400,t)."),
        ("TIED TIMES","Compute all rows at one timestamp before updating state. Normalize codes such as 123 and 123.0 consistently.")]:
        y=body(0,y,f"<font color='{TEAL}'><b>{label}</b></font>  {desc}",19)
    y=body(0,y,"Training labels are assumed available at each cutoff; actual arrival times are unknown. Previously viewed test results remain retrospective. Anonymous entity codes are not verified identities.",18,MUTED)

    y=section(0,y+5,"05","Capacity and uncertainty")
    y=body(0,y,"<b>K = ceil(r x N)</b>; hits = fraud labels in the top K.<br/><b>Precision = hits/K</b>; recall = hits/total fraud.<br/>Primary r=0.5%; also 0.1%, 1%, 2% and legacy Top100/500. AP is average precision [3].",19)
    y=body(0,y,"Main contrasts: LR2-LR0 and H3-H0. Each period uses <b>1,000 paired 3-day moving-block draws</b> for percentile 95% intervals. Daily 8/16/32-review scenarios match the global comparator's actual total capacity.",19)
    ends=[y]

    # Column 2: full fixed-candidate matrix and vector charts.
    y=section(1,470,"06","All candidates, all periods")
    y=body(1,y,"Fraud hits at <b>0.5% period capacity</b>. Column budgets differ with evaluation size; compare models within each period. Test contains 3,083 fraud labels.",19)
    x=xs[1]
    widths=[112]+[(cw-112)/4]*4
    rows=[[m]+[str(metrics[p][m]["period_rates"]["0.005"]["hits"]) for p in PERIODS] for m in MODELS]
    y=table(1,y,["Model","Dev F1","Dev F2","Valid.","Test"],rows,widths,size=19,row_height=37,highlights={0})
    y=table(1,y,["Reviews","429","459","443","443"],[],widths,size=18)
    y=body(1,y,"<b>Test precision / recall / AP</b><br/>LR0: 0.00% / 0.00% / 0.178<br/>LR2: 86.46% / 12.42% / 0.370<br/>H0: 93.00% / 13.36% / 0.500<br/>H3: 91.42% / 13.14% / 0.497",19)
    y=body(1,y,"Amount ranking: <b>8/443</b>; random conditional expectation: <b>15.42/443</b>. LR0 probability ranking gives 6 hits; margin ranking gives 0. Removing probability saturation alone does not repair the ranking.",18,MUTED)

    y=section(1,y+5,"07","Budget sensitivity on test")
    plot_top=y+8; chart_x=x+49; chart_w=cw-65; chart_h=160
    rates=plan["budget"]["period_rates"]
    plot_models=[("LR0",ORANGE),("LR2",TEAL),("LGB-H0",INK),("LGB-H3",PURPLE)]
    for tick in [0,25,50,75,100]:
        yy=plot_top+chart_h*(1-tick/100)
        line(chart_x,yy,chart_x+chart_w,yy)
        para(str(tick),x,yy-9,39,16,MUTED)
    for i,rate in enumerate(rates):
        px=chart_x+(rate-rates[0])/(rates[-1]-rates[0])*chart_w
        para(f"{rate*100:g}%",px-18,plot_top+chart_h+15,55,16,MUTED)
    plotted["test_budget_precision"]={}
    for model,color in plot_models:
        vals=[metrics["test"][model]["period_rates"][str(r)]["precision"] for r in rates]
        plotted["test_budget_precision"][model]=vals
        points=[(chart_x+(rates[i]-rates[0])/(rates[-1]-rates[0])*chart_w,plot_top+chart_h*(1-v)) for i,v in enumerate(vals)]
        for a,b in zip(points,points[1:]): line(*a,*b,color,3)
        for px,py in points:
            c.setFillColor(HexColor(color)); c.circle(px,H-py,4,stroke=0,fill=1)
    y=plot_top+chart_h+41
    for i,(model,color) in enumerate(plot_models):
        xx=x+i*cw/4; line(xx,y+9,xx+20,y+9,color,3)
        para(model,xx+25,y,cw/4-25,16,color,True)
    y=body(1,y+31,"Y: precision (%). X: review rate. H0/H3 are close; LR0 fails most severely at tight capacity. Complete seven-model results remain in the evidence.",18,MUTED)

    y=section(1,y+5,"08","Paired effects across time")
    y=body(1,y,"LR2-LR0: hit difference [3-day-block 95% interval]",19)
    lr_rows=[]
    for p,label in zip(PERIODS,["Dev F1","Dev F2","Valid.","Test"]):
        d=summary["paired_analysis"]["periods"][p]["contrasts"]["LR2 minus LR0"]["period_rate"]["hits"]
        lr_rows.append([label,f"{d['point_difference']:+.0f}",f"[{d['lower']:.2f}, {d['upper']:.2f}]"])
    y=table(1,y,["Period","Difference","95% interval"],lr_rows,[110,133,cw-243],size=18,row_height=32)
    y=body(1,y,"<b>H3-H0: no stable positive history increment</b>",20)
    plot_top=y+7; chart_x=x+100; chart_w=231
    def px(v): return chart_x+(v+30)/42*chart_w
    for tick in [-30,-20,-10,0,10]:
        line(px(tick),plot_top,px(tick),plot_top+133,INK if tick==0 else RULE,1)
        para(str(tick),px(tick)-10,plot_top+144,40,15,MUTED)
    for i,(p,label) in enumerate(zip(PERIODS,["Dev F1","Dev F2","Valid.","Test"])):
        d=summary["paired_analysis"]["periods"][p]["contrasts"]["LGB-H3 minus LGB-H0"]["period_rate"]["hits"]
        yy=plot_top+15+i*36
        para(label,x,yy-11,94,18)
        line(px(d['lower']),yy,px(d['upper']),yy,PURPLE,3)
        c.setFillColor(HexColor(PURPLE)); c.circle(px(d['point_difference']),H-yy,5,stroke=0,fill=1)
        para(f"{d['point_difference']:+.0f} [{d['lower']:.0f}, {d['upper']:.0f}]",x+350,yy-11,cw-350,17,PURPLE)
    y=body(1,plot_top+174,"H3-H0 changes sign: <b>+2, -11, +4, -7</b>. Every main 3-day interval crosses zero; this does not establish equivalence. AP and top-K hits describe different ranking regions.",19)

    y=section(1,y+5,"09","What the ablations establish")
    y=body(1,y,"<b>Clipping drives the repair.</b> LR1-LR0: +27/+13/+13/+376 hits. Stronger L2 (LR2-LR1): -3/-1/-4/+7 across F1/F2/validation/test; no consistent extra gain.",19)
    ends.append(y)

    # Column 3: diagnostic evidence and independently checked work.
    y=section(2,470,"10","Diagnosing the LR failure")
    y=body(2,y,"<b>394 columns x 3 partitions</b>: missingness, unique values, constants, quantiles, non-finite/unseen values and train-range exceedances. Raw maxima:",19)
    audit_rows=[]
    for feature in ["V323","V324","V177"]:
        f=quality["features"][feature]
        audit_rows.append([feature,f"{f['train']['quantiles']['1']:g}",f"{f['test']['quantiles']['1']:g}"])
    y=table(2,y,["Field","Train max","Test max"],audit_rows,[146,171,cw-317],size=19,row_height=30)
    y=body(2,y,"V323 median: <b>733 on days 171-172</b>, 0 on other test days (nonmissing). Score contributions and date concentration support an extreme-input diagnosis, not a unique causal explanation.",19)
    y=body(2,y,"<b>LR0 top 443 reviews: 442 from two dates; 0 hits.</b>",19)
    x=xs[2]; barw=cw-8
    rect(x,y,barw*169/443,30,ORANGE)
    rect(x+barw*169/443,y,barw*273/443,30,"#E2A16E")
    rect(x+barw*442/443,y,barw/443,30,INK)
    para("Day 171: 169",x+9,y+5,barw*169/443-12,17,"#FFFFFF",True)
    para("Day 172: 273",x+barw*169/443+10,y+5,barw*273/443-15,17,INK,True)
    y=body(2,y+39,"One review remains on day 181. Duplicate inputs are also audited; identical anonymous fields alone do not prove leakage. No automatic deduplication.",18,MUTED)

    y=section(2,y+5,"11","Daily capacity: matched totals")
    y=body(2,y,"Test: <b>16 reviews/day = 496 slots</b>. Compare with the same 496 global reviews; differences are daily minus global hits.",19)
    daily_rows=[]
    for m in ["LR0","LR2","LGB-H0","LGB-H3"]:
        d=metrics["test"][m]["daily"]["fixed_16"]
        daily_rows.append([m,str(d["hits"]),str(d["matched_global"]["hits"]),f"{d['hits']-d['matched_global']['hits']:+d}"])
    y=table(2,y,["Model","Daily","Global","Diff."],daily_rows,[136,116,116,cw-368],size=19,row_height=30)
    y=body(2,y,"LR0: +301 hits, 3-day 95% interval <b>[-38, 341.02]</b>. LR2/H0/H3 lose 53/26/24 hits. This end-of-day batch scenario gives no universal improvement.",19)
    rect(x,y,cw,111,"#FCF0E7")
    para("POST-HOC DATE CHECK",x+12,y+10,cw-24,17,ORANGE,True)
    para("Omit days 171-172: LR0 daily advantage flips<br/><b>+301 to -8 hits</b>; LR2 still leads LR0 by<br/><b>87 hits (362 vs 275; 414 period slots).</b>",x+12,y+37,cw-24,19,INK,limit=75)
    y=body(2,y+125,"Fixed models, scores and history; evaluation rows are filtered and budgets rematched. All <b>332 continuous 1/2/7-day omission windows</b> are reported; the full test remains primary.",18,MUTED)

    y=section(2,y+5,"12","How sensitive is inference?")
    y=body(2,y,"Test H3-H0 = -7 hits. Post-hoc block-length checks retain the original 3-day result:",19)
    sens_rows=[]
    for block in [1,3,7]:
        d=sensitivity["periods"]["test"]["blocks"][str(block)]["contrasts"]["LGB-H3 minus LGB-H0"]["period_rate"]["hits"]
        sens_rows.append([f"{block} day" if block==1 else f"{block} days",f"[{d['lower']:.0f}, {d['upper']:.0f}]","Yes" if d['lower']<=0<=d['upper'] else "No"])
    y=table(2,y,["Block length","95% interval","Crosses 0?"],sens_rows,[165,169,cw-334],size=18,row_height=32)
    y=body(2,y,"7-day blocks give a negative interval; 1/3-day blocks cross zero. None supports stable positive history gains. LR2-LR0 is positive on test at all three block lengths, with weaker evidence elsewhere.",19)
    y=body(2,y,"Fixed-model intervals omit training uncertainty and multiplicity correction. <b>31 test day buckets</b> limit small-effect and long-term claims; resampling adds no new months.",18,MUTED)

    y=section(2,y+5,"13","Verified engineering and evidence")
    checks=summary["verification"]["checks"]
    y=body(2,y,f"<b>10 correctness tests passed.</b> Independent checks: <b>{checks['period_model_pairs']} period/config groups</b>, <b>{checks['daily_groups']:,} daily groups</b> and <b>{checks['queue_rows']:,} queue records</b>.",19)
    sc=supplement_qa["checks"]
    y=body(2,y,f"Supplement: <b>{sc['raw_missingness_field_partition_groups']:,} missingness values</b>, 28 baseline metric groups and <b>{sc['independent_omission_model_checks']} independent omission/model checks</b>.",19)
    y=body(2,y,"Clean environment: install, pip check, 10 tests and synthetic pipeline passed; <b>no real-data refit</b>.",18,MUTED)
    ends.append(y)

    if max(ends)>2162:
        raise ValueError(f"Columns overflow footer: {ends}")
    rect(44,2181,W-88,103,INK)
    footer_items=[("REPAIR","Train-only clipping restores much of the LR ranking."),
                  ("RETAIN NEGATIVE RESULTS","History gains do not persist across evaluation periods."),
                  ("TEST ON NEW TIME","Next: a genuinely unseen labeled period under a fixed protocol.")]
    for i,(title,desc) in enumerate(footer_items):
        para(title,xs[i]+13,2195,cw-26,18,"#A4DECF",True)
        para(desc,xs[i]+13,2225,cw-26,20,"#FFFFFF",limit=53)
    para("Scope: retrospective ranking, uncalibrated scores and assumed review budgets. No verified deployment, causal mechanism, label-delay model, financial return or long-run staffing optimum is claimed.",44,2296,W-88,15,MUTED,limit=43)
    para("Sources: verified study_20261121_01; executed_summary.json; supplement_20261121_01; frozen plan2. [1] Kaggle / Vesta, IEEE-CIS Fraud Detection (2019). [2] Ke et al., LightGBM, NeurIPS (2017). [3] Saito &amp; Rehmsmeier, precision-recall evaluation, PLOS ONE (2015).",44,2338,W-88,13,MUTED,limit=40)
    # Every text rectangle must be disjoint; touching edges are allowed.
    for i,a in enumerate(boxes):
        for b in boxes[i+1:]:
            if min(a['x']+a['width'],b['x']+b['width'])-max(a['x'],b['x'])>.5 and min(a['bottom'],b['bottom'])-max(a['top'],b['top'])>.5:
                raise ValueError(f"Text overlap: {a['text'][:60]} / {b['text'][:60]}")
    c.showPage(); c.save()
    manifest={"created_at_utc":datetime.now(timezone.utc).isoformat(),
        "page_mm":[594,841],"orientation":"portrait","language":"en","layout":"three_columns_13_sections",
        "source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "output_sha256":hashlib.sha256(target.read_bytes()).hexdigest(),
        "column_bottoms_pt":ends,"sections":sections,"plotted_data":plotted,"text_boxes":boxes}
    (OUT/"poster_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    print(target)
    print(f"Column bottoms: {ends}; text boxes: {len(boxes)}")


if __name__ == "__main__":
    main()
