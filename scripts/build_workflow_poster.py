"""Create a separate A1 workflow poster from verified aggregate evidence."""
from datetime import datetime, timezone
import hashlib
import json
import math
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
STEM = "Team-FraudML-Poster-Workflow"
W, H = 594*mm, 841*mm
INK, TEAL, PURPLE, ORANGE = "#17343D", "#007F80", "#78568F", "#BA5936"
MUTED, PALE, RULE = "#526872", "#F0F5F5", "#CBD9DA"
MODELS = ["LR0", "LR1", "LR2", "LGB-H0", "LGB-H1", "LGB-H2", "LGB-H3"]
PERIODS = ["dev_F1", "dev_F2", "validation", "test"]


def main():
    paths = [ROOT/p for p in ["evidence/executed_summary.json",
        "evidence/supplement_20261121_01/sensitivity.json",
        "evidence/supplement_20261121_01/quality_audit.json",
        "evidence/supplement_20261121_01/verification.json", "configs/study_plan.json"]]
    summary, sensitivity, audit, extra_qa, plan = [json.loads(p.read_text()) for p in paths]
    metrics = summary["metrics"]
    omission = next(r for r in sensitivity["periods"]["test"]["omissions"] if r["omitted_days"] == [171,172])
    fonts = Path(os.environ.get("FRAUDML_POSTER_FONT_DIR", "/System/Library/Fonts/Supplemental"))
    for name, filename in [("Flow", "Arial.ttf"), ("FlowBold", "Arial Bold.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(fonts/filename)))
    pdfmetrics.registerFontFamily("Flow", normal="Flow", bold="FlowBold")
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT/f"{STEM}.pdf"
    c = canvas.Canvas(str(target), pagesize=(W,H), pageCompression=1, initialFontName="Flow")
    c.setTitle("FraudML | From payment transactions to reliable ranking evidence")
    c.setAuthor("Team FraudML | DSC5001")
    boxes, arrows, nodes, chart_data = [], [], [], {}
    m, gap = 44, 28
    full = W-2*m

    def rect(x,y,w,h,fill,radius=0):
        c.setFillColor(HexColor(fill))
        if radius: c.roundRect(x,H-y-h,w,h,radius,stroke=0,fill=1)
        else: c.rect(x,H-y-h,w,h,stroke=0,fill=1)

    def line(x1,y1,x2,y2,color=RULE,width=1):
        c.setStrokeColor(HexColor(color)); c.setLineWidth(width)
        c.line(x1,H-y1,x2,H-y2)

    def text(value,x,y,w,size=21,color=INK,bold=False,limit=None,leading=None):
        style=ParagraphStyle("flow",fontName="FlowBold" if bold else "Flow",fontSize=size,
            leading=leading or size*1.22,textColor=HexColor(color))
        p=Paragraph(str(value),style);_,height=p.wrap(w,H)
        if limit is not None and height>limit+.1:
            raise ValueError(f"Text exceeds region ({height:.1f}>{limit}): {value}")
        if x<20 or x+w>W-20 or y<0 or y+height>H-12:
            raise ValueError(f"Text outside page: {value}")
        p.drawOn(c,x,H-y-height)
        boxes.append({"x":x,"top":y,"width":w,"bottom":y+height,"font_size":size,"text":str(value)})
        return y+height

    def arrow(points,color=TEAL,width=2.6):
        for a,b in zip(points,points[1:]): line(*a,*b,color,width)
        a,b=points[-2:]; angle=math.atan2(b[1]-a[1],b[0]-a[0]); length=9; spread=4
        back=(b[0]-length*math.cos(angle),b[1]-length*math.sin(angle))
        p=c.beginPath();p.moveTo(b[0],H-b[1])
        p.lineTo(back[0]+spread*math.sin(angle),H-back[1]+spread*math.cos(angle))
        p.lineTo(back[0]-spread*math.sin(angle),H-back[1]-spread*math.cos(angle));p.close()
        c.setFillColor(HexColor(color));c.drawPath(p,fill=1,stroke=0)
        arrows.append({"points":points,"color":color})

    def node(name,x,y,w,h,title,body,color=TEAL,fill="#E1F0EC",title_size=22):
        rect(x,y,w,h,fill,9);rect(x,y,5,h,color)
        bottom=text(title,x+16,y+13,w-28,title_size,color,True,limit=35)
        text(body,x+16,bottom+9,w-28,18,INK,limit=y+h-bottom-18)
        nodes.append({"id":name,"x":x,"top":y,"width":w,"height":h})

    # A question-led reading path, with a full-width branching experiment diagram.
    rect(0,0,W,H,"#FFFFFF");rect(0,0,W,244,INK)
    text("DSC5001  /  STATISTICAL MACHINE LEARNING I",44,25,full,20,"#A3DED0",True)
    text("Payment fraud ranking",41,65,full+6,65,"#FFFFFF",True)
    text("From transactions to evidence: what survives a change in time?",44,147,full,33,"#FFFFFF",True)
    text("Team FraudML  |  Study: 21 November 2026  |  Workflow edition  |  Retrospective analysis",44,203,full,20,"#D3E7E7")
    authors=[("Wang Yashi","60926887"),("Li Yihui","60451042"),("Wang Shihong","60995230"),("Wu Jiaqi","61034244"),("Chen Situ","60791221")]
    for i,(name,uid) in enumerate(authors):text(f"<b>{name}</b>  |  {uid}",m+i*full/5,262,full/5-9,17)
    stats=[("590,540","labeled transactions"),("394 / 182","raw columns / relative days"),("21 + 3","core fits / legacy reruns"),("7 / 4","candidates / evaluation periods"),("332","date-omission scenarios")]
    for i,(value,label) in enumerate(stats):
        xx=m+i*full/5;text(value,xx,301,full/5-12,37,TEAL,True);text(label,xx,351,full/5-12,18,MUTED)
    rect(m,392,full,78,"#E1F0EC",8)
    text("THE QUESTION",m+15,408,200,19,TEAL,True)
    text("At the same review capacity, which changes improve fraud ranking -<br/>and does the improvement persist across evaluation periods?",m+225,402,full-245,25,INK,True,limit=62)

    text("01",m,478,43,21,TEAL,True)
    text("Follow the experiment: one controlled pipeline, two model families",m+55,474,full-55,29,INK,True)
    node("transactions",60,554,222,128,"Labeled data","20,663 fraud (3.50%)<br/>Transaction table [1]<br/>Identity: audit only")
    node("time",324,554,244,128,"Split by time","3 training snapshots<br/>F1 / F2 / valid. / test<br/>No random-row CV")
    node("preprocess",610,554,240,128,"Train-only fitting","Impute, encode, scale<br/>Clip numeric fields<br/>Strict-past histories")
    node("logistic",896,514,300,82,"LR0  >  LR1  >  LR2","Original / clip / stronger L2",TEAL,title_size=23)
    node("history",896,615,300,87,"H0  >  H1  >  H2  >  H3","Cumulative / missing fix / 24h",PURPLE,"#EEE9F2",21)
    node("evaluation",1242,554,382,128,"Same capacity, then compare","Top-K hits, precision, recall<br/>AP + paired date-block intervals<br/>Inspect failure and date sensitivity",INK,"#E6ECEE",21)
    arrow([(282,618),(324,618)])
    arrow([(568,618),(610,618)])
    arrow([(850,618),(872,618),(872,555),(896,555)])
    arrow([(850,618),(872,618),(872,658),(896,658)],PURPLE)
    arrow([(1196,555),(1218,555),(1218,618),(1242,618)])
    arrow([(1196,658),(1218,658),(1218,618),(1242,618)],PURPLE)
    cols=[m,m+(full+gap)/3,m+2*(full+gap)/3];colw=(full-2*gap)/3
    notes=[("TIME DESIGN","F1: train 223,738 / evaluate 85,610 (28 days). F2: 309,348 / 91,732 (28 days). Evaluation windows [61,89) and [89,117)."),
           ("SAME FITTED SNAPSHOT","Later validation/test use one 413,378-row training snapshot; each evaluates 88,581 rows (33/31 day buckets). Expanding training sets overlap."),
           ("BOUNDARIES AND ASSUMPTIONS","History uses only inputs at times &lt;t, never labels; tied timestamps update together. Labels are assumed available at cutoff. Previously viewed test is retrospective.")]
    for xx,(title,body) in zip(cols,notes):
        text(title,xx,724,colw,18,TEAL,True)
        text(body,xx,755,colw,20,limit=98)

    # Three evidence columns retain the workflow and expand the controlled work.
    colw=(full-2*gap)/3
    cols=[m,m+colw+gap,m+2*(colw+gap)]
    for xx in [cols[1]-gap/2,cols[2]-gap/2]:line(xx,860,xx,2094)

    def section(col,y,num,title):
        x=cols[col];line(x,y,x+colw,y,TEAL,1.8)
        text(num,x,y+12,34,18,TEAL,True)
        return text(title,x+39,y+9,colw-39,24,INK,True)+12

    def body(col,y,value,size=18,color=INK):
        return text(value,cols[col],y,colw,size,color,leading=size*1.16)+7

    def compact_table(col,y,headers,rows,widths,size=17,row_h=29):
        x=cols[col];rect(x,y,colw,30,INK)
        for i,label in enumerate(headers):text(label,x+sum(widths[:i])+7,y+6,widths[i]-12,size-1,"#FFFFFF",True,limit=22)
        y+=30
        for ridx,row in enumerate(rows):
            height=row_h
            for i,value in enumerate(row):
                p=Paragraph(str(value),ParagraphStyle('cell',fontName='FlowBold' if i==0 else 'Flow',fontSize=size,leading=size*1.16))
                _,hh=p.wrap(widths[i]-12,H);height=max(height,hh+10)
            if ridx%2==0:rect(x,y,colw,height,PALE)
            for i,value in enumerate(row):text(value,x+sum(widths[:i])+7,y+5,widths[i]-12,size,INK,i==0,limit=height-7,leading=size*1.16)
            y+=height
        line(x,y,x+colw,y)
        return y+10

    # Column 1: model construction, full test ranking, and repair contrasts.
    y=section(0,860,'02','Seven fixed ablations')
    model_rows=[['LR0','Original numeric pipeline; L2, C=1.'],
        ['LR1','Train-fitted 0.1-99.9% clipping; C=1.'],
        ['LR2','Same clipping; stronger L2, C=0.1.'],
        ['LGB-H0','Base LightGBM; no added history.'],
        ['LGB-H1','+4 cumulative fields; shared missing group.'],
        ['LGB-H2','Same 4 fields; missing entities keep history missing.'],
        ['LGB-H3','+4 card1/addr1 24h-count and elapsed-time fields.']]
    y=compact_table(0,y,['Candidate','Controlled change'],model_rows,[108,colw-108],17,30)
    y=body(0,y,'<b>7 models x 3 snapshots = 21 core fits</b>; all 9 LR fits converged. LR: balanced classes, L-BFGS, max_iter=1000; rank by margin. LightGBM [2]: 300 trees, learning rate 0.05, 31 leaves, min_child_samples=20, seed 42.',17,MUTED)
    y=body(0,y,'Exclude ID, target and split time from model inputs. Fit clipping, encoding, imputation and scaling on train only. Clip amount, dist1/2 and C/D/V fields; normalize entity codes. H3 uses [t-86400,t); tied-time groups update after all histories are computed.',17)

    y=section(0,y+4,'03','What repairs the LR failure?')
    y=body(0,y,'Test: <b>3,083 fraud labels / 88,581 rows</b>. The same 0.5% capacity gives <b>443 reviews</b>.',18)
    bx=cols[0]+112;bw=colw-264;hit_data=[]
    text('Hits',cols[0]+colw-131,y,55,16,MUTED,True);text('AP',cols[0]+colw-61,y,55,16,MUTED,True);y+=28
    for i,model in enumerate(MODELS):
        d=metrics['test'][model]['period_rates']['0.005'];yy=y+i*29
        color=ORANGE if model=='LR0' else TEAL if model.startswith('LR') else INK if model=='LGB-H0' else PURPLE
        text(model,cols[0],yy-1,106,17,INK,model in {'LR2','LGB-H0'})
        rect(bx,yy+2,bw,15,'#DCE7E7')
        if d['hits']:rect(bx,yy+2,bw*d['hits']/d['reviews'],15,color)
        text(str(d['hits']),cols[0]+colw-131,yy-2,64,18,color,True)
        text(f"{d['average_precision']:.3f}",cols[0]+colw-61,yy-1,61,17,MUTED)
        hit_data.append({'model':model,'hits':d['hits'],'reviews':d['reviews'],'ap':d['average_precision']})
    chart_data['test_primary']=hit_data;y+=7*29+13
    path_y=y;nw=(colw-44)/3
    for i,(title,model) in enumerate([('ORIGINAL','LR0'),('+ CLIP','LR1'),('+ L2','LR2')]):
        xx=cols[0]+i*(nw+22);rect(xx,path_y,nw,56,'#E1F0EC',5)
        text(title,xx+8,path_y+7,nw-16,14,TEAL,True)
        text(f"{metrics['test'][model]['period_rates']['0.005']['hits']} hits",xx+8,path_y+28,nw-16,20,INK,True)
        if i<2:arrow([(xx+nw,path_y+29),(xx+nw+22,path_y+29)])
    y=body(0,path_y+69,'Across F1/F2/validation/test, clipping (LR1-LR0) adds <b>+27/+13/+13/+376</b> hits. Stronger L2 (LR2-LR1) adds <b>-3/-1/-4/+7</b>; no consistent extra gain.',18)
    y=body(0,y,'<b>LR2-LR0: hit differences and paired 95% intervals</b>',18)
    lr_rows=[]
    for p,label in zip(PERIODS,['Dev F1','Dev F2','Valid.','Test']):
        d=summary['paired_analysis']['periods'][p]['contrasts']['LR2 minus LR0']['period_rate']['hits']
        lr_rows.append([label,f"{d['point_difference']:+.0f}",f"[{d['lower']:.2f}, {d['upper']:.2f}]"])
    y=compact_table(0,y,['Period','Diff.','3-day-block 95% interval'],lr_rows,[104,76,colw-180],16,29)
    y=body(0,y,'Amount baseline: <b>8/443 hits</b>; random expected hits: <b>15.42</b>. LR0 probability ranking gives 6 hits, margin ranking 0: removing probability saturation alone does not fix the linear ranking.',17,MUTED)
    ends=[y]

    # Column 2: all-period matrix, capacity sensitivity, and history evidence.
    y=section(1,860,'04','All periods, all candidates')
    y=body(1,y,'Fraud hits at <b>0.5% period capacity</b>. Budgets vary with row count; compare candidates within each period.',18)
    widths=[108]+[(colw-108)/4]*4
    matrix=[[model]+[str(metrics[p][model]['period_rates']['0.005']['hits']) for p in PERIODS] for model in MODELS]
    y=compact_table(1,y,['Model','F1','F2','Valid.','Test'],matrix,widths,17,29)
    budgets=[metrics[p]['LR0']['period_rates']['0.005']['reviews'] for p in PERIODS]
    y=compact_table(1,y,['Reviews']+[str(k) for k in budgets],[],widths,16)
    chart_data['period_hit_matrix']={'periods':PERIODS,'models':MODELS,'rows':matrix,'review_budgets':budgets}
    y=body(1,y,'<b>Test precision / recall / AP</b><br/>LR2: 86.46% / 12.42% / 0.370<br/>H0: 93.00% / 13.36% / 0.500<br/>H3: 91.42% / 13.14% / 0.497',17)
    y=section(1,y+4,'05','Capacity changes the result')
    x=cols[1];px=x+43;pw=colw-58;ph=137;py=y+8
    rates=plan['budget']['period_rates']
    def bp(rate):return px+(rate-rates[0])/(rates[-1]-rates[0])*pw
    for tick in [0,50,100]:
        yy=py+ph*(1-tick/100);line(px,yy,px+pw,yy);text(str(tick),x,yy-8,36,15,MUTED)
    budget_models=[('LR0',ORANGE),('LR2',TEAL),('LGB-H0',INK),('LGB-H3',PURPLE)]
    chart_data['budget_precision']={}
    for model,color in budget_models:
        values=[metrics['test'][model]['period_rates'][str(r)]['precision'] for r in rates]
        points=[(bp(r),py+ph*(1-v)) for r,v in zip(rates,values)]
        for a,b in zip(points,points[1:]):line(*a,*b,color,2.5)
        for xx,yy in points:
            c.setFillColor(HexColor(color));c.circle(xx,H-yy,3.6,stroke=0,fill=1)
        chart_data['budget_precision'][model]=values
    for rate in rates:text(f'{rate*100:g}%',bp(rate)-16,py+ph+13,50,15,MUTED)
    y=py+ph+42
    for i,(model,color) in enumerate(budget_models):
        xx=x+i*colw/4;line(xx,y+8,xx+17,y+8,color,2.5);text(model,xx+21,y,colw/4-21,15,color,True)
    y=body(1,y+28,'Y: precision (%); X: review rate. LR0 fails most at tight capacity; H0/H3 are close. Daily 8/16/32 and legacy Top100/500 are also reported.',17,MUTED)
    y=section(1,y+4,'06','Does history help over time?')
    y=body(1,y,'<b>H3-H0: change in fraud hits</b><br/>1,000 paired 3-day moving-block draws; percentile 95% intervals. Four evaluation periods share the fixed protocol.',18)
    fx=x+103;fw=colw-277;ft=y+7;fh=128
    def fxp(v):return fx+(v+30)/42*fw
    for tick in [-30,-20,-10,0,10]:
        line(fxp(tick),ft-5,fxp(tick),ft+fh,INK if tick==0 else RULE)
        text(str(tick),fxp(tick)-10,ft+fh+10,38,14,MUTED)
    forest=[]
    for i,(p,label) in enumerate(zip(PERIODS,['Dev F1','Dev F2','Valid.','Test'])):
        d=summary['paired_analysis']['periods'][p]['contrasts']['LGB-H3 minus LGB-H0']['period_rate']['hits'];yy=ft+i*39
        text(label,x,yy-10,95,18)
        line(fxp(d['lower']),yy,fxp(d['upper']),yy,PURPLE,3)
        c.setFillColor(HexColor(PURPLE));c.circle(fxp(d['point_difference']),H-yy,4.5,stroke=0,fill=1)
        text(f"{d['point_difference']:+.0f} [{d['lower']:.0f}, {d['upper']:.0f}]",x+colw-156,yy-10,156,17,PURPLE)
        forest.append({'period':p,**d})
    chart_data['history_contrast']=forest
    y=body(1,ft+fh+41,'Effects change sign: <b>+2, -11, +4, -7</b>. All main intervals cross zero; this does not prove equivalence. More features give no stable gain; no test-based deployment winner is selected.',18)
    y=body(1,y,'<b>K = ceil(rate x N)</b>; precision = hits/K; recall = hits/fraud labels. AP = average precision, not trapezoidal PR area.',17,MUTED)
    ends.append(y)

    # Column 3: diagnostics, matched capacity and sensitivity to dates/blocks.
    y=section(2,860,'07','Investigate the failure')
    y=body(2,y,'<b>V323 daily median</b>, nonmissing values; orange band marks days 171-172.',18)
    x=cols[2];px=x+49;pw=colw-66;py=y+9;ph=124
    daily=audit['focused_features']['V323']['by_test_day'];days=sorted(map(int,daily))
    def dx(day):return px+(day-days[0])/(days[-1]-days[0])*pw
    def dy(value):return py+ph*(1-value/1200)
    step=pw/(days[-1]-days[0]);rect(dx(170.5),py,2*step,ph,'#F7DCC8')
    for tick in [0,600,1200]:
        line(px,dy(tick),px+pw,dy(tick));text(str(tick),x,dy(tick)-8,44,15,MUTED)
    points=[(dx(d),dy(daily[str(d)]['quantiles']['0.5'])) for d in days]
    for a,b in zip(points,points[1:]):line(*a,*b,ORANGE,2.5)
    for d,label in [(152,'152'),(160,'160'),(171.5,'171-172'),(182,'182')]:text(label,dx(d)-17,py+ph+14,64 if d==171.5 else 40,15,MUTED)
    chart_data['v323_daily_median']=[{'day':d,'median':daily[str(d)]['quantiles']['0.5']} for d in days]
    y=body(2,py+ph+44,'<b>49 to 1,411</b>: V323 train/test maximum.<br/><b>442 of LR0\'s top 443 reviews</b> come from the two dates; all 443 are false positives.',18)
    y=body(2,y,'<b>394 columns x 3 partitions</b>: missingness, quantiles, constants, non-finite/unseen values and train-range exceedances. Drift, score contributions and date concentration support an extreme-input diagnosis, not a unique cause.',17)
    y=body(2,y,'<b>Duplicate audit:</b> 1,215 input groups (ID/label/time excluded); 399 cross partitions, 17 mixed-label. No automatic deduplication.',17,MUTED)
    y=section(2,y+4,'08','Match daily and global totals')
    y=body(2,y,'16 reviews/day = <b>496 test slots</b>; global ranking gets exactly the same total.',18)
    daily_rows=[]
    for model in ['LR0','LR2','LGB-H0','LGB-H3']:
        d=metrics['test'][model]['daily']['fixed_16'];daily_rows.append([model,str(d['hits']),str(d['matched_global']['hits']),f"{d['hits']-d['matched_global']['hits']:+d}"])
    y=compact_table(2,y,['Model','Daily','Global','Diff.'],daily_rows,[124,113,113,colw-350],17,29)
    full_adv=metrics['test']['LR0']['daily']['fixed_16']['hits']-metrics['test']['LR0']['daily']['fixed_16']['matched_global']['hits']
    omit_adv=omission['models']['LR0']['daily_minus_matched_global_hits'];boxw=(colw-30)/2
    for i,(label,value,detail) in enumerate([('ALL 31 DATES',full_adv,'496 matched slots'),('OMIT 171-172',omit_adv,'464 matched slots')]):
        xx=x+i*(boxw+30);rect(xx,y,boxw,72,'#F8EADD',5)
        text(label,xx+9,y+6,boxw-18,14,ORANGE,True)
        text(f'{value:+d} hits',xx+9,y+27,boxw-18,22,INK,True)
        text(detail,xx+9,y+55,boxw-18,13,MUTED)
    arrow([(x+boxw,y+37),(x+boxw+30,y+37)],ORANGE)
    chart_data['daily_comparison']={'rows':daily_rows,'full_lr0_advantage':full_adv,'omit_lr0_advantage':omit_adv}
    y=body(2,y+84,'LR0 daily advantage: <b>+301</b>, 3-day 95% interval <b>[-38, 341.02]</b>; omitting two dates gives -8. LR2 still leads by <b>87 period hits (362 vs 275; 414 slots)</b>.',17)
    y=section(2,y+4,'09','How sensitive is inference?')
    sensitivity_rows=[]
    for block in [1,3,7]:
        d=sensitivity['periods']['test']['blocks'][str(block)]['contrasts']['LGB-H3 minus LGB-H0']['period_rate']['hits']
        sensitivity_rows.append([f'{block} day' if block==1 else f'{block} days',f"[{d['lower']:.0f}, {d['upper']:.0f}]",'Yes' if d['lower']<=0<=d['upper'] else 'No'])
    y=body(2,y,'Test H3-H0 = -7 hits. Post-hoc block-length comparison:',17)
    y=compact_table(2,y,['Block length','95% interval','Crosses 0?'],sensitivity_rows,[163,168,colw-331],16,28)
    y=body(2,y,'All <b>332 continuous 1/2/7-day omissions</b> keep scores/history/models fixed and rematch budgets. The 7-day interval is negative; 1/3-day intervals cross zero.',17)
    y=body(2,y,'<b>31 test days</b> limit small-effect and long-term claims. Fixed-model intervals omit training uncertainty/multiplicity correction. Uncalibrated scores; no deployment, causal or financial benefit.',17,MUTED)
    ends.append(y)
    if max(ends)>2100:raise ValueError(f'Columns overflow evidence footer: {ends}')
    chart_data['column_bottoms']=ends

    # A separate verification ledger makes executed work countable.
    text('10',m,2116,43,19,TEAL,True)
    text('Executed work and independent checks',m+53,2111,full-53,26,INK,True)
    check=summary['verification']['checks'];extra=extra_qa['checks']
    qa_stats=[('10','correctness tests'),(str(check['period_model_pairs']),'period/config groups'),(f"{check['daily_groups']:,}",'daily evaluations'),(f"{check['queue_rows']:,}",'queue records'),(f"{extra['raw_missingness_field_partition_groups']:,}",'missingness checks'),(str(extra['independent_omission_model_checks']),'omission/model checks')]
    for i,(value,label) in enumerate(qa_stats):
        xx=m+i*full/6;rect(xx,2158,full/6-12,72,PALE,5)
        text(value,xx+11,2163,full/6-34,30,TEAL,True);text(label,xx+11,2204,full/6-34,16,MUTED)
    text('Supplement verification also checks 126 focused quantiles and 28 baseline groups. Three legacy reruns retain numerical/environment differences. Clean isolated installation: pip check, 10 tests and synthetic end-to-end passed; no full real-data refit in that environment. Source, code and artifact hashes are retained.',m,2241,full,17,limit=43)
    rect(m,2293,full,57,INK,7)
    final=[('REPAIR BEFORE COMPARING','Train-only clipping repairs much of the LR failure.'),('RETAIN NEGATIVE RESULTS','History gains did not persist across periods.'),('VALIDATE ON NEW TIME','Obtain a genuinely unseen labeled period next.')]
    tw=(full-44)/3
    for i,(title,desc) in enumerate(final):
        xx=m+14+i*(full-15)/3;text(title,xx,2301,tw,16,'#A3DED0',True);text(desc,xx,2326,tw,17,'#FFFFFF',limit=22)
    text('Retrospective study; assumed review budgets and label availability. Sources: verified study_20261121_01; executed_summary.json; supplement_20261121_01; frozen plan2. [1] Kaggle/Vesta, IEEE-CIS (2019). [2] Ke et al., LightGBM, NeurIPS (2017).',m,2358,full,10.5,MUTED,limit=14)
    for i,a in enumerate(boxes):
        for b in boxes[i+1:]:
            if min(a['x']+a['width'],b['x']+b['width'])-max(a['x'],b['x'])>.5 and min(a['bottom'],b['bottom'])-max(a['top'],b['top'])>.5:
                raise ValueError(f"Text overlap: {a['text'][:65]} / {b['text'][:65]}")
    c.showPage();c.save()
    manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),'page_mm':[594,841],
        'edition':'workflow_dense','layout':'workflow_plus_three_evidence_columns','language':'en','source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'text_boxes':boxes,'workflow_nodes':nodes,'arrows':arrows,'chart_data':chart_data}
    (OUT/'poster_workflow_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(target)
    print(f"{len(nodes)} workflow nodes, {len(arrows)} arrows, {len(boxes)} text regions")


if __name__ == '__main__':
    main()
