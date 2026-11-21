"""Build the Chinese coursework report and post-hoc figures from saved evidence."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_study import ROOT, digest, write_json


def load(path):
    return json.loads(path.read_text())


def ci(row):
    return f"{row['point_difference']:+.0f} [{row['lower']:.2f}, {row['upper']:.2f}]"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, default=ROOT / "docs/新版作业报告.md")
    args = parser.parse_args()
    evidence = args.evidence_dir.resolve()
    report = args.report_path.resolve()
    supplement = evidence / "补充分析结果.md"
    figures = evidence / "figures"
    if report.exists() or supplement.exists() or figures.exists():
        parser.error("Refusing to overwrite reports or figures")
    config = load(ROOT / "configs/supplement_plan.json")
    study = ROOT / "results" / config["source_run_id"]
    quality = load(evidence / "quality_audit.json")
    sensitivity = load(evidence / "sensitivity.json")
    clean = load(evidence / "clean_environment.json")
    verification = load(evidence / "verification.json")
    if clean["status"] != "passed" or not verification["passed"]:
        raise ValueError("Environment and supplemental checks must pass before report generation")
    metrics = load(study / "metrics.json")
    prior = load(study / "paired_intervals.json")
    run = load(study / "run.json")
    summary = load(ROOT / "evidence/executed_summary.json")
    models = config["sensitivity"]["models"]
    contrasts = config["sensitivity"]["main_contrasts"]
    omit_focus = next(row for row in sensitivity["periods"]["test"]["omissions"] if row["omitted_days"] == [171, 172])
    total_omissions = sum(len(row["omissions"]) for row in sensitivity["periods"].values())
    lines = ["# 数据质量与稳健性补充分析", "", "本次分析在查看主实验结果后制定，属于事后诊断；不增加候选、训练或测试后选模。主实验文件和冻结plan2均保留。", "",
             "## 数据质量", "", f"已审计有标签交易表全部{quality['audited_columns']}列的训练、验证、测试分区。数值分位数、缺失率与极值只作描述；未把异常列或日期从主实验删除。", "",
             "| 字段 | 训练缺失率 | 测试缺失率 | 训练最大值 | 测试最大值 | 第171/172日有值中位数 | 其他测试日有值中位数 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for field in config["quality_audit"]["focused_features"]:
        row = quality["features"][field]
        focus = quality["focused_features"][field]
        lines.append(f"| {field} | {row['train']['missing_rate']:.2%} | {row['test']['missing_rate']:.2%} | {row['train']['quantiles']['1']:g} | {row['test']['quantiles']['1']:g} | {focus['days_171_172']['quantiles']['0.5']:g} | {focus['other_test_days']['quantiles']['0.5']:g} |")
    lines += ["", "V323等字段的训练与测试取值范围明显不同，且高值在部分日期聚集。这与已有线性贡献和日期集中度诊断一致，但无法从匿名字段确定变化原因，也不能将单一字段认定为唯一因果机制。这里的z值按训练期有值样本计算，与模型先填补再标准化的z值不同。", "",
              "| 重复口径 | 重复组 | 组内交易数 | 额外重复行 | 跨分区组 | 标签不一致组 |", "|---|---:|---:|---:|---:|---:|"]
    for name, row in quality["duplicates"].items():
        label = "输入特征含时间，排除ID/标签" if name == "including_time" else "输入特征排除ID/标签/时间"
        lines.append(f"| {label} | {row['duplicate_groups']} | {row['rows_in_duplicate_groups']} | {row['extra_duplicate_rows']} | {row['cross_partition_groups']} | {row['conflicting_label_groups']} |")
    lines += ["", "重复候选先由哈希筛选，再按完整原始特征逐值核实。无时间输入相同不等于同一真实交易：可能存在合法重复支付、匿名信息不足或标签差异，不能直接认定泄漏或误标，也不自动去重。训练期没有全缺失输入列；V107的有值部分为常量。", "",
              "## 块长敏感性", "", "各设置均使用1000次配对重采样，固定种子；3日块直接沿用主分析。1日块只是忽略跨日相关的参照，7日块包含的有效块很少。区间不含训练不确定性，没有多重比较校正。", "",
              "| 期别 | 对照 | 点差异及1日块95%区间 | 点差异及3日块95%区间 | 点差异及7日块95%区间 |", "|---|---|---:|---:|---:|"]
    for period, row in sensitivity["periods"].items():
        for lhs, rhs in contrasts:
            name = f"{lhs} minus {rhs}"
            values = [ci(row["blocks"][str(block)]["contrasts"][name]["period_rate"]["hits"]) for block in [1, 3, 7]]
            lines.append(f"| {period} | {lhs}−{rhs} | {' | '.join(values)} |")
    lines += ["", "LR2−LR0在测试期的三个区间均为正，但开发F2及验证期的一些区间触及或跨过0，因此不能说每期、每种统计设定都显著。H3−H0的测试期7日块区间完全为负，而1/3日块跨0；不可只挑一种块长宣布显著或无差异。共同支持的表述是：没有稳定正增益的证据，部分设定提示负向影响。", "",
              "## 日期影响分析", "", f"遍历每期全部连续1/2/7日日窗，共{total_omissions}个省略场景，七配置均计算。模型、原始评分及原历史序列保持固定，省略只作用于评价行；不表示重新训练或删除交易后的业务仿真。每次按剩余交易重算0.5%名额；每日16笔与全期对照仍匹配实际总容量。", "",
              "| 期别 | 省略天数 | LR2−LR0命中差范围 | H3−H0命中差范围 | LR0每日16笔减同容量全期命中差范围 |", "|---|---:|---:|---:|---:|"]
    for period, row in sensitivity["periods"].items():
        for size in [1, 2, 7]:
            selected = [r for r in row["omissions"] if len(r["omitted_days"]) == size]
            arrays = [[r["contrasts"][f"{a} minus {b}"]["period_hits"] for r in selected] for a, b in contrasts]
            arrays.append([r["models"]["LR0"]["daily_minus_matched_global_hits"] for r in selected])
            ranges = [f"[{min(v):+d}, {max(v):+d}]" for v in arrays]
            lines.append(f"| {period} | {size} | {' | '.join(ranges)} |")
    lines += ["", "上述最小/最大值是影响范围，**不是置信区间**；不同省略场景的数据和总预算也不同。不能将筛出的最好场景替代主结果。", "",
              "### 第171、172日的具体影响", "", "这两个日期由已观察的LR0高分集中现象选取，是事后解释案例，已同时公开所有连续日窗结果。", "",
              "| 配置 | 完整测试期0.5%命中/名额 | 排除两日后命中/名额 | 完整期每日16笔−同容量全期命中 | 排除后每日16笔−同容量全期命中 |", "|---|---:|---:|---:|---:|"]
    for model in ["LR0", "LR1", "LR2", "LGB-H0", "LGB-H3"]:
        old, changed = metrics["test"][model], omit_focus["models"][model]
        primary, daily = old["period_rates"]["0.005"], old["daily"]["fixed_16"]
        difference = daily["hits"] - daily["matched_global"]["hits"]
        lines.append(f"| {model} | {primary['hits']}/{primary['reviews']} | {changed['period_hits']}/{changed['period_reviews']} | {difference:+d} | {changed['daily_minus_matched_global_hits']:+d} |")
    lines += ["", "LR0恢复到275/414，而LR2仍达到362/414，说明异常日期解释了相当部分退化，但不足以解释全部差距。LR0的每日固定容量优势在该省略场景反转，更说明不能推广成长期最优分配策略。原完整测试期仍是主结果。", "",
              "## 独立安装与核验", "", f"在include-system-site-packages=false的新虚拟环境完成依赖安装、pip check、10项正确性测试，以及完整的合成训练→分析→独立核验。实际依赖另存requirements-clean-macos-arm64.txt。主真实数据实验没有在此环境重训。补充计算已与原指标核对{sensitivity['checks']['baseline_model_period_groups']}组，并独立过滤、排序核对{sensitivity['checks']['independent_omission_model_checks']}组省略结果。", "",
              "## 文件", "", "- quality_audit.json：394列、三分区、关键字段每日分布及重复检查。", "- sensitivity.json：全部块长区间、全部连续日窗省略结果与来源哈希。", "- clean_environment.json：独立环境、命令、输出和合成核验。", "- figures/：本次图表与输入/输出哈希。", ""]
    supplement.write_text("\n".join(lines))
    figures.mkdir()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 170})
    periods = list(sensitivity["periods"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = {1: "#9a6d2e", 3: "#176c83", 7: "#7c418d"}
    for ax, (lhs, rhs) in zip(axes, contrasts):
        for block in [1, 3, 7]:
            for index, period in enumerate(periods):
                row = sensitivity["periods"][period]["blocks"][str(block)]["contrasts"][f"{lhs} minus {rhs}"]["period_rate"]["precision"]
                y = index + (block == 1) * -.18 + (block == 7) * .18
                ax.hlines(y, 100 * row["lower"], 100 * row["upper"], color=colors[block], linewidth=2)
                ax.scatter([100 * row["point_difference"]], [y], s=25, color=colors[block], label=f"{block}-day blocks" if index == 0 else None)
        ax.axvline(0, color="black", linewidth=.8)
        ax.set_yticks(range(len(periods)), periods)
        ax.invert_yaxis()
        ax.set_title(f"{lhs} minus {rhs}")
        ax.set_xlabel("Precision difference (percentage points)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Post-hoc block-length sensitivity | fixed scores, 0.5% period budget")
    fig.tight_layout(rect=[0, .08, 1, .95])
    for suffix in ["png", "pdf"]:
        fig.savefig(figures / f"block_sensitivity.{suffix}")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, field in zip(axes, ["V323", "V332", "V324"]):
        by_day = quality["focused_features"][field]["by_test_day"]
        x = [int(day) for day in by_day]
        for q, label, color in [("0.5", "Observed median", "#186f86"), ("0.99", "Observed 99th percentile", "#8a418e")]:
            y = [np.log10(1 + by_day[str(day)]["quantiles"][q]) if by_day[str(day)]["quantiles"] else np.nan for day in x]
            ax.plot(x, y, label=label, color=color)
        train_max = quality["features"][field]["train"]["quantiles"]["1"]
        ax.axhline(np.log10(1 + train_max), color="gray", linestyle="--", label="Train observed maximum")
        ax.axvspan(170.5, 172.5, color="#debc6f", alpha=.25)
        ax.set_title(field)
        ax.set_xlabel("Relative day (test)")
        ax.set_ylabel("log10(1 + raw value)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Post-hoc raw-feature audit | shaded: days 171 and 172 | missing values excluded")
    fig.tight_layout(rect=[0, .08, 1, .95])
    for suffix in ["png", "pdf"]:
        fig.savefig(figures / f"feature_daily_drift.{suffix}")
    plt.close(fig)

    report_lines = ["# 支付欺诈排序的时间稳定性与特征改进评估", "", "DSC5001新版课程作业报告｜实测版｜2026年11月21日", "",
        "## 摘要", "", "本项目在有限审核容量下评价支付欺诈排序，重点研究数值处理、历史特征和日期分布如何影响模型结论。使用IEEE-CIS有标签交易表590,540笔交易、394列、20,663笔欺诈，设置两个28天开发评价期，并对原验证与测试期作回顾性比较。固定七配置、三个训练快照，共21次核心拟合；全部逻辑回归收敛。全期0.5%审核预算下，测试期原逻辑回归margin排序命中0/443，经训练期分位数截尾后为376/443，结合更强正则化后为383/443；基础LightGBM为412/443，历史增强版H3为405/443。历史增益随时期反转，未显示稳定正收益。新增394列质量审计和块长/日期影响分析进一步显示：极端输入和少数日期会显著影响排序及每日名额比较。所有结论限于观察数据和假设预算，不解释为长期人力最优配置或真实财务收益。", "",
        "## 1. 研究问题与相对原方案的贡献", "", "原方案已经包含时间切分、训练期预处理、严格历史统计、固定Top-K评价、LR失效讨论和本地服务原型。新版保留这些基础，集中补充可重复的对照证据，回答三个问题：", "",
        "1. 固定审核比例下，不同模型的排序结论是否跨时期、跨预算保持？", "2. 训练期截尾与更强正则化，分别为逻辑回归带来什么变化？", "3. 修正缺失实体处理、增加近期历史，是否稳定改善基础LightGBM？", "",
        "每日8/16/32笔是补充的日末批处理场景。新版不研究跨日动态排班。全部改进及与原方案对应关系见[原方案改进对照](原方案改进对照.md)。", "",
        "## 2. 数据与时间设计", "", "原数据保持不变。交易ID、标签与原始切分时间不进入模型；identity只做覆盖审计。数据共有182个相对日桶，交易量与日期量分别报告。", "",
        "| 评价期 | 训练笔数 | 评价笔数 | 欺诈数 | 日桶数 | 0.5%名额 |", "|---|---:|---:|---:|---:|---:|"]
    train_counts = {"dev_F1": 223738, "dev_F2": 309348, "validation": 413378, "test": 413378}
    for period in metrics:
        row = metrics[period]["LR0"]["period_rates"]["0.005"]
        report_lines.append(f"| {period} | {train_counts[period]:,} | {row['rows']:,} | {row['fraud']:,} | {len(sensitivity['periods'][period]['observed_days'])} | {row['reviews']} |")
    report_lines += ["", "开发期评价日窗为[61,89)、[89,117)，评价期互不重叠，但后期训练包含前期数据，因此不能视为独立重复样本。原验证和测试使用同一个原训练快照，避免将该对比中的变化混同为重训影响。历史标签假定在训练截点已可用，缺少真实标签到达时间。原测试已被查看，所有比较均标注回顾性；固定候选不能将其重新变成盲测。", "",
        "## 3. 方法与评价口径", "", "LR0为原数值处理、C=1；LR1仅增加训练期0.1%–99.9%分位数截尾；LR2在LR1基础上改为C=0.1。三者以decision_function排序，同时保留概率诊断。LGB-H0为基础LightGBM；H1增加四个累计历史字段并沿用缺失实体共享组；H2仅修正该缺失处理；H3再增加card1和addr1各自的24小时次数和距上笔时间。模型参数固定，没有大范围搜索。", "",
        "逐值规范化保证123、123.0及其字符串形式一致。同时间戳整组计算后再更新历史，严格使用<t信息；24小时窗为[t−86400,t)，不用标签。未知实体并非已验证的真实身份。预处理阈值、编码、填补与缩放只在训练期拟合。", "",
        "主预算为全期0.5%，并完整报告0.1%、1%、2%及旧Top100/500口径。命中数、精确率和召回率为主指标，AP为补充；AP指average_precision_score，不混称梯形PR曲线面积。随机基线以同容量条件期望为主，保留seed42示例。每日预算向上取整后与全期对照匹配实际总名额。", "",
        "主要对照为LR2−LR0和H3−H0；分解截尾、正则化与历史构成的对照均完整保留。主分析使用1000次配对3日移动块重采样，区间只条件于固定模型与观察日期，不含训练不确定性，也不校正诊断比较的多重性。", "",
        "## 4. 全期0.5%审核结果", "", "| 期别 | 配置 | 命中/名额 | 精确率 | 召回率 | AP |", "|---|---|---:|---:|---:|---:|"]
    for period, candidates in metrics.items():
        for model in models:
            row = candidates[model]["period_rates"]["0.005"]
            report_lines.append(f"| {period} | {model} | {row['hits']}/{row['reviews']} | {row['precision']:.2%} | {row['recall']:.2%} | {row['average_precision']:.6f} |")
    report_lines += ["", "| 期别 | LR2−LR0命中差及95%区间 | H3−H0命中差及95%区间 |", "|---|---:|---:|"]
    for period in metrics:
        rows = [ci(prior["periods"][period]["contrasts"][f"{a} minus {b}"]["period_rate"]["hits"]) for a, b in contrasts]
        report_lines.append(f"| {period} | {' | '.join(rows)} |")
    report_lines += ["", "训练期截尾明显缓解LR的后期退化。LR2相对LR1的命中差依次为−3、−1、−4、+7，因此不能只凭测试期提升宣称更强正则化稳定有效。LR0概率排序测试期为6/443，而margin排序为0/443：解除概率饱和没有自动修复底层线性排序。", "",
        "H3相对H0的主预算差异依次为+2、−11、+4、−7，主分析区间均跨0，不能宣称稳定提升，也不能据此证明两者等价。AP与队列前端命中关注不同排序区域，需结合报告。基础LightGBM表现良好不意味着该模型在未观测时期必然稳健。", "",
        "## 5. 质量审计与LR退化解释", "", "新增审计逐列描述全部394列的训练、验证、测试缺失率、唯一值、常量、非有限值、分位数及超出训练范围的计数；类别列另记未知值。V323训练最大49、测试最大1411；V324由33变为976。V323在第171/172日有值中位数为733，其他测试日为0。某些字段的缺失率也明显变化，例如M8测试较训练下降约28.29个百分点。", "",
        "已有模型贡献分解显示，LR0测试最高分443笔全部误报，其中442笔来自第171/172日；V323等字段有巨大正贡献，另一些字段存在巨大负向抵消。贡献重构误差小于7×10⁻¹³。分布、得分贡献与日期集中度共同支持极端输入关联的解释，但匿名语义、字段相关性与未做因果干预限制了原因识别。", "",
        "排除ID和标签、保留时间时，发现3组共6笔相同输入，均未跨原分区；再排除时间后，发现1215组、共2718笔，其中399组跨分区、17组标签不一致。这些组合不必然是同一交易，不能直接断言泄漏或标注错误。没有自动去重或测试后筛列。", "",
        "## 6. 块长与日期影响", "", "补充分析完整计算1/3/7日块，并遍历每期全部连续1/2/7日省略窗，共332个场景。原3日分析保留，新增分析明确为事后。LR2−LR0测试期区间在三种块长均为正，但开发F2及验证部分设置触及或跨0。H3−H0测试期的1/3/7日块区间分别为[−19,1]、[−20,1]、[−19,−1]，说明具体统计判断对设定敏感；整体仍不支持历史特征的稳定正收益。", "",
        "省略第171/172日后，LR0为275/414、LR2为362/414，H0为389/414、H3为383/414。异常日解释了部分差距，但LR修复的剩余增量仍有87笔。该分析只过滤评价行，历史和模型均固定，不能当作删除困难数据后得到的更好主实验。完整日窗结果与更多指标见[补充分析结果](../evidence/supplement_20261121_01/补充分析结果.md)。", "",
        "## 7. 每日有限名额的补充场景", "", "测试期每日16笔共496笔，使用同样496笔的全期排序作比较：", "",
        "| 配置 | 每日命中 | 同容量全期命中 | 每日减全期及3日块95%区间 |", "|---|---:|---:|---:|"]
    for model in ["LR0", "LR2", "LGB-H0", "LGB-H3"]:
        row = metrics["test"][model]["daily"]["fixed_16"]
        contrast = prior["periods"]["test"]["daily_vs_matched_global"][model]["fixed_16"]["hits"]
        report_lines.append(f"| {model} | {row['hits']} | {row['matched_global']['hits']} | {ci(contrast)} |")
    report_lines += ["", "每日容量限制减少了少数异常日期占据LR0全部名额的程度，但其主区间跨0；对其他模型则减少命中。省略第171/172日后，LR0每日相对全期从+301变为−8，进一步显示该现象依赖日期组成。每日策略不能概括为普适改善。", "",
        "测试期每天0.5%取整共458笔，全期0.5%为443笔；新版已匹配容量，避免把多审核15笔的效果误归因于策略。8/16/32均无真实团队容量依据。59万行足以观察显著排序退化，但测试只有31个日桶，不足以保证识别几笔命中的细微差异或长期人力规律。", "",
        "## 8. 复现、证据与原报告处理", "", "主实验状态verified：独立复算36组时期×配置、4320组每日评价和76995条队列记录；10项正确性测试覆盖规范化、历史、边界、分块、截尾、排序、重采样与保存加载。补充分析与原分数对齐28组，独立直接过滤和排序检查259组省略结果。全列缺失率和重点分位数另作独立复算。", "",
        "原代码另完成3次重跑：基础LightGBM复现归档至1e−10容差；旧LR及旧历史版部分数值略变，原值和环境差异全部保留，尚未逐项隔离依赖影响。当前原代码与新版共同基线在两后期的六组逐笔margin/概率均在1e−10容差内一致。", "",
        "新建不继承系统包的独立环境，完成锁定主要依赖安装、pip check、10项测试和合成端到端核验；完整真实数据没有在该环境再次拟合。主实验实际环境与独立安装环境的完整依赖分别保留，不宣称所有平台结果逐位一致。", "",
        "旧Word记载的部分CV、PDP、置换重要性、子群分析及共享附录缺少完整可执行证据，本报告不继承其数值结论。原HTTP/Java原型作为原项目工程扩展保留，新版验收聚焦离线评分、历史与队列一致性。", "",
        "## 9. 讨论与局限", "", "本作业的主要贡献是将失效现象转成受控对照、把历史特征收益拆解成可检验增量，并用不确定性和日期影响约束结论。增加复杂特征没有稳定获益本身就是有效研究结果。", "",
        "仍缺乏新的未查看时期、标签成熟时间、上游匿名字段定义、真实调查结果和审核成本。日期相关、实体相关、训练长度变化与有限时间跨度均限制外推。没有概率校准、财务收益、因果机制或生产可用性证据。增加重采样次数不会产生新月份。下一步优先获取真正新的带标签时期，并按事前协议评价，而不是继续按已有测试结果加模型。", "",
        "## 参考资料与附录", "", "本报告沿用原方案的数据与方法出处；未新增系统性文献综述。", "",
        "1. Kaggle. IEEE-CIS Fraud Detection. 2019. 数据来源沿用原项目记录。", "2. Ke, G., et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS, 2017.", "3. Saito, T., and Rehmsmeier, M. The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. PLOS ONE, 2015. DOI: 10.1371/journal.pone.0118432. 此文是原项目指标选择的参考，不代替本项目的统计验证。", "",
        "可复算依据：[主实验汇总](../evidence/executed_summary.json)、[冻结方案](../configs/study_plan.json)、[补充配置](../configs/supplement_plan.json)、[补充分析](../evidence/supplement_20261121_01/补充分析结果.md)、[复现指南](复现指南.md)。报告由scripts/build_report.py从实际结果生成，来源与输出哈希见补充目录report_manifest.json。", ""]
    if total_omissions != 332 or len(run["completed_fits"]) != 21:
        raise ValueError("Report's execution counts differ from saved evidence")
    report.parent.mkdir(parents=True, exist_ok=True)
    report_text = "\n".join(report_lines).replace("../evidence/supplement_20261121_01/补充分析结果.md", os.path.relpath(supplement, report.parent))
    report_text = report_text.replace("59万行足以观察显著排序退化", "59万行足以观察明显排序退化")
    figure_section = ["## 研究图表", "", "图1展示全部预定预算的结果；图2和图3为事后补充分析。", ""]
    for title, path in [("图1：固定候选的审核预算曲线", study / "figures/budget_precision.png"),
                        ("图2：关键匿名字段的测试期每日分布", figures / "feature_daily_drift.png"),
                        ("图3：不同块长下的主要差异区间", figures / "block_sensitivity.png")]:
        figure_section += [f"![{title}]({os.path.relpath(path, report.parent)})", "", title + "。", ""]
    report_text = report_text.replace("## 参考资料与附录", "\n".join(figure_section) + "## 参考资料与附录")
    report.write_text(report_text)
    sources = [evidence / name for name in ["quality_audit.json", "sensitivity.json", "clean_environment.json", "verification.json"]]
    sources += [study / name for name in ["metrics.json", "paired_intervals.json", "run.json"]]
    sources += [ROOT / "evidence/executed_summary.json", ROOT / "configs/supplement_plan.json"]
    outputs = [report, supplement, *sorted(figures.iterdir())]
    write_json(evidence / "report_manifest.json", {"command": sys.argv, "code_sha256": digest(Path(__file__)),
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "outputs_sha256": {str(p.relative_to(ROOT)): digest(p) for p in outputs}})
    print(f"Wrote {report}, {supplement} and two PNG/PDF figures")


if __name__ == "__main__":
    main()
