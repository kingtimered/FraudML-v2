"""Analyze saved study scores with paired moving-day-block intervals.

Uses multiplicities rather than copying sampled days. Average precision and
Top-K are exactly weighted over the duplicated fixed predictions, including ties.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results/.matplotlib"))

import numpy as np
import pandas as pd

from fraudml_v2.evaluation import WeightedRanking
from run_study import digest, write_json


def moving_block_weights(number_days, block_length, draws, seed):
    if number_days < block_length:
        raise ValueError("Fewer days than the specified block size")
    rng = np.random.default_rng(seed)
    output = np.zeros((draws, number_days), dtype=np.int16)
    for row in output:
        starts = rng.integers(0, number_days - block_length + 1, math.ceil(number_days / block_length))
        selected = (starts[:, None] + np.arange(block_length)).ravel()[:number_days]
        row[:] = np.bincount(selected, minlength=number_days)
    return output


def interval(draws, point):
    result = {}
    for index, metric in enumerate(["hits", "precision", "recall", "average_precision"]):
        values = draws[:, index]
        valid = values[np.isfinite(values)]
        result[metric] = {
            "point_difference": float(point[index]) if np.isfinite(point[index]) else None,
            "valid_draws": len(valid),
            "lower": float(np.quantile(valid, .025)) if len(valid) else None,
            "upper": float(np.quantile(valid, .975)) if len(valid) else None,
        }
    return result


def bootstrap_metrics(ranking, weights, rate, scenario_capacities):
    """Reuse each draw's cumulative counts for all matched capacities."""
    primary, matched = [], {name: [] for name in scenario_capacities}
    for index, day_weights in enumerate(weights):
        row_weights = day_weights[ranking.day_codes]
        counts = np.cumsum(row_weights)
        positives = np.cumsum(row_weights * ranking.y)
        total, fraud = int(counts[-1]), int(positives[-1])
        capacities = [math.ceil(rate * total)] + [int(values[index]) for values in scenario_capacities.values()]
        k = np.asarray(capacities)
        end = np.searchsorted(counts, k, side="left")
        previous_counts = np.where(end > 0, counts[np.maximum(0, end - 1)], 0)
        previous_hits = np.where(end > 0, positives[np.maximum(0, end - 1)], 0)
        hits = previous_hits + (k - previous_counts) * ranking.y[end]
        precision = np.divide(hits, k, out=np.full(len(k), np.nan), where=k > 0)
        recall = hits / fraud if fraud else np.full(len(k), np.nan)
        cumulative_hits = positives[ranking.tie_ends]
        cumulative_rows = counts[ranking.tie_ends]
        increments = np.diff(np.r_[0, cumulative_hits])
        tie_precision = np.divide(cumulative_hits, cumulative_rows, out=np.zeros(len(cumulative_rows)), where=cumulative_rows > 0)
        ap = float(np.sum(increments * tie_precision) / fraud) if fraud else np.nan
        values = np.column_stack([hits, precision, recall, np.full(len(k), np.nan)])
        values[0, 3] = ap
        primary.append(values[0])
        for position, name in enumerate(matched, 1):
            matched[name].append(values[position])
    return np.asarray(primary), {name: np.asarray(values) for name, values in matched.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--smoke-draws", type=int)
    args = parser.parse_args()
    out = ROOT / "results" / args.run_id
    run = json.loads((out / "run.json").read_text())
    if run["status"] not in {"executed", "executed_smoke", "verified"}:
        parser.error("Study run did not complete")
    if (out / "paired_intervals.json").exists():
        parser.error("Analysis already exists; do not overwrite")
    plan = json.loads((out / "protocol.json").read_text())
    if args.smoke_draws and run["kind"] != "synthetic_smoke":
        parser.error("Draw reduction is allowed only for synthetic smoke runs")
    draws = args.smoke_draws or plan["uncertainty"]["draws"]
    metrics = json.loads((out / "metrics.json").read_text())
    splits = json.loads((out / "splits.json").read_text())
    daily = pd.read_csv(out / "daily_metrics.csv")
    daily = daily.loc[daily.is_final_generation]
    primary = plan["budget"]["primary_rate"]
    contrasts = plan["main_contrasts"] + plan["diagnostic_contrasts"] + [["LGB-H0", "amount"]]
    all_intervals, score_manifest = {}, {}
    started = perf_counter()
    for period, configurations in metrics.items():
        print(f"Analyzing {period}: {draws} paired 3-day-block draws", flush=True)
        predictions, rankings, global_draws, daily_draws = {}, {}, {}, {}
        reference = None
        for model, values in configurations.items():
            generation = values["generation"]
            path = out / "predictions" / f"{period}__{model}__{generation}.parquet"
            table = pd.read_parquet(path).sort_values(["TransactionDT", "TransactionID"], kind="stable").reset_index(drop=True)
            if reference is None:
                reference = table[["TransactionID", "TransactionDT", "isFraud"]]
            elif not reference.equals(table[["TransactionID", "TransactionDT", "isFraud"]]):
                raise ValueError("Predictions not aligned across models")
            predictions[model] = table
            rankings[model] = WeightedRanking(table, table.score)
            score_manifest[path.name] = digest(path)
        days = next(iter(rankings.values())).days
        weights = moving_block_weights(len(days), plan["uncertainty"]["block_days"], draws, plan["seeds"]["bootstrap"])
        unit_weights = np.ones(len(days), dtype=int)
        day_index = pd.Index(days)
        for model, ranking in rankings.items():
            tables = {}
            capacities = {}
            for scenario, group in daily.loc[daily.period.eq(period) & daily.model.eq(model)].groupby("scenario"):
                table = group.set_index("day_id").reindex(day_index)
                if table.reviews.isna().any():
                    raise ValueError("Daily records are missing an evaluation date")
                capacity = weights @ table.reviews.to_numpy()
                hits = weights @ table.hits.to_numpy()
                positives = weights @ table.fraud.to_numpy()
                samples = np.column_stack([
                    hits, np.divide(hits, capacity, out=np.full(draws, np.nan), where=capacity > 0),
                    np.divide(hits, positives, out=np.full(draws, np.nan), where=positives > 0),
                    np.full(draws, np.nan),
                ])
                capacities[scenario] = capacity
                tables[scenario] = {"samples": samples}
            global_draws[model], matched = bootstrap_metrics(ranking, weights, primary, capacities)
            for scenario in tables:
                tables[scenario]["matched"] = matched[scenario]
            daily_draws[model] = tables
        period_intervals = {"contrasts": {}, "daily_vs_matched_global": {}}
        for lhs, rhs in contrasts:
            left, right = configurations[lhs], configurations[rhs]
            valid = bool(left.get("converged", True) and right.get("converged", True))
            record = {
                "lhs": lhs, "rhs": rhs, "primary_contrast": [lhs, rhs] in plan["main_contrasts"],
                "comparison_eligible": valid,
                "warning": None if valid else "At least one LR configuration did not converge; do not claim comparative superiority.",
                "period_rate": interval(global_draws[lhs] - global_draws[rhs], rankings[lhs].compute(unit_weights, primary) - rankings[rhs].compute(unit_weights, primary)),
                "daily": {},
            }
            for scenario in daily_draws[lhs]:
                def point(model):
                    row = configurations[model]["daily"][scenario]
                    return np.asarray([row["hits"], row["precision"], row["recall"], np.nan])
                record["daily"][scenario] = interval(daily_draws[lhs][scenario]["samples"] - daily_draws[rhs][scenario]["samples"], point(lhs) - point(rhs))
            period_intervals["contrasts"][f"{lhs} minus {rhs}"] = record
        for model in configurations:
            period_intervals["daily_vs_matched_global"][model] = {}
            for scenario, values in daily_draws[model].items():
                row = configurations[model]["daily"][scenario]
                point = np.asarray([
                    row["hits"] - row["matched_global"]["hits"],
                    row["precision"] - row["matched_global"]["precision"],
                    row["recall"] - row["matched_global"]["recall"], np.nan,
                ])
                difference = values["samples"] - values["matched"]
                difference[:, 3] = np.nan
                period_intervals["daily_vs_matched_global"][model][scenario] = interval(difference, point)
        all_intervals[period] = period_intervals
    analysis = {
        "run_id": args.run_id, "kind": run["kind"], "method": plan["uncertainty"],
        "actual_draws": draws, "score_sha256": score_manifest, "elapsed_seconds": perf_counter() - started,
        "analysis_code_sha256": digest(Path(__file__)),
        "periods": all_intervals,
        "limitations": ["Conditional on fixed fitted models and observed dates", "No correction for multiple diagnostic comparisons", "Three-day blocks do not guarantee removal of all temporal/entity dependence"],
    }
    write_json(out / "paired_intervals.json", analysis)
    make_report(out, run, plan, metrics, analysis)
    print(f"Wrote paired intervals and Chinese result report ({perf_counter()-started:.1f}s)", flush=True)


def make_report(out, run, plan, metrics, analysis):
    lines = ["# 新版实验结果", "", f"运行编号：{run['run_id']}。类型：{run['kind']}。协议：{plan['protocol_version']}。", ""]
    if run["kind"] == "synthetic_smoke":
        lines += ["**本文件仅为合成数据流程检查，不能用作作业模型结论。**", ""]
    lines += ["分析为回顾性比较，所有固定候选完整展示；未按测试表现选择赢家。", "", "## 全期审核0.5%的主要结果", "", "| 期别 | 配置 | 收敛 | 命中/名额 | 精确率 | 召回率 | AP |", "|---|---|---|---:|---:|---:|---:|"]
    key = f"{plan['budget']['primary_rate']:g}"
    for period, models in metrics.items():
        for model, result in models.items():
            row = result["period_rates"][key]
            lines.append(f"| {period} | {model} | {'是' if result['converged'] else '否'} | {row['hits']}/{row['reviews']} | {row['precision']:.3%} | {row['recall']:.3%} | {row['average_precision']:.6f} |")
    lines += ["", "## 预先指定的主要增量对照", "", "差异方向为左侧配置减右侧配置。区间为固定模型的1000次配对3日块重采样；合成试跑以实际次数为准。", "", "| 期别 | 对照 | 命中差异及95%区间 | 精确率差异及95%区间（百分点） | 可用于比较 |", "|---|---|---:|---:|---|"]
    for period, results in analysis["periods"].items():
        for name, contrast in results["contrasts"].items():
            if not contrast["primary_contrast"]:
                continue
            hits = contrast["period_rate"]["hits"]
            precision = contrast["period_rate"]["precision"]
            lines.append(f"| {period} | {name} | {hits['point_difference']:+.0f} [{hits['lower']:.1f}, {hits['upper']:.1f}] | {100*precision['point_difference']:+.2f} [{100*precision['lower']:.2f}, {100*precision['upper']:.2f}] | {'是' if contrast['comparison_eligible'] else '否：LR未收敛'} |")
    lines += ["", "## 随机审核的条件期望", "", "期望命中=审核名额×本期欺诈比例。seed42只是一组示例，不用该示例偶然的高低判断收益。", "", "| 期别 | 全期0.5%名额 | 期望命中 | 实际随机示例命中 |", "|---|---:|---:|---:|"]
    for period, models in metrics.items():
        row = models["random_seed42"]["period_rates"][key]
        lines.append(f"| {period} | {row['reviews']} | {row['random_expected_hits']:.2f} | {row['hits']} |")
    lines += ["", "## 解释性消融（全部预设对照）", "", "下表仅为机制诊断；区间包含0不等于证明等价，不从多个对照中筛选显著结果。", "", "| 期别 | 对照 | 命中差异及95%区间 | AP差异及95%区间 |", "|---|---|---:|---:|"]
    for period, results in analysis["periods"].items():
        for name, contrast in results["contrasts"].items():
            if [contrast["lhs"], contrast["rhs"]] not in plan["diagnostic_contrasts"]:
                continue
            h, ap = contrast["period_rate"]["hits"], contrast["period_rate"]["average_precision"]
            lines.append(f"| {period} | {name} | {h['point_difference']:+.0f} [{h['lower']:.1f}, {h['upper']:.1f}] | {ap['point_difference']:+.4f} [{ap['lower']:.4f}, {ap['upper']:.4f}] |")
    lines += ["", "## 每日16笔及相同总名额的全期对照", "", "全期Top-K是依赖整期分数的离线参照；日末固定名额是另一种约束。差异不能解释为可直接实现的实时收益。", "", "| 期别 | 配置 | 每日策略命中/名额 | 同总名额全期命中 | 命中差异及95%区间 |", "|---|---|---:|---:|---:|"]
    for period, models in metrics.items():
        for model in ["amount", "LR0", "LR2", "LGB-H0", "LGB-H3"]:
            row = models[model]["daily"]["fixed_16"]
            ci = analysis["periods"][period]["daily_vs_matched_global"][model]["fixed_16"]["hits"]
            lines.append(f"| {period} | {model} | {row['hits']}/{row['reviews']} | {row['matched_global']['hits']} | {row['hits']-row['matched_global']['hits']:+d} [{ci['lower']:.1f}, {ci['upper']:.1f}] |")
    lines += ["", "## 证据与解释边界", "", "- primary_results.csv、metrics.json保存所有期别、预算和原Top100/500口径；分数与模型均保存于本地。", "- paired_intervals.json同时给出预先指定对照与解释性对照，不按区间是否跨0筛选。", "- review_queues.parquet、daily_metrics.csv、weekly_metrics.csv对应同一排序与容量。", "- probability_diagnostics.json记录概率饱和与margin变化；diagnostics.json记录收敛、迭代和截尾范围。", "- 原始、统一重试的模型与分数都保留；默认表只展示最终代次，未收敛比较标注受限。", "- 两个开发评价期不重叠，但模型训练量不同且交易可能跨日相关。原验证/测试共用同一个训练快照。", "- 只对观察期、假设容量和日批处理作结论；不推断长期稳定、因果作用、真实损失避免或实时部署效果。", ""]
    (out / "新版实验结果.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
