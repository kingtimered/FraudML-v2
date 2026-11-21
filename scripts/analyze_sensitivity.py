"""Post-hoc block-length and exhaustive date-influence checks on fixed scores."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from analyze_results import bootstrap_metrics, interval, moving_block_weights
from fraudml_v2.evaluation import WeightedRanking
from run_study import ROOT, digest, write_json


def daily_metrics(table, weights):
    capacity = weights @ table.reviews.to_numpy()
    hits = weights @ table.hits.to_numpy()
    fraud = weights @ table.fraud.to_numpy()
    return np.column_stack([hits, hits / capacity, hits / fraud, np.full(len(weights), np.nan)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.out_dir / "sensitivity.json"
    if output.exists():
        parser.error("Refusing to overwrite existing sensitivity analysis")
    config_path = ROOT / "configs/supplement_plan.json"
    config = json.loads(config_path.read_text())
    settings = config["sensitivity"]
    out = ROOT / "results" / config["source_run_id"]
    run = json.loads((out / "run.json").read_text())
    if run["status"] != "verified" or run["kind"] != "full_study":
        parser.error("Requires verified real-data scores")
    metrics = json.loads((out / "metrics.json").read_text())
    original = json.loads((out / "paired_intervals.json").read_text())
    if original["actual_draws"] != settings["draws"] or original["method"]["block_days"] != 3:
        raise ValueError("Existing three-day analysis does not match supplementary settings")
    plan = json.loads((out / "protocol.json").read_text())
    if plan["seeds"]["bootstrap"] != settings["seed"] or plan["budget"]["primary_rate"] != settings["period_rate"]:
        raise ValueError("Main protocol mismatch")
    daily = pd.read_csv(out / "daily_metrics.csv")
    daily = daily.loc[daily.is_final_generation & daily.scenario.eq(settings["daily_scenario"])]
    contrasts = settings["main_contrasts"] + settings["diagnostic_contrasts"]
    rate = settings["period_rate"]
    source_files = [out / name for name in ["run.json", "metrics.json", "paired_intervals.json", "daily_metrics.csv", "protocol.json"]]
    periods, checks = {}, {"baseline_model_period_groups": 0, "independent_omission_model_checks": 0, "three_day_intervals_reused": True}
    started = perf_counter()
    for period, results in metrics.items():
        print(f"Analyzing {period}: blocks 1/3/7 and every 1/2/7-day omission", flush=True)
        frames, rankings, tables = {}, {}, {}
        reference = None
        for model in settings["models"]:
            path = out / "predictions" / f"{period}__{model}__{results[model]['generation']}.parquet"
            source_files.append(path)
            if digest(path) != original["score_sha256"][path.name]:
                raise ValueError("Scores changed since core analysis")
            frame = pd.read_parquet(path).sort_values(["TransactionDT", "TransactionID"], kind="stable").reset_index(drop=True)
            keys = frame[["TransactionID", "TransactionDT", "isFraud"]]
            if reference is not None and not reference.equals(keys):
                raise ValueError("Misaligned candidate transactions")
            reference = keys
            frames[model], rankings[model] = frame, WeightedRanking(frame, frame.score)
            days = rankings[model].days
            tables[model] = daily.loc[daily.period.eq(period) & daily.model.eq(model)].set_index("day_id").reindex(days)
            if tables[model].reviews.isna().any():
                raise ValueError("Missing daily records")
            point = rankings[model].compute(np.ones(len(days), dtype=int), rate)
            expected = results[model]["period_rates"][f"{rate:g}"]
            if not np.allclose(point, [expected[k] for k in ["hits", "precision", "recall", "average_precision"]], atol=1e-12, rtol=0):
                raise ValueError("Saved baseline metrics do not match scores")
            checks["baseline_model_period_groups"] += 1
        if not np.all(np.diff(days) == 1):
            raise ValueError("Contiguous day windows require consecutive observed days")
        unit = np.ones(len(days), dtype=int)
        points = {model: ranking.compute(unit, rate) for model, ranking in rankings.items()}
        daily_points = {model: daily_metrics(table, unit[None, :])[0] for model, table in tables.items()}
        blocks = {}
        for size in settings["block_days"]:
            if size == 3:
                prior = original["periods"][period]
                blocks["3"] = {"source": "unchanged_core_analysis", "contrasts": {}, "daily_vs_matched_global": {}}
                for lhs, rhs in contrasts:
                    name = f"{lhs} minus {rhs}"
                    row = prior["contrasts"][name]
                    blocks["3"]["contrasts"][name] = {
                        "primary_contrast": row["primary_contrast"], "period_rate": row["period_rate"],
                        "daily": {settings["daily_scenario"]: row["daily"][settings["daily_scenario"]]}}
                blocks["3"]["daily_vs_matched_global"] = {
                    model: prior["daily_vs_matched_global"][model][settings["daily_scenario"]] for model in settings["models"]}
                continue
            weights = moving_block_weights(len(days), size, settings["draws"], settings["seed"])
            global_samples, daily_samples, matched_samples = {}, {}, {}
            for model, ranking in rankings.items():
                capacity = weights @ tables[model].reviews.to_numpy()
                global_samples[model], matched = bootstrap_metrics(ranking, weights, rate, {"daily16": capacity})
                matched_samples[model] = matched["daily16"]
                daily_samples[model] = daily_metrics(tables[model], weights)
            record = {"source": "post_hoc_fixed_scores", "contrasts": {}, "daily_vs_matched_global": {}}
            for lhs, rhs in contrasts:
                record["contrasts"][f"{lhs} minus {rhs}"] = {
                    "primary_contrast": [lhs, rhs] in settings["main_contrasts"],
                    "period_rate": interval(global_samples[lhs] - global_samples[rhs], points[lhs] - points[rhs]),
                    "daily": {settings["daily_scenario"]: interval(daily_samples[lhs] - daily_samples[rhs], daily_points[lhs] - daily_points[rhs])}}
            for model in rankings:
                capacity = int(tables[model].reviews.sum())
                matched_point = rankings[model].compute(unit, rate, capacity=capacity)
                point = daily_points[model] - matched_point
                point[3] = np.nan
                samples = daily_samples[model] - matched_samples[model]
                samples[:, 3] = np.nan
                record["daily_vs_matched_global"][model] = interval(samples, point)
            blocks[str(size)] = record
        omissions = []
        for size in settings["omit_every_contiguous_window_days"]:
            for start in range(len(days) - size + 1):
                weights = unit.copy()
                weights[start:start + size] = 0
                record = {"omitted_days": days[start:start + size].tolist(), "models": {}, "contrasts": {}}
                independently_check = start in {0, (len(days) - size) // 2, len(days) - size} or (size == 2 and days[start] == 171)
                for model, ranking in rankings.items():
                    value = ranking.compute(weights, rate)
                    table = tables[model]
                    capacity = int(weights @ table.reviews.to_numpy())
                    daily_hits = int(weights @ table.hits.to_numpy())
                    matched_hits = int(ranking.compute(weights, rate, capacity=capacity)[0])
                    remaining_rows = int(weights[ranking.day_codes].sum())
                    record["models"][model] = {
                        "rows": remaining_rows, "period_reviews": math.ceil(rate * remaining_rows),
                        "period_hits": int(value[0]), "period_precision": float(value[1]),
                        "period_recall": float(value[2]), "average_precision": float(value[3]),
                        "daily_reviews": capacity, "daily_hits": daily_hits,
                        "matched_global_hits": matched_hits, "daily_minus_matched_global_hits": daily_hits - matched_hits}
                    if independently_check:
                        # Independent direct filtering and pandas sorting, without WeightedRanking.
                        kept = frames[model].loc[~(frames[model].TransactionDT // 86400).isin(record["omitted_days"])]
                        ordered = kept.sort_values(["score", "TransactionDT", "TransactionID"], ascending=[False, True, True], kind="stable")
                        direct_k = math.ceil(rate * len(kept))
                        direct_daily = sum(int(group.sort_values(["score", "TransactionDT", "TransactionID"], ascending=[False, True, True], kind="stable").head(16).isFraud.sum())
                                           for _, group in kept.groupby(kept.TransactionDT // 86400))
                        if int(ordered.head(direct_k).isFraud.sum()) != value[0] or int(ordered.head(capacity).isFraud.sum()) != matched_hits or direct_daily != daily_hits:
                            raise ValueError("Independent omission check failed")
                        checks["independent_omission_model_checks"] += 1
                for lhs, rhs in contrasts:
                    left, right = [record["models"][model] for model in [lhs, rhs]]
                    if left["period_reviews"] != right["period_reviews"] or left["daily_reviews"] != right["daily_reviews"]:
                        raise ValueError("Mismatched omission capacities")
                    record["contrasts"][f"{lhs} minus {rhs}"] = {
                        "period_hits": left["period_hits"] - right["period_hits"],
                        "period_precision": left["period_precision"] - right["period_precision"],
                        "daily_hits": left["daily_hits"] - right["daily_hits"]}
                omissions.append(record)
        periods[period] = {"observed_days": days.tolist(), "blocks": blocks, "omissions": omissions}
    payload = {"status": "executed_with_checks", "analysis_type": config["analysis_type"], "training_executed": False,
               "created_at_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": perf_counter() - started,
               "config_sha256": digest(config_path), "code_sha256": digest(Path(__file__)),
               "implementation_sha256": {name: digest(ROOT / name) for name in ["fraudml_v2/evaluation.py", "scripts/analyze_results.py"]},
               "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in source_files},
               "settings": settings, "checks": checks, "periods": periods, "limitations": config["interpretation"]}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(output, payload)
    print(f"Wrote {output} ({payload['elapsed_seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
