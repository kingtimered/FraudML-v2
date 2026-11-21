"""Independently recompute saved rankings, queues and capacity arithmetic."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from run_study import digest, write_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message):
    require(actual is None and expected is None or actual is not None and expected is not None and
            math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    out = ROOT / "results" / args.run_id
    run = json.loads((out / "run.json").read_text())
    require(run["status"] in {"executed", "executed_smoke", "verified"}, "Incomplete run")
    require(not (out / "verification.json").exists(), "Verification already exists")
    plan = json.loads((out / "protocol.json").read_text())
    require(digest(out / "protocol.json") == digest(ROOT / "configs/study_plan.json"), "Protocol changed")
    require(digest(ROOT / "configs/study_plan.json") == run["protocol_sha256"], "Original protocol digest changed")
    metrics = json.loads((out / "metrics.json").read_text())
    splits = json.loads((out / "splits.json").read_text())
    diagnostics = json.loads((out / "diagnostics.json").read_text())
    checks = json.loads((out / "integration_checks.json").read_text())
    require(all(row["passed"] for row in checks), "Prediction integration check failed")
    expected_fits = {(snapshot, model["id"]) for snapshot in splits for model in plan["models"]}
    actual_fits = {(row["snapshot"], row["model"]) for row in run["completed_fits"] if row["generation"] == "initial"}
    require(expected_fits == actual_fits and len(actual_fits) == 21, "Core fit count mismatch")
    for snapshot, model_results in diagnostics.items():
        initial_failed = any(not values["initial"]["converged"] for model, values in model_results.items() if model.startswith("LR"))
        if initial_failed and run["kind"] == "full_study":
            require(all("uniform_retry" in model_results[model] for model in ["LR0", "LR1", "LR2"]), "Incomplete uniform LR retry")
    source = None
    if run["kind"] == "full_study":
        data = (ROOT / plan["data"]["transaction"]).resolve()
        require(digest(data) == run["data_sha256"], "Input data bytes changed")
        audit = json.loads((ROOT / "evidence/data_audit.json").read_text())
        audited = next(record for record in audit["inputs"] if (ROOT / record["path"]).resolve() == data)
        require(run["data_sha256"] == audited["sha256"], "Training input differs from audited data")
        source = pd.read_csv(data, usecols=["TransactionID", "TransactionDT", "isFraud"]).set_index("TransactionID")
    daily = pd.read_csv(out / "daily_metrics.csv").loc[lambda frame: frame.is_final_generation]
    weekly = pd.read_csv(out / "weekly_metrics.csv").loc[lambda frame: frame.is_final_generation]
    queues = pd.read_parquet(out / "review_queues.parquet").loc[lambda frame: frame.is_final_generation]
    manifest, counts = {}, {"period_model_pairs": 0, "daily_groups": 0, "queue_rows": 0}
    for period, candidates in metrics.items():
        reference = None
        snapshot = next(name for name, split in splits.items() if period in split["evaluations"])
        for model, result in candidates.items():
            path = out / "predictions" / f"{period}__{model}__{result['generation']}.parquet"
            scores = pd.read_parquet(path).sort_values(["TransactionDT", "TransactionID"], kind="stable").reset_index(drop=True)
            require(not scores.TransactionID.duplicated().any(), "Duplicate prediction IDs")
            require(np.isfinite(scores.score).all(), "Invalid prediction scores")
            ids_hash = hashlib.sha256(scores.TransactionID.to_numpy(dtype="int64").tobytes()).hexdigest()
            require(ids_hash == splits[snapshot]["evaluations"][period]["ids_sha256"], "Split IDs changed")
            core = scores[["TransactionID", "TransactionDT", "isFraud"]]
            if reference is None:
                reference = core
            else:
                require(core.equals(reference), "Candidates evaluated on different rows")
            if source is not None:
                input_rows = source.loc[scores.TransactionID].reset_index().loc[:, core.columns]
                require(input_rows.equals(core), "Prediction labels or timestamps differ from input")
            ranked = scores.sort_values(["score", "TransactionDT", "TransactionID"], ascending=[False, True, True], kind="stable")
            fraud = int(scores.isFraud.sum())
            ap = float(average_precision_score(scores.isFraud, scores.score)) if fraud else None
            for rate, row in result["period_rates"].items():
                k = math.ceil(float(rate) * len(scores))
                hits = int(ranked.head(k).isFraud.sum())
                require(row["reviews"] == k and row["hits"] == hits, "Period Top-K mismatch")
                close(row["precision"], hits / k, "Period precision mismatch")
                close(row["recall"], hits / fraud if fraud else None, "Period recall mismatch")
                close(row["average_precision"], ap, "Period AP mismatch")
            for scenario, total in result["daily"].items():
                selected_ids, total_hits, total_reviews = [], 0, 0
                table = daily.loc[daily.period.eq(period) & daily.model.eq(model) & daily.scenario.eq(scenario)].set_index("day_id")
                queue = queues.loc[queues.period.eq(period) & queues.model.eq(model) & queues.scenario.eq(scenario)]
                for day, group in scores.groupby(scores.TransactionDT // 86400):
                    k = min(len(group), math.ceil(len(group) * float(scenario[5:])) if scenario.startswith("rate_") else int(scenario[6:]))
                    chosen = group.sort_values(["score", "TransactionDT", "TransactionID"], ascending=[False, True, True], kind="stable").head(k)
                    hits = int(chosen.isFraud.sum())
                    row = table.loc[day]
                    require(int(row.reviews) == k and int(row.hits) == hits, "Daily metrics mismatch")
                    require(int(row.fraud) == int(group.isFraud.sum()) and int(row.rows) == len(group), "Daily denominators mismatch")
                    close(row.precision, hits / k, "Daily precision mismatch")
                    q = queue.loc[queue.day_id.eq(day)].sort_values("review_rank")
                    require(q.TransactionID.tolist() == chosen.TransactionID.tolist(), "Queue selection mismatch")
                    require(q.review_rank.tolist() == list(range(1, k + 1)), "Queue ranks mismatch")
                    np.testing.assert_allclose(q.score, chosen.score, rtol=0, atol=0)
                    selected_ids.extend(chosen.TransactionID.tolist())
                    total_hits += hits
                    total_reviews += k
                    counts["daily_groups"] += 1
                require(total_hits == total["hits"] and total_reviews == total["reviews"], "Daily totals mismatch")
                require(len(queue) == total_reviews and not queue.TransactionID.duplicated().any(), "Queue duplicates or capacity mismatch")
                require(total["matched_global"]["reviews"] == total_reviews, "Unequal total review capacity")
                require(int(ranked.head(total_reviews).isFraud.sum()) == total["matched_global"]["hits"], "Matched global hits mismatch")
                week = weekly.loc[weekly.period.eq(period) & weekly.model.eq(model) & weekly.scenario.eq(scenario)]
                require(int(week.hits.sum()) == total_hits and int(week.reviews.sum()) == total_reviews, "Weekly totals mismatch")
                counts["queue_rows"] += len(queue)
            counts["period_model_pairs"] += 1
            manifest[str(path.relative_to(out))] = digest(path)
    for path in (out / "models").glob("*.joblib"):
        manifest[str(path.relative_to(out))] = digest(path)
    analysis_path = out / "paired_intervals.json"
    if run["kind"] == "full_study":
        require(analysis_path.exists(), "Full verification requires paired analysis")
    if analysis_path.exists():
        analysis = json.loads(analysis_path.read_text())
        if run["kind"] == "full_study":
            require(analysis["actual_draws"] == 1000, "Incomplete bootstrap analysis")
        for name, expected in analysis["score_sha256"].items():
            require(digest(out / "predictions" / name) == expected, "Analysis used different scores")
    require(counts["period_model_pairs"] == 36, "Expected 4 periods by 9 candidates")
    write_json(out / "verification.json", {"passed": True, "kind": run["kind"], "checks": counts,
        "artifact_sha256": manifest, "verification_code_sha256": digest(Path(__file__)),
        "analysis_present": analysis_path.exists(), "limitations": ["Does not independently retrain models", "Bootstrap coverage and model generalization are not proven by arithmetic checks"]})
    run["status"] = "verified" if run["kind"] == "full_study" else "executed_smoke"
    run["verification_passed"] = True
    write_json(out / "run.json", run)
    print(f"PASS: {counts}")


if __name__ == "__main__":
    main()
