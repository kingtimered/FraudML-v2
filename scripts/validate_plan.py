"""Validate plan2 configuration, source snapshots, local links and data audit.

Only the Python standard library is required. This does not fit any model.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, message):
    require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), message)


def main():
    plan_path = ROOT / "configs/study_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    require(plan["protocol_version"] == "2026-11-21.plan2", "Wrong protocol version")
    require(plan["status"] == "planned" and plan["training_executed"] is False, "Unexpected training status")
    require(plan["analysis_type"] == "retrospective", "Must retain retrospective analysis label")
    require(plan["label_gap_days"] == 0, "Unsupported label-delay assumption")
    require(not plan["data"]["identity_modeling"], "Identity modeling is outside core scope")
    require(not plan["budget"]["drop_endpoint_days_by_default"], "Do not blindly drop endpoint days")
    require(plan["budget"]["match_total_capacity"], "Capacity matching is required")
    require(plan["budget"]["daily_role"] == "supplementary", "Daily operations are supplementary")
    require(plan["budget"]["primary_rate"] in plan["budget"]["period_rates"], "Primary rate is missing")
    require(all(0 < rate <= 1 for rate in plan["budget"]["period_rates"]), "Invalid audit proportion")
    require(not plan["selection"]["winner_selection"] and not plan["selection"]["test_based_tuning"], "Test winner selection is prohibited")

    models = plan["models"]
    ids = [model["id"] for model in models]
    require(len(models) == len(set(ids)) == 7, "Expected seven distinct core models")
    for pair in plan["main_contrasts"] + plan["diagnostic_contrasts"]:
        require(len(pair) == 2 and all(model in ids for model in pair), "Unknown contrast model")
    folds = plan["development_folds"]
    require(len(folds) == 2, "Expected two development folds")
    last_end = None
    for fold in folds:
        start, end = fold["train_days"]
        eval_start, eval_end = fold["evaluation_days"]
        require(start < end == eval_start < eval_end, "Invalid or gapped fold")
        require(eval_end - eval_start == 28, "Expected 28-day evaluation")
        require(last_end is None or eval_start >= last_end, "Overlapping evaluation windows")
        last_end = eval_end
    require(plan["fit_budget"]["snapshots"] == len(folds) + 1, "Snapshot budget mismatch")
    require(plan["fit_budget"]["core_fits"] == len(models) * (len(folds) + 1) == 21, "Fit count mismatch")
    require(not plan["later_snapshot"]["refit_per_period"], "Later periods must share a fitted model")

    for relative in plan["required_docs"] + plan["required_scripts"]:
        path = ROOT / relative
        require(path.is_file(), f"Missing planned file: {relative}")
        if path.suffix == ".md":
            content = path.read_text(encoding="utf-8")
            require("2026-11-21.plan1" not in content, f"Stale version: {relative}")
            for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", content):
                if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("#"):
                    continue
                require((path.parent / target.split("#", 1)[0]).exists(), f"Broken link: {relative} -> {target}")

    manifest = json.loads((ROOT / "evidence/source_manifest.json").read_text(encoding="utf-8"))
    snapshots = set()
    for record in manifest["files"]:
        source = ROOT / record["source"]
        require(source.is_file(), f"Missing source: {source}")
        require(sha256(source) == record["sha256"], f"Source changed: {source}")
        require(source.stat().st_size == record["bytes"], f"Source size mismatch: {source}")
        if "snapshot" in record:
            copy = ROOT / record["snapshot"]
            require(copy.is_file() and sha256(copy) == record["sha256"], f"Invalid snapshot: {copy}")
            snapshots.add(copy.name)
    require(snapshots == set(plan["legacy_reports"]), "Legacy snapshot list mismatch")
    for name in plan["legacy_reports"]:
        report = json.loads((ROOT / "evidence/legacy_metrics" / name).read_text(encoding="utf-8"))
        positives = report["evaluation_fraud_count"]
        for model, result in report["results"].items():
            for key, metric in result.items():
                if not key.startswith("k="):
                    continue
                k, hits = metric["k"], metric["hits_at_k"]
                require(0 <= hits <= min(k, positives), f"Invalid hits: {name}/{model}")
                close(metric["precision_at_k"], hits / k, f"Precision mismatch: {name}/{model}")
                close(metric["recall_at_k"], hits / positives, f"Recall mismatch: {name}/{model}")
                require(0 <= metric["pr_auc"] <= 1, f"Invalid AP: {name}/{model}")

    audit = json.loads((ROOT / "evidence/data_audit.json").read_text(encoding="utf-8"))
    require(audit["protocol_version"] == plan["protocol_version"], "Audit version mismatch")
    require(audit["plan_sha256"] == sha256(plan_path), "Audit is stale for this configuration")
    require(not audit["training_executed"], "Data audit must not claim training")
    require(audit["scope"]["full_feature_quality_audit"] is False, "Overstated audit scope")
    require(audit["overall"]["rows"] == 590540 and audit["overall"]["fraud"] == 20663, "Unexpected input snapshot")
    for record in audit["inputs"]:
        path = ROOT / record["path"]
        require(path.is_file() and path.stat().st_size == record["bytes"], f"Data file missing or size changed: {path}")
        require(re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is not None, "Invalid data digest")
    # Do not silently hash 1.3 GB on every static check. The auditor records the
    # input hashes; re-running it is required to verify current data bytes.
    for fold in folds:
        actual = audit["development_folds"][fold["id"]]
        require(actual["evaluation"]["observed_days"] == 28, "Audit fold day count mismatch")
        require(actual["train"]["end_dt"] < actual["evaluation"]["start_dt"], "Audit leakage boundary failure")
    original = json.loads((ROOT / plan["later_snapshot"]["bounds_source"]).read_text(encoding="utf-8"))
    for name, boundary in original["split"]["periods"].items():
        summary = audit["original_periods"][name]
        for field in ["rows", "start_dt", "end_dt"]:
            require(summary[field] == boundary[field], f"Audit original period mismatch: {name}/{field}")
        require(sum(day["rows"] for day in summary["daily"]) == summary["rows"], "Daily row count mismatch")
        require(sum(day["fraud"] for day in summary["daily"]) == summary["fraud"], "Daily fraud count mismatch")
        for budget in plan["budget"]["daily_fixed"]:
            expected = sum(min(budget, day["rows"]) for day in summary["daily"])
            require(summary["daily_fixed_budgets"][str(budget)]["total_reviews"] == expected, "Capacity mismatch")
    print("PASS: plan2 files, local links, fixed contrasts, 28-day windows and 21-fit budget")
    print("PASS: source hashes, four legacy snapshots and metric arithmetic")
    print("PASS: data-audit configuration, archived boundaries and daily capacities")
    print("Input data hashes are recorded; this static check only checks current file sizes.")
    print("No model training, model improvement, statistical power or future validity is established.")


if __name__ == "__main__":
    main()
