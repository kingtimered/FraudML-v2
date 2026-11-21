"""Export a small shareable evidence snapshot, without scores or models."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from run_study import ROOT, digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--legacy-run-id", required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "evidence/executed_summary.json")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Evidence snapshot already exists")
    out, legacy = ROOT / "results" / args.run_id, ROOT / "results" / args.legacy_run_id
    run = json.loads((out / "run.json").read_text())
    old = json.loads((legacy / "run.json").read_text())
    if run["status"] != "verified" or old["status"] != "executed":
        parser.error("Requires verified study and completed legacy replay")
    metrics = json.loads((out / "metrics.json").read_text())
    analysis = json.loads((out / "paired_intervals.json").read_text())
    verification = json.loads((out / "verification.json").read_text())
    diagnostic = json.loads((out / "lr_score_contributions.json").read_text())
    concentration = json.loads((out / "lr_day_concentration.json").read_text())
    alignment = json.loads((out / "legacy_alignment.json").read_text())
    if not (out / "figures/manifest.json").exists():
        parser.error("Missing figures")
    now = datetime.now(timezone.utc).isoformat()
    run["delivery_completed_at_utc"] = now
    run["manifest"] = [str(path.relative_to(out)) for path in sorted(out.rglob("*")) if path.is_file() and path.name != "run.json"]
    write_json(out / "run.json", run)
    source_files = [out / name for name in ["run.json", "metrics.json", "paired_intervals.json", "verification.json", "lr_score_contributions.json", "lr_day_concentration.json", "legacy_alignment.json", "figures/manifest.json"]]
    source_files += [legacy / "run.json", legacy / "archived_comparison.json"]
    summary = {"created_at_utc": now, "scope": "Verified summary only; row scores and models remain local",
        "export_code_sha256": digest(Path(__file__)),
        "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in source_files},
        "study": {key: run[key] for key in ["run_id", "status", "kind", "analysis_type", "protocol_version", "protocol_sha256", "data_sha256", "environment", "completed_fits", "elapsed_seconds"]},
        "splits": json.loads((out / "splits.json").read_text()),
        "metrics": {period: {model: {key: values[key] for key in ["converged", "generation", "period_rates", "daily", "legacy_budgets"]} for model, values in models.items()} for period, models in metrics.items()},
        "paired_analysis": {"method": analysis["method"], "actual_draws": analysis["actual_draws"], "periods": analysis["periods"], "limitations": analysis["limitations"]},
        "verification": {key: verification[key] for key in ["passed", "checks", "analysis_present", "limitations"]},
        "lr_descriptive_diagnostic": diagnostic,
        "lr_day_concentration": concentration,
        "legacy_alignment": alignment,
        "legacy_replay": {"run_id": old["run_id"], "status": old["status"], "all_archived_metrics_match": old["all_archived_metrics_match"], "environment": old["environment"], "execution_differences": old["execution_differences"], "comparisons": json.loads((legacy / "archived_comparison.json").read_text())}}
    write_json(args.out, summary)
    print(f"Wrote shareable evidence: {args.out}")


if __name__ == "__main__":
    main()
