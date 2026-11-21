"""Independently check supplemental provenance and chunkwise raw missingness."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_data import ROOT, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    args.evidence_dir = args.evidence_dir.resolve()
    destination = args.evidence_dir / "verification.json"
    if destination.exists():
        parser.error("Verification already exists")
    quality = json.loads((args.evidence_dir / "quality_audit.json").read_text())
    sensitivity = json.loads((args.evidence_dir / "sensitivity.json").read_text())
    clean = json.loads((args.evidence_dir / "clean_environment.json").read_text())
    config = json.loads((ROOT / "configs/supplement_plan.json").read_text())
    main_plan = json.loads((ROOT / "configs/study_plan.json").read_text())
    source = (ROOT / main_plan["data"]["transaction"]).resolve()
    if digest(source) != quality["source_sha256"]:
        raise ValueError("Data hash mismatch")
    for item, script in [(quality, "audit_quality.py"), (sensitivity, "analyze_sensitivity.py")]:
        if item["config_sha256"] != digest(ROOT / "configs/supplement_plan.json") or item["code_sha256"] != digest(ROOT / "scripts" / script):
            raise ValueError("Supplement code/config hash mismatch")
    for name, value in {**sensitivity["source_sha256"], **sensitivity["implementation_sha256"]}.items():
        if digest(ROOT / name) != value:
            raise ValueError(f"Source changed: {name}")
    if clean["status"] != "passed" or "include-system-site-packages = false" not in clean["pyvenv_cfg"]:
        raise ValueError("Clean environment has not passed")
    if any(check["exit_code"] for check in clean["checks"]):
        raise ValueError("Clean environment includes failed checks")
    audit = json.loads((ROOT / "evidence/data_audit.json").read_text())
    counts = {name: pd.Series(0, index=quality["features"], dtype="int64") for name in quality["partition_counts"]}
    rows = {name: 0 for name in counts}
    fraud = {name: 0 for name in counts}
    for chunk in pd.read_csv(source, chunksize=20000, low_memory=False):
        for name, bounds in audit["original_periods"].items():
            part = chunk.loc[chunk.TransactionDT.between(bounds["start_dt"], bounds["end_dt"])]
            counts[name] += part.isna().sum().reindex(counts[name].index)
            rows[name] += len(part)
            fraud[name] += int(part.isFraud.sum())
    checked_missingness = 0
    for name, missing in counts.items():
        expected = quality["partition_counts"][name]
        if rows[name] != expected["rows"] or fraud[name] != expected["fraud"]:
            raise ValueError("Raw partition totals differ")
        for column, count in missing.items():
            if count != quality["features"][column][name]["missing"]:
                raise ValueError(f"Missing count differs: {column}, {name}")
            checked_missingness += 1
    fields = config["quality_audit"]["focused_features"]
    selected = pd.read_csv(source, usecols=["TransactionDT", *fields])
    checked_quantiles = 0
    for field in fields:
        for name, bounds in audit["original_periods"].items():
            values = selected.loc[selected.TransactionDT.between(bounds["start_dt"], bounds["end_dt"]), field].dropna()
            direct = values.quantile(config["quality_audit"]["numeric_quantiles"]).to_numpy()
            expected = [quality["features"][field][name]["quantiles"][f"{q:g}"] for q in config["quality_audit"]["numeric_quantiles"]]
            if not np.allclose(direct, expected, rtol=1e-12, atol=1e-12):
                raise ValueError("Focused quantiles differ")
            checked_quantiles += len(direct)
    original = json.loads((ROOT / "results" / config["source_run_id"] / "paired_intervals.json").read_text())
    for period, item in sensitivity["periods"].items():
        for name, row in item["blocks"]["3"]["contrasts"].items():
            if row["period_rate"] != original["periods"][period]["contrasts"][name]["period_rate"]:
                raise ValueError("Original intervals changed")
        expected = sum(len(item["observed_days"]) - size + 1 for size in config["sensitivity"]["omit_every_contiguous_window_days"])
        if len(item["omissions"]) != expected:
            raise ValueError("Missing omission windows")
    result = {"passed": True, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "analysis_type": "independent_supplement_verification", "code_sha256": digest(Path(__file__)),
              "checks": {"raw_missingness_field_partition_groups": checked_missingness,
                         "focused_raw_quantile_values": checked_quantiles,
                         "all_omission_windows_present": True, "original_three_day_intervals_unchanged": True,
                         "clean_environment_passed": True, **sensitivity["checks"]},
              "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in [args.evidence_dir / name for name in ["quality_audit.json", "sensitivity.json", "clean_environment.json"]]},
              "limitations": ["Does not establish statistical coverage or causal mechanism", "Clean environment tested synthetic end-to-end; no full real-data refit"]}
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["checks"], ensure_ascii=False))


if __name__ == "__main__":
    main()
