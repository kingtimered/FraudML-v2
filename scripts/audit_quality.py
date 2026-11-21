"""Audit every raw field in bounded-memory batches; never fit or edit data.

Writes a separate post-hoc evidence file. Hash-screened duplicate candidates are
checked against complete raw feature values before reporting exact duplicates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from audit_data import ROOT, digest


def write_json(path, value):
    def native(item):
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=native, allow_nan=False) + "\n")


def describe(series, quantiles, reference=None):
    result = {"rows": len(series), "missing": int(series.isna().sum()),
              "missing_rate": float(series.isna().mean()), "unique_nonmissing": int(series.nunique()),
              "dtype": str(series.dtype)}
    result["all_missing"] = result["unique_nonmissing"] == 0
    result["constant_nonmissing"] = result["unique_nonmissing"] == 1
    if pd.api.types.is_numeric_dtype(series):
        array = series.to_numpy(dtype=float)
        finite = array[np.isfinite(array)]
        result.update({"kind": "numeric", "infinite": int(np.isinf(array).sum()),
                       "quantiles": {f"{q:g}": float(v) for q, v in zip(quantiles, np.quantile(finite, quantiles))} if len(finite) else {},
                       "mean": float(finite.mean()) if len(finite) else None,
                       "std_population": float(finite.std()) if len(finite) else None})
        if reference is not None and reference["quantiles"]:
            lower, upper = [reference["quantiles"][key] for key in ["0.001", "0.999"]]
            outside = (finite < lower) | (finite > upper)
            sd = reference["std_population"]
            result["outside_train_001_999_count"] = int(outside.sum())
            result["outside_train_001_999_rate_among_finite"] = float(outside.mean()) if len(finite) else None
            result["max_abs_train_observed_z"] = float(np.abs((finite - reference["mean"]) / sd).max()) if len(finite) and sd else None
            result["outside_train_min_max_count"] = int(((finite < reference["quantiles"]["0"]) | (finite > reference["quantiles"]["1"])).sum())
    else:
        counts = series.value_counts().head(5)
        result.update({"kind": "categorical", "top_values": [{"value": str(k), "count": int(v)} for k, v in counts.items()]})
    if reference is not None:
        result["missing_rate_change_pp"] = 100 * (result["missing_rate"] - reference["missing_rate"])
    return result


def verify_duplicates(path, keys, signatures, dtype_map, subset):
    candidate = pd.Series(signatures).duplicated(keep=False).to_numpy()
    if not candidate.any():
        return {"candidate_rows": 0, "rows_in_duplicate_groups": 0, "extra_duplicate_rows": 0,
                "duplicate_groups": 0, "cross_partition_groups": 0, "conflicting_label_groups": 0}
    pieces, offset = [], 0
    for chunk in pd.read_csv(path, dtype=dtype_map, chunksize=20000):
        mask = candidate[offset:offset + len(chunk)]
        if mask.any():
            part = chunk.loc[mask].copy()
            part["_signature"] = signatures[offset:offset + len(chunk)][mask]
            part["_partition"] = keys.partition.iloc[offset:offset + len(chunk)].to_numpy()[mask]
            pieces.append(part)
        offset += len(chunk)
    candidates = pd.concat(pieces, ignore_index=True)
    groups, cross, conflicts = 0, 0, 0
    for _, group in candidates.groupby("_signature", sort=False):
        if len(group[subset].drop_duplicates()) != 1:
            raise ValueError("Hash collision detected: exact duplicate summary must not be inferred from hashes")
        groups += 1
        cross += int(group._partition.nunique() > 1)
        conflicts += int(group.isFraud.nunique() > 1)
    return {"candidate_rows": len(candidates), "rows_in_duplicate_groups": len(candidates),
            "extra_duplicate_rows": int(candidates.duplicated(subset=subset).sum()),
            "duplicate_groups": groups, "cross_partition_groups": cross, "conflicting_label_groups": conflicts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Refusing to overwrite existing evidence")
    started = perf_counter()
    config_path = ROOT / "configs/supplement_plan.json"
    config = json.loads(config_path.read_text())
    plan = json.loads((ROOT / "configs/study_plan.json").read_text())
    source = (ROOT / plan["data"]["transaction"]).resolve()
    source_hash = digest(source)
    old_audit = json.loads((ROOT / "evidence/data_audit.json").read_text())
    if source_hash != old_audit["inputs"][0]["sha256"]:
        raise ValueError("Input differs from core study audit")
    keys = pd.read_csv(source, usecols=["TransactionID", "TransactionDT", "isFraud"])
    if keys.TransactionID.isna().any() or keys.TransactionID.duplicated().any() or not keys.isFraud.isin([0, 1]).all():
        raise ValueError("Invalid identifiers or labels")
    keys["day_id"] = keys.TransactionDT // 86400
    keys["partition"] = ""
    for name, bounds in old_audit["original_periods"].items():
        mask = keys.TransactionDT.between(bounds["start_dt"], bounds["end_dt"])
        if keys.loc[mask, "partition"].ne("").any() or mask.sum() != bounds["rows"]:
            raise ValueError("Partition mismatch")
        keys.loc[mask, "partition"] = name
    if keys.partition.eq("").any():
        raise ValueError("Unassigned rows")
    masks = {name: keys.partition.eq(name) for name in config["quality_audit"]["partitions"]}
    fields = pd.read_csv(source, nrows=0).columns.tolist()
    quantiles = config["quality_audit"]["numeric_quantiles"]
    results, focused, dtype_map = {}, {}, {}
    # Equal rows necessarily share these signatures; positives are then verified.
    signatures = {name: np.zeros(len(keys), dtype=np.uint64) for name in ["including_time", "excluding_time"]}
    excluded = {"TransactionID", "isFraud"}
    for start in range(0, len(fields), 32):
        columns = fields[start:start + 32]
        frame = pd.read_csv(source, usecols=columns, low_memory=False)
        dtype_map.update({column: str(frame[column].dtype) for column in columns})
        for kind in signatures:
            included = [c for c in columns if c not in excluded and (kind == "including_time" or c != "TransactionDT")]
            if included:
                part_hash = pd.util.hash_pandas_object(frame[included], index=False).to_numpy(dtype=np.uint64)
                signatures[kind] = signatures[kind] * np.uint64(1000003) ^ part_hash
        for column in columns:
            train = describe(frame.loc[masks["train"], column], quantiles)
            record = {"train": train}
            train_categories = set(frame.loc[masks["train"], column].dropna()) if train["kind"] == "categorical" else None
            for name in ["validation", "test"]:
                part = frame.loc[masks[name], column]
                info = describe(part, quantiles, train)
                if train_categories is not None:
                    observed = part.dropna()
                    info["unseen_nonmissing_count"] = int((~observed.isin(train_categories)).sum())
                    info["unseen_rate_among_nonmissing"] = float((~observed.isin(train_categories)).mean()) if len(observed) else None
                record[name] = info
            results[column] = record
            if column in config["quality_audit"]["focused_features"]:
                by_day = {}
                for day in sorted(keys.loc[masks["test"], "day_id"].unique()):
                    mask = masks["test"] & keys.day_id.eq(day)
                    by_day[str(day)] = describe(frame.loc[mask, column], quantiles, train)
                focus_mask = masks["test"] & keys.day_id.isin(config["quality_audit"]["focused_test_days"])
                focused[column] = {"by_test_day": by_day,
                                   "days_171_172": describe(frame.loc[focus_mask, column], quantiles, train),
                                   "other_test_days": describe(frame.loc[masks["test"] & ~focus_mask, column], quantiles, train)}
        print(f"Audited fields {min(start+32, len(fields))}/{len(fields)}", flush=True)
    duplicates = {}
    for kind, signature in signatures.items():
        subset = [c for c in fields if c not in excluded and (kind == "including_time" or c != "TransactionDT")]
        duplicates[kind] = verify_duplicates(source, keys, signature, dtype_map, subset)
    feature_names = [c for c in fields if c not in {"TransactionID", "TransactionDT", "isFraud"}]
    rankings = {}
    for name in ["validation", "test"]:
        rankings[name] = {
            "largest_absolute_missing_rate_changes": sorted(
                [{"feature": c, "change_pp": results[c][name]["missing_rate_change_pp"]} for c in feature_names],
                key=lambda r: abs(r["change_pp"]), reverse=True)[:15],
            "largest_max_abs_train_observed_z": sorted(
                [{"feature": c, "value": results[c][name]["max_abs_train_observed_z"]} for c in feature_names
                 if results[c][name].get("max_abs_train_observed_z") is not None], key=lambda r: r["value"], reverse=True)[:15]}
    payload = {
        "status": "executed", "analysis_type": config["analysis_type"], "training_executed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": perf_counter() - started,
        "config_sha256": digest(config_path), "code_sha256": digest(Path(__file__)),
        "source_sha256": source_hash, "source_columns": len(fields), "audited_columns": len(results),
        "partition_counts": {name: {"rows": int(mask.sum()), "fraud": int(keys.loc[mask, "isFraud"].sum())} for name, mask in masks.items()},
        "features": results, "focused_features": focused, "rankings": rankings, "duplicates": duplicates,
        "train_all_missing_features": [c for c in feature_names if results[c]["train"]["all_missing"]],
        "train_constant_nonmissing_features": [c for c in feature_names if results[c]["train"]["constant_nonmissing"]],
        "limitations": [
            "Raw numeric codes are described numerically without asserting business meaning",
            "Observed-value z scores exclude missing values; these are not the imputed model's standardized values",
            "Missingness changes and extreme values are descriptive, not evidence of causal drift",
            "Repeated predictor vectors can be legitimate transactions and are not removed",
            "Post-hoc diagnosis does not authorize new feature selection on test",
            "Only the labeled transaction table is audited here; identity remains audit-only in the core study"
        ]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, payload)
    print(f"Wrote {args.out} ({payload['elapsed_seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
