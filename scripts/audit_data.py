"""Audit data scale and planned time windows without fitting a model.

Requires pandas and numpy. Run from FraudML-v2 or supply an explicit --out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def json_native(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value)}")


def summarize(part, parent_daily, plan):
    if part.empty:
        raise ValueError("Planned partition is empty")
    daily = part.groupby("day_id", sort=True).agg(
        rows=("isFraud", "size"), fraud=("isFraud", "sum"),
        first_dt=("TransactionDT", "min"), last_dt=("TransactionDT", "max"),
    )
    daily["parent_rows"] = parent_daily.reindex(daily.index)["rows"]
    daily["split_truncated"] = daily["rows"] < daily["parent_rows"]
    daily["fraud_rate"] = daily["fraud"] / daily["rows"]
    n, fraud = len(part), int(part["isFraud"].sum())
    fixed = {}
    for budget in plan["budget"]["daily_fixed"]:
        capacity = daily["rows"].clip(upper=budget)
        total = int(capacity.sum())
        fixed[str(budget)] = {
            "total_reviews": total,
            "expected_random_hits": float(
                (capacity * daily["fraud"] / daily["rows"]).sum()
            ),
            "days_with_fewer_fraud_than_budget": int((daily["fraud"] < budget).sum()),
            "daily_precision_one_hit_step_percentage_points": 100.0 / budget,
            "pooled_precision_one_hit_step_percentage_points": 100.0 / total,
        }
    rate = plan["budget"]["daily_rate"]
    daily_capacity = daily["rows"].map(lambda count: min(count, math.ceil(rate * count)))
    return {
        "rows": n, "fraud": fraud, "fraud_rate": fraud / n,
        "start_dt": int(part["TransactionDT"].min()),
        "end_dt": int(part["TransactionDT"].max()),
        "first_day_id": int(daily.index.min()), "last_day_id": int(daily.index.max()),
        "observed_days": len(daily),
        "missing_day_ids": sorted(set(range(int(daily.index.min()), int(daily.index.max()) + 1)) - set(daily.index)),
        "daily_rows_min_median_max": [int(daily["rows"].min()), float(daily["rows"].median()), int(daily["rows"].max())],
        "daily_fraud_min_median_max": [int(daily["fraud"].min()), float(daily["fraud"].median()), int(daily["fraud"].max())],
        "split_truncated_day_ids": [int(day) for day in daily.index[daily["split_truncated"]]],
        "boundary_note": "Only truncation relative to the full CSV is identified; missing source collection cannot be inferred.",
        "period_rate_budgets": {
            str(r): {"reviews": min(n, math.ceil(r * n)), "expected_random_hits": min(n, math.ceil(r * n)) * fraud / n}
            for r in plan["budget"]["period_rates"]
        },
        "daily_fixed_budgets": fixed,
        "daily_rate_budget": {
            "rate": rate, "reviews": int(daily_capacity.sum()),
            "matched_global_reviews": int(daily_capacity.sum()),
            "expected_random_hits": float((daily_capacity * daily["fraud"] / daily["rows"]).sum()),
        },
        "product_groups": part.groupby("ProductCD", dropna=False)["isFraud"].agg(["size", "sum"]).reset_index().to_dict("records"),
        "daily": daily.reset_index().to_dict("records"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "evidence/data_audit.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.out.exists() and not args.overwrite:
        parser.error(f"Output exists: {args.out}; use a new path or explicit --overwrite")

    config_path = ROOT / "configs/study_plan.json"
    plan = json.loads(config_path.read_text(encoding="utf-8"))
    tx_path = (ROOT / plan["data"]["transaction"]).resolve()
    identity_path = (ROOT / plan["data"]["identity_audit_only"]).resolve()
    columns = ["TransactionID", "TransactionDT", "isFraud", "TransactionAmt", "ProductCD", "card1", "addr1"]
    raw_columns = pd.read_csv(tx_path, nrows=0).columns.tolist()
    frame = pd.read_csv(tx_path, usecols=columns)
    if frame["TransactionID"].isna().any() or frame["TransactionID"].duplicated().any():
        raise ValueError("Missing or duplicate TransactionID")
    if not frame["isFraud"].isin([0, 1]).all():
        raise ValueError("Labels must be complete and binary")
    if not np.isfinite(frame["TransactionDT"].to_numpy()).all():
        raise ValueError("Timestamps must be complete and finite")
    if not (frame["TransactionDT"] % 1 == 0).all():
        raise ValueError("Expected integer elapsed seconds")
    frame["day_id"] = (frame["TransactionDT"] // plan["data"]["day_seconds"]).astype(int)
    parent_daily = frame.groupby("day_id").agg(rows=("isFraud", "size"))
    original = json.loads((ROOT / plan["later_snapshot"]["bounds_source"]).read_text(encoding="utf-8"))
    periods = {}
    assigned = pd.Series(False, index=frame.index)
    for name, boundary in original["split"]["periods"].items():
        mask = frame["TransactionDT"].between(boundary["start_dt"], boundary["end_dt"])
        if (assigned & mask).any():
            raise ValueError("Overlapping original partitions")
        assigned |= mask
        part = frame.loc[mask]
        if len(part) != boundary["rows"]:
            raise ValueError(f"Original {name} row count differs from archived report")
        periods[name] = summarize(part, parent_daily, plan)
    if not assigned.all():
        raise ValueError("Original partitions do not cover every transaction")

    folds = {}
    previous_end = None
    for fold in plan["development_folds"]:
        train_start, train_end = fold["train_days"]
        eval_start, eval_end = fold["evaluation_days"]
        if not train_start < train_end <= eval_start < eval_end:
            raise ValueError("Invalid temporal order")
        if previous_end is not None and previous_end > eval_start:
            raise ValueError("Development evaluation periods overlap")
        previous_end = eval_end
        train = frame.loc[frame["day_id"].ge(train_start) & frame["day_id"].lt(train_end)]
        evaluation = frame.loc[frame["day_id"].ge(eval_start) & frame["day_id"].lt(eval_end)]
        if train["TransactionDT"].max() >= evaluation["TransactionDT"].min():
            raise ValueError("Development train/evaluation timestamp overlap")
        if evaluation["TransactionDT"].max() > periods["train"]["end_dt"]:
            raise ValueError("Development evaluation extends outside original training portion")
        if evaluation["day_id"].nunique() != eval_end - eval_start:
            raise ValueError("Missing evaluation days")
        if train["isFraud"].nunique() < 2 or evaluation["isFraud"].nunique() < 2:
            raise ValueError("A planned fold lacks either label class")
        folds[fold["id"]] = {
            "train": summarize(train, parent_daily, plan),
            "evaluation": summarize(evaluation, parent_daily, plan),
        }

    identity = pd.read_csv(identity_path, usecols=["TransactionID"])
    if identity["TransactionID"].isna().any() or identity["TransactionID"].duplicated().any():
        raise ValueError("Identity join key is missing or duplicated")
    covered = frame["TransactionID"].isin(identity["TransactionID"])
    identity_groups = frame.assign(has_identity=covered).groupby("has_identity")["isFraud"].agg(["size", "sum", "mean"])
    payload = {
        "status": "executed_data_audit_only", "training_executed": False,
        "protocol_version": plan["protocol_version"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
        "plan_sha256": digest(config_path),
        "inputs": [
            {"path": plan["data"]["transaction"], "sha256": digest(tx_path), "bytes": tx_path.stat().st_size},
            {"path": plan["data"]["identity_audit_only"], "sha256": digest(identity_path), "bytes": identity_path.stat().st_size},
        ],
        "scope": {"raw_column_count": len(raw_columns), "audited_columns": columns, "full_feature_quality_audit": False},
        "overall": summarize(frame, parent_daily, plan),
        "original_periods": periods, "development_folds": folds,
        "identity_audit_only": {
            "identity_rows": len(identity), "matched_rows": int(covered.sum()),
            "coverage": float(covered.mean()),
            "unmatched_identity_ids": int((~identity["TransactionID"].isin(frame["TransactionID"])).sum()),
            "groups": identity_groups.reset_index().to_dict("records"),
        },
        "missingness": {column: int(frame[column].isna().sum()) for column in columns},
        "limitations": [
            "No models fitted and no model effect or statistical power demonstrated.",
            "Observed days are not necessarily independent evaluation units.",
            "Only selected input columns were audited.",
            "Source collection completeness cannot be established from first/last timestamps.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w" if args.overwrite else "x", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2, default=json_native, allow_nan=False)
        output.write("\n")
    print(f"Wrote {args.out}")
    for name, summary in [("all", payload["overall"]), *periods.items()]:
        print(f"{name}: rows={summary['rows']}, fraud={summary['fraud']}, days={summary['observed_days']}, split_truncated_days={summary['split_truncated_day_ids']}")
    for name, summary in folds.items():
        part = summary["evaluation"]
        print(f"{name}: evaluation_rows={part['rows']}, fraud={part['fraud']}, days={part['observed_days']}")
    print("Data scale and window audit only; no model training or improvement claimed.")


if __name__ == "__main__":
    main()
