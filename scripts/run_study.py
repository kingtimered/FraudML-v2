"""Run fixed plan2 candidates, save predictions, queues and auditable checkpoints.

Use --smoke for a synthetic wiring test; smoke settings are never study results.
Full runs use all configured models and windows. Existing output paths are rejected.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from fraudml_v2.evaluation import evaluate_period
from fraudml_v2.features import canonicalize_codes, history_variants
from fraudml_v2.models import FraudModel


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, payload):
    def native(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value).__name__)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, default=native, allow_nan=False)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(encoded + "\n", encoding="utf-8")
    temporary.replace(path)


def synthetic_data():
    rng = np.random.default_rng(42)
    n = 2800
    times = np.sort(rng.integers(86400, 183 * 86400, n))
    amount = rng.lognormal(4, 1, n)
    card = rng.integers(1, 60, n).astype(float)
    addr = rng.integers(1, 15, n).astype(float)
    card[rng.random(n) < .08] = np.nan
    addr[rng.random(n) < .1] = np.nan
    fraud = rng.random(n) < (0.04 + .3 * (amount > 120))
    return pd.DataFrame({
        "TransactionID": np.arange(1, n + 1), "TransactionDT": times,
        "TransactionAmt": amount, "card1": card, "addr1": addr,
        "C1": rng.poisson(2, n), "V1": rng.normal(size=n),
        "ProductCD": rng.choice(["C", "W"], n), "isFraud": fraud.astype(int),
    })


def build_snapshots(frame, plan, smoke):
    day = frame.TransactionDT // 86400
    snapshots = []
    for fold in plan["development_folds"]:
        a, b = fold["train_days"]
        c, d = fold["evaluation_days"]
        train = np.flatnonzero((day >= a) & (day < b))
        evaluation = np.flatnonzero((day >= c) & (day < d))
        snapshots.append((fold["id"], train, {fold["id"]: evaluation}))
    if smoke:
        a, b = int(.7 * len(frame)), int(.85 * len(frame))
        while a < len(frame) and frame.TransactionDT.iloc[a] == frame.TransactionDT.iloc[a - 1]:
            a += 1
        while b < len(frame) and frame.TransactionDT.iloc[b] == frame.TransactionDT.iloc[b - 1]:
            b += 1
        snapshots.append(("original_train", np.arange(a), {"validation": np.arange(a, b), "test": np.arange(b, len(frame))}))
    else:
        legacy = json.loads((ROOT / plan["later_snapshot"]["bounds_source"]).read_text())
        indices = {}
        for name, period in legacy["split"]["periods"].items():
            selected = np.flatnonzero(frame.TransactionDT.between(period["start_dt"], period["end_dt"]))
            if len(selected) != period["rows"]:
                raise ValueError(f"Archived split mismatch: {name}")
            indices[name] = selected
        snapshots.append(("original_train", indices["train"], {name: indices[name] for name in ["validation", "test"]}))
    for _, train, evaluations in snapshots:
        if len(train) == 0 or frame.isFraud.iloc[train].nunique() < 2:
            raise ValueError("Empty or single-class training split")
        for period, selected in evaluations.items():
            if len(selected) == 0 or frame.TransactionDT.iloc[train].max() >= frame.TransactionDT.iloc[selected].min():
                raise ValueError(f"Empty or temporally overlapping period: {period}")
    return snapshots


def summarize_split(frame, positions):
    part = frame.iloc[positions]
    days = part.TransactionDT // 86400
    full_counts = (frame.TransactionDT // 86400).value_counts()
    counts = days.value_counts().sort_index()
    return {
        "rows": len(part), "fraud": int(part.isFraud.sum()),
        "start_dt": int(part.TransactionDT.min()), "end_dt": int(part.TransactionDT.max()),
        "observed_days": int(days.nunique()),
        "split_truncated_day_ids": [int(day) for day, count in counts.items() if count < full_counts.loc[day]],
        "ids_sha256": hashlib.sha256(part.TransactionID.to_numpy(dtype="int64").tobytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not args.run_id or not all(character.isalnum() or character in "-_" for character in args.run_id):
        parser.error("run-id must contain only letters, numbers, hyphens and underscores")
    out = ROOT / "results" / args.run_id
    if out.exists():
        parser.error("Output already exists; choose another run-id")
    out.mkdir(parents=True)
    started = perf_counter()
    plan_path = ROOT / "configs/study_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    original_plan_hash = digest(plan_path)
    run = {
        "run_id": args.run_id, "status": "running", "kind": "synthetic_smoke" if args.smoke else "full_study",
        "protocol_version": plan["protocol_version"], "protocol_sha256": original_plan_hash,
        "analysis_type": "retrospective", "command": sys.argv,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "source_hashes": {str(path.relative_to(ROOT)): digest(path) for path in sorted([*ROOT.glob("fraudml_v2/*.py"), *ROOT.glob("scripts/*.py")])},
        "completed_fits": [], "fit_failures": [],
    }
    for name in ["numpy", "pandas", "scikit-learn", "scipy", "lightgbm", "joblib", "pyarrow", "threadpoolctl"]:
        run["environment"][name] = importlib.metadata.version(name)
    write_json(out / "run.json", run)
    write_json(out / "protocol.json", plan)
    try:
        if args.smoke:
            frame = synthetic_data()
            # Deliberate speed-only overrides are isolated from study evidence.
            plan["lightgbm_defaults"]["n_estimators"] = 12
            plan["logistic_defaults"]["max_iter"] = 300
            run["smoke_overrides"] = {"lightgbm_n_estimators": 12, "logistic_max_iter": 300}
        else:
            data_path = (ROOT / plan["data"]["transaction"]).resolve()
            print("Loading complete labeled transaction data...", flush=True)
            run["data_sha256"] = digest(data_path)
            frame = pd.read_csv(data_path)
        if frame.TransactionID.isna().any() or frame.TransactionID.duplicated().any():
            raise ValueError("Invalid transaction identifiers")
        if not frame.isFraud.isin([0, 1]).all():
            raise ValueError("Invalid labels")
        frame = frame.sort_values(["TransactionDT", "TransactionID"], kind="stable").reset_index(drop=True)
        frame = canonicalize_codes(frame, plan["preprocessing"]["canonical_integer_codes"])
        snapshots = build_snapshots(frame, plan, args.smoke)
        history_started = perf_counter()
        histories = history_variants(frame, plan["history"]["entities"])
        run["history_seconds"] = perf_counter() - history_started
        base_columns = [column for column in frame if column not in {"TransactionID", "TransactionDT", "isFraud"}]
        split_info = {}
        all_metrics = {}
        daily_parts, weekly_parts, queues = [], [], []
        diagnostics = {}
        probability_diagnostics = {}
        integration_checks = []
        (out / "models").mkdir()
        (out / "predictions").mkdir()

        def record_scores(period, model_id, predictions, metadata, *, generation="initial"):
            selected = period_indices[period]
            part = frame.iloc[selected]
            score = predictions["score"]
            prediction_frame = part[["TransactionID", "TransactionDT", "isFraud"]].copy()
            for key, values in predictions.items():
                prediction_frame[key] = values
            prediction_frame["period"] = period
            prediction_frame["model"] = model_id
            prediction_frame["seed"] = plan["seeds"]["model"]
            prediction_frame["converged"] = metadata.get("converged", True)
            prediction_frame["generation"] = generation
            for entity in plan["history"]["entities"]:
                seen = set(frame.iloc[train_indices][entity].dropna())
                prediction_frame[f"{entity}__train_seen"] = part[entity].isin(seen).to_numpy()
                prediction_frame[f"{entity}__missing"] = part[entity].isna().to_numpy()
            prediction_frame.to_parquet(out / "predictions" / f"{period}__{model_id}__{generation}.parquet", index=False)
            results, daily, weekly, queue = evaluate_period(part, score, plan)
            results["converged"] = metadata.get("converged", True)
            results["generation"] = generation
            if model_id.startswith("LR"):
                p_results, _, _, _ = evaluate_period(part, predictions["probability"], plan, keep_queues=False)
                results["probability_ranking"] = p_results["period_rates"]
            truncated = split_info[snapshot_id]["evaluations"][period]["split_truncated_day_ids"]
            keep = ~((part.TransactionDT // 86400).isin(truncated)).to_numpy()
            if truncated and keep.any():
                boundary_results, boundary_daily, _, _ = evaluate_period(part.loc[keep], score[keep], plan, keep_queues=False)
                results["exclude_split_truncated_days"] = {"excluded_day_ids": truncated, "metrics": boundary_results}
            all_metrics.setdefault(period, {})[model_id] = results
            for table in [daily, weekly, queue]:
                table["period"] = period
                table["model"] = model_id
                table["generation"] = generation
            daily_parts.append(daily)
            weekly_parts.append(weekly)
            queues.append(queue)
            if "probability" in predictions:
                probability_diagnostics.setdefault(period, {})[model_id] = {
                    "exact_probability_one": int(np.count_nonzero(predictions["probability"] == 1.0)),
                    "exact_probability_zero": int(np.count_nonzero(predictions["probability"] == 0.0)),
                    "margin_quantiles": np.quantile(predictions["margin"], [0, .01, .5, .99, 1]).tolist(),
                    "probability_quantiles": np.quantile(predictions["probability"], [0, .01, .5, .99, 1]).tolist(),
                    "primary_top_false_positive_ids": part.iloc[
                        np.lexsort((part.TransactionID, part.TransactionDT, -score))[:int(np.ceil(.005 * len(part)))]]
                        .loc[lambda rows: rows.isFraud.eq(0), "TransactionID"].head(20).tolist(),
                }

        with threadpool_limits(limits=2):
            for snapshot_id, train_indices, period_indices in snapshots:
                split_info[snapshot_id] = {
                    "train": summarize_split(frame, train_indices),
                    "evaluations": {name: summarize_split(frame, selected) for name, selected in period_indices.items()},
                }
                write_json(out / "splits.json", split_info)
                print(f"Snapshot {snapshot_id}: train={len(train_indices):,}", flush=True)
                for period, selected in period_indices.items():
                    part = frame.iloc[selected]
                    random_score = np.empty(len(part))
                    day = part.TransactionDT.to_numpy() // 86400
                    for value in np.unique(day):
                        positions = np.flatnonzero(day == value)
                        rng = np.random.default_rng(np.random.SeedSequence([plan["seeds"]["random_example"], int(value)]))
                        random_score[positions] = rng.random(len(positions))
                    for baseline, score in [("amount", part.TransactionAmt.to_numpy(dtype=float)), ("random_seed42", random_score)]:
                        record_scores(period, baseline, {"score": score}, {})

                def fit_spec(spec, max_iter=None, generation="initial"):
                    model_id = spec["id"]
                    extra = histories.get(model_id)
                    X = frame[base_columns] if extra is None else pd.concat([frame[base_columns], extra], axis=1)
                    model_started = perf_counter()
                    model = FraudModel(spec, plan, max_iter=max_iter).fit(X.iloc[train_indices], frame.isFraud.iloc[train_indices])
                    filename = f"{snapshot_id}__{model_id}__{generation}.joblib"
                    model.save(out / "models" / filename)
                    diagnostic = model.diagnostics()
                    diagnostic["generation"] = generation
                    diagnostic["artifact"] = f"models/{filename}"
                    diagnostics.setdefault(snapshot_id, {}).setdefault(model_id, {})[generation] = diagnostic
                    for period, selected in period_indices.items():
                        prediction_started = perf_counter()
                        predictions = model.predict(X.iloc[selected], prepared=True)
                        diagnostic.setdefault("score_seconds", {})[period] = perf_counter() - prediction_started
                        record_scores(period, model_id, predictions, diagnostic, generation=generation)
                        # Loaded model must give the same score for each isolated row.
                        sample = X.iloc[selected[:3]].copy()
                        if len(sample):
                            sample["card1"] = sample.card1.map(lambda value: None if pd.isna(value) else float(value))
                            loaded = FraudModel.load(out / "models" / filename)
                            batch = loaded.predict(sample)["score"]
                            alone = np.asarray([loaded.predict(sample.iloc[[index]])["score"][0] for index in range(len(sample))])
                            direct = model.predict(X.iloc[selected[:3]], prepared=True)["score"]
                            error = float(max(np.max(np.abs(batch - alone)), np.max(np.abs(batch - direct))))
                            if not np.allclose(batch, alone, rtol=1e-12, atol=1e-12) or not np.allclose(batch, direct, rtol=1e-12, atol=1e-12):
                                raise ValueError(f"Batch/save-load score inconsistency: {model_id}")
                            integration_checks.append({"snapshot": snapshot_id, "period": period, "model": model_id, "generation": generation, "rows": len(sample), "max_absolute_error": error, "passed": True})
                    run["completed_fits"].append({"snapshot": snapshot_id, "model": model_id, "generation": generation, "converged": model.converged, "seconds": perf_counter() - model_started})
                    write_json(out / "diagnostics.json", diagnostics)
                    write_json(out / "metrics.json", all_metrics)
                    write_json(out / "run.json", run)
                    print(f"  {model_id} {generation}: {perf_counter()-model_started:.1f}s, converged={model.converged}", flush=True)
                    return model.converged

                lr_converged = []
                for spec in plan["models"]:
                    converged = fit_spec(spec)
                    if spec["family"] == "logistic":
                        lr_converged.append(converged)
                if not all(lr_converged) and not args.smoke:
                    print("  LR convergence warning: uniformly retrying all three LR candidates at 3000 iterations", flush=True)
                    for spec in plan["models"]:
                        if spec["family"] == "logistic":
                            fit_spec(spec, max_iter=plan["fit_budget"]["logistic_uniform_retry_max_iter"], generation="uniform_retry")

        # Retry artifacts are preserved; default tables contain only final generation.
        for filename, parts in [("daily_metrics.csv", daily_parts), ("weekly_metrics.csv", weekly_parts), ("review_queues.parquet", queues)]:
            table = pd.concat(parts, ignore_index=True)
            final = table.apply(lambda row: row["generation"] == all_metrics[row["period"]][row["model"]]["generation"], axis=1)
            table["is_final_generation"] = final
            if filename.endswith(".parquet"):
                table.to_parquet(out / filename, index=False)
            else:
                table.to_csv(out / filename, index=False)
        write_json(out / "probability_diagnostics.json", probability_diagnostics)
        write_json(out / "integration_checks.json", integration_checks)
        write_json(out / "diagnostics.json", diagnostics)
        write_json(out / "metrics.json", all_metrics)
        rows = []
        rate = f"{plan['budget']['primary_rate']:g}"
        for period, models in all_metrics.items():
            for model_id, metrics in models.items():
                rows.append({"period": period, "model": model_id, "converged": metrics["converged"], "generation": metrics["generation"], **metrics["period_rates"][rate]})
        pd.DataFrame(rows).to_csv(out / "primary_results.csv", index=False)
        run["status"] = "executed_smoke" if args.smoke else "executed"
        run["elapsed_seconds"] = perf_counter() - started
        run["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        run["manifest"] = [str(path.relative_to(out)) for path in sorted(out.rglob("*")) if path.is_file() and path.name != "run.json"]
        write_json(out / "run.json", run)
        print(f"Completed {run['kind']}: {out} ({run['elapsed_seconds']:.1f}s)", flush=True)
    except BaseException as error:
        run["status"] = "failed"
        run["error"] = repr(error)
        run["elapsed_seconds"] = perf_counter() - started
        run["traceback"] = traceback.format_exc()
        write_json(out / "run.json", run)
        raise


if __name__ == "__main__":
    main()
