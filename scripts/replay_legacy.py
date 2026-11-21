"""Retrain exactly three archived candidates using untouched original modules.

Environment and execution differences are recorded; old numbers are never
replaced. All new artifacts live in FraudML-v2/results/<run-id>.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
import importlib.metadata
import json
import platform
import sys
import traceback
import warnings
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "FraudML-main"))
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from fraudml.data import chronological_split, feature_columns
from fraudml.decision import amount_baseline, random_baseline
from fraudml.features import add_past_entity_stats, as_categorical
from fraudml.metrics import evaluate
from fraudml.models.tabular import TabularFraudModel
from run_study import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not args.run_id or not all(value.isalnum() or value in "-_" for value in args.run_id):
        parser.error("Invalid run ID")
    out = ROOT / "results" / args.run_id
    if out.exists():
        parser.error("Output already exists")
    out.mkdir(parents=True)
    (out / "models").mkdir()
    (out / "predictions").mkdir()
    started = perf_counter()
    source_paths = [ROOT.parent / "FraudML-main" / "fraudml" / path for path in ["data.py", "features.py", "metrics.py", "decision.py", "models/tabular.py"]]
    data_path = ROOT.parent / "ieee-fraud-detection/train_transaction.csv"
    run = {"run_id": args.run_id, "kind": "legacy_replay", "status": "running", "completed_fits": [],
        "data_sha256": digest(data_path), "source_sha256": {str(path.relative_to(ROOT.parent)): digest(path) for path in source_paths},
        "replay_code_sha256": digest(Path(__file__)),
        "environment": {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "joblib"]}},
        "execution_differences": ["Current recorded Python/numpy/pandas/scipy environment; original run lockfile is unavailable", "LightGBM n_jobs=2 for bounded resource use; original did not specify thread count", "Single original training snapshot scores both original validation and test"],
        "analysis_type": "retrospective"}
    write_json(out / "run.json", run)
    results, comparisons = {}, []
    try:
        print("Loading original inputs for three legacy fits...", flush=True)
        frame = as_categorical(pd.read_csv(data_path), ["card1", "addr1"])
        with threadpool_limits(limits=2):
            for name, family, history in [("legacy_LR", "logistic", False), ("legacy_LGB", "lightgbm", False), ("legacy_history", "lightgbm", True)]:
                if history:
                    frame = add_past_entity_stats(frame, ["card1", "addr1"])
                split = chronological_split(frame)
                cols = feature_columns(frame)
                kwargs = {"n_jobs": 2, "verbosity": -1} if family == "lightgbm" else {}
                model = TabularFraudModel(family, random_state=42, **kwargs)
                fit_started = perf_counter()
                with (out / "training.log").open("a") as log, redirect_stdout(log), redirect_stderr(log), warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model.fit(split.train[cols], split.train.isFraud)
                diagnostic = {"name": name, "family": family, "history": history,
                    "seconds": perf_counter() - fit_started, "converged": not any(issubclass(w.category, ConvergenceWarning) for w in caught),
                    "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in caught],
                    "parameters": model.pipeline.named_steps["model"].get_params(),
                    "iterations": np.asarray(model.pipeline.named_steps["model"].n_iter_).tolist()}
                model.entity_cols = ("card1", "addr1") if history else ()
                model.categorical_codes = ("card1", "addr1")
                model.save(out / "models" / f"{name}.joblib")
                for period, part in [("validation", split.valid), ("test", split.test)]:
                    with warnings.catch_warnings(record=True):
                        probability = model.predict_proba(part[cols])
                    transformed = model.pipeline.named_steps["preprocess"].transform(part[cols])
                    estimator = model.pipeline.named_steps["model"]
                    with warnings.catch_warnings(record=True):
                        margin = estimator.decision_function(transformed) if family == "logistic" else estimator.predict(transformed, raw_score=True)
                    table = part[["TransactionID", "TransactionDT", "isFraud"]].copy()
                    table["probability"], table["margin"] = probability, margin
                    table.to_parquet(out / "predictions" / f"{period}__{name}.parquet", index=False)
                    computed = {"probability": {str(k): evaluate(part.isFraud, probability, k) for k in [100, 500]},
                        "margin": {str(k): evaluate(part.isFraud, margin, k) for k in [100, 500]},
                        "exact_probability_one": int(np.count_nonzero(probability == 1)),
                        "exact_probability_zero": int(np.count_nonzero(probability == 0)),
                        "converged": diagnostic["converged"]}
                    results.setdefault(period, {})[name] = computed
                    old_name = ("validation_history.json" if history else "validation_base.json") if period == "validation" else ("final_history_metrics.json" if history else "final_metrics.json")
                    old = json.loads((ROOT / "evidence/legacy_metrics" / old_name).read_text())
                    expected = old["results"][family]
                    for k in [100, 500]:
                        archived, replay = expected[f"k={k}"], computed["probability"][str(k)]
                        comparisons.append({"period": period, "candidate": name, "k": k,
                            "archived_hits": archived["hits_at_k"], "replay_hits": replay["hits_at_k"],
                            "archived_ap": archived["pr_auc"], "replay_ap": replay["pr_auc"],
                            "delta_hits": replay["hits_at_k"] - archived["hits_at_k"],
                            "delta_ap": replay["pr_auc"] - archived["pr_auc"],
                            "same_metrics_within_1e-10": replay["hits_at_k"] == archived["hits_at_k"] and abs(replay["pr_auc"] - archived["pr_auc"]) < 1e-10})
                    if name == "legacy_LR":
                        for baseline, score in [("random", random_baseline(len(part), seed=42)), ("amount_descending", amount_baseline(part))]:
                            results[period][baseline] = {str(k): evaluate(part.isFraud, score, k) for k in [100, 500]}
                run["completed_fits"].append(diagnostic)
                write_json(out / "metrics.json", results)
                write_json(out / "run.json", run)
                print(f"{name}: {diagnostic['seconds']:.1f}s, converged={diagnostic['converged']}", flush=True)
                del split, model, transformed
        for path in source_paths:
            if digest(path) != run["source_sha256"][str(path.relative_to(ROOT.parent))]:
                raise ValueError("Original source changed during replay")
        pd.DataFrame(comparisons).to_csv(out / "archived_comparison.csv", index=False)
        write_json(out / "archived_comparison.json", comparisons)
        run["status"], run["elapsed_seconds"] = "executed", perf_counter() - started
        run["all_archived_metrics_match"] = all(row["same_metrics_within_1e-10"] for row in comparisons)
        run["artifact_sha256"] = {str(path.relative_to(out)): digest(path) for path in sorted(out.rglob("*")) if path.is_file() and path.name != "run.json"}
        write_json(out / "run.json", run)
        print(f"Completed legacy replay. Archived metrics match: {run['all_archived_metrics_match']}", flush=True)
    except BaseException as error:
        run.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        write_json(out / "run.json", run)
        raise


if __name__ == "__main__":
    main()
