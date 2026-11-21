"""Post-hoc explanation of saved LR test rankings, without any further fitting.

Reconstruct linear-score contributions for the top-ranked transactions. This
describes the fitted score and cannot establish feature availability or causes.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.sparse import issparse

from fraudml_v2.models import FraudModel
from run_study import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--timing-only", action="store_true", help="Only write the saved-score day-concentration sidecar")
    args = parser.parse_args()
    out = ROOT / "results" / args.run_id
    run = json.loads((out / "run.json").read_text())
    if run["kind"] != "full_study" or run["status"] not in {"executed", "verified"}:
        parser.error("Requires a completed real-data run")
    target = out / "lr_score_contributions.json"
    timing_target = out / "lr_day_concentration.json"
    if target.exists() and not args.timing_only:
        parser.error("Diagnostic already exists")
    if timing_target.exists():
        parser.error("Day-concentration diagnostic already exists")
    metrics = json.loads((out / "metrics.json").read_text())
    plan = json.loads((out / "protocol.json").read_text())
    day_records = {}
    for name in ["LR0", "LR1", "LR2"]:
        generation = metrics["test"][name]["generation"]
        path = out / "predictions" / f"test__{name}__{generation}.parquet"
        table = pd.read_parquet(path)
        k = math.ceil(plan["budget"]["primary_rate"] * len(table))
        chosen = table.sort_values(["score", "TransactionDT", "TransactionID"], ascending=[False, True, True], kind="stable").head(k)
        group = chosen.groupby(chosen.TransactionDT // 86400).agg(reviews=("isFraud", "size"), hits=("isFraud", "sum"))
        day_records[name] = {"reviews": len(chosen), "hits": int(chosen.isFraud.sum()),
            "selected_days": len(group), "largest_day_share": float(group.reviews.max() / len(chosen)),
            "day_records": [{"day_id": int(day), "reviews": int(row.reviews), "hits": int(row.hits)} for day, row in group.iterrows()],
            "prediction_sha256": digest(path)}
    write_json(timing_target, {"analysis_type": "post_hoc_descriptive_no_tuning", "code_sha256": digest(Path(__file__)),
        "test_top_primary_budget": day_records, "interpretation": "Selected score extremes may concentrate on a few dates; resampling dates can therefore produce wide fixed-model intervals."})
    if args.timing_only:
        print(f"Wrote {timing_target.name}")
        return
    data_path = ROOT / plan["data"]["transaction"]
    artifacts = {}
    selected, models, predictions = {}, {}, {}
    for name in ["LR0", "LR1", "LR2"]:
        generation = metrics["test"][name]["generation"]
        prediction_path = out / "predictions" / f"test__{name}__{generation}.parquet"
        model_path = out / "models" / f"original_train__{name}__{generation}.joblib"
        table = pd.read_parquet(prediction_path)
        k = math.ceil(plan["budget"]["primary_rate"] * len(table))
        selected[name] = table.sort_values(["score", "TransactionDT", "TransactionID"], ascending=[False, True, True], kind="stable").head(k)
        predictions[name] = table
        models[name] = FraudModel.load(model_path)
        artifacts[str(prediction_path.relative_to(out))] = digest(prediction_path)
        artifacts[str(model_path.relative_to(out))] = digest(model_path)
    ids = set().union(*(set(table.TransactionID) for table in selected.values()))
    # Force a stable categorical schema even when an individual CSV chunk happens
    # to contain integer-looking rather than floating-looking entity values.
    object_columns = [column for name, _, columns in models["LR0"].pipeline.named_steps["preprocess"].transformers_ if name.startswith("cat") for column in columns]
    parts = []
    for chunk in pd.read_csv(data_path, chunksize=20000, dtype={column: "object" for column in object_columns}):
        part = chunk.loc[chunk.TransactionID.isin(ids)]
        if len(part):
            parts.append(part)
    raw = pd.concat(parts).set_index("TransactionID", drop=False)
    output = {}
    for name, model in models.items():
        chosen = selected[name]
        X = model.prepare(raw.loc[chosen.TransactionID, model.columns])
        clipper = model.pipeline.named_steps["clip"]
        preprocessor = model.pipeline.named_steps["preprocess"]
        transformed = preprocessor.transform(clipper.transform(X))
        matrix = transformed.toarray() if issparse(transformed) else np.asarray(transformed)
        estimator = model.pipeline.named_steps["model"]
        contributions = matrix * estimator.coef_[0]
        reconstructed = contributions.sum(axis=1) + float(estimator.intercept_[0])
        np.testing.assert_allclose(reconstructed, chosen.score, rtol=1e-10, atol=1e-8)
        np.testing.assert_allclose(model.predict(X, prepared=True)["score"], chosen.score, rtol=1e-10, atol=1e-8)
        names = preprocessor.get_feature_names_out()
        average = contributions.mean(axis=0)
        order = np.argsort(-np.abs(average), kind="stable")[:12]
        top = []
        for position in order:
            top.append({"feature": names[position], "coefficient": float(estimator.coef_[0, position]),
                "mean_margin_contribution": float(average[position]),
                "median_transformed_value": float(np.median(matrix[:, position])),
                "max_absolute_transformed_value": float(np.max(np.abs(matrix[:, position]))),
                "min_contribution": float(contributions[:, position].min()),
                "max_contribution": float(contributions[:, position].max())})
        original = model.prepare(raw.loc[chosen.TransactionID, model.columns])
        changed = sum(int(((original[column] < lower) | (original[column] > upper)).sum()) for column, (lower, upper) in clipper.bounds_.items())
        output[name] = {"reviews": len(chosen), "hits": int(chosen.isFraud.sum()),
            "converged": metrics["test"][name]["converged"], "intercept": float(estimator.intercept_[0]),
            "max_margin_reconstruction_error": float(np.max(np.abs(reconstructed - chosen.score))),
            "margin_quantiles": np.quantile(chosen.score, [0, .5, 1]).tolist(),
            "top_features_by_absolute_mean_contribution": top,
            "clipped_cells_in_selected_rows": changed}
    write_json(target, {"run_id": args.run_id, "analysis_type": "post_hoc_descriptive_no_tuning",
        "code_sha256": digest(Path(__file__)), "inputs_sha256": artifacts, "test_top_primary_budget": output,
        "limitations": ["Contributions describe the fitted linear score, not causal effects", "Correlated anonymous features limit mechanistic interpretation", "No further models, thresholds or hyperparameters were chosen from this diagnostic"]})
    print(f"Verified LR score decomposition and wrote {target.name}")


if __name__ == "__main__":
    main()
