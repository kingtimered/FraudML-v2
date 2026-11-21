"""Compare migrated baseline predictions to current original-code replays."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_study import ROOT, digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--legacy-run-id", required=True)
    args = parser.parse_args()
    out, old = ROOT / "results" / args.run_id, ROOT / "results" / args.legacy_run_id
    target = out / "legacy_alignment.json"
    if target.exists():
        parser.error("Alignment already exists")
    run = json.loads((out / "run.json").read_text())
    replay = json.loads((old / "run.json").read_text())
    if run["status"] != "verified" or replay["status"] != "executed":
        parser.error("Both runs must be completed and study verified")
    if run["data_sha256"] != replay["data_sha256"]:
        parser.error("Different transaction data")
    records = []
    for period in ["validation", "test"]:
        for current, original in [("LR0", "legacy_LR"), ("LGB-H0", "legacy_LGB"), ("LGB-H1", "legacy_history")]:
            paths = [out / "predictions" / f"{period}__{current}__initial.parquet", old / "predictions" / f"{period}__{original}.parquet"]
            new, legacy = [pd.read_parquet(path).sort_values("TransactionID").reset_index(drop=True) for path in paths]
            columns = ["TransactionID", "TransactionDT", "isFraud"]
            if not new[columns].equals(legacy[columns]):
                raise ValueError("Rows differ between migrated and original baseline")
            errors = {}
            for score in ["margin", "probability"]:
                np.testing.assert_allclose(new[score], legacy[score], rtol=1e-10, atol=1e-10)
                errors[score] = float(np.max(np.abs(new[score] - legacy[score])))
            records.append({"period": period, "migrated": current, "original": original, "rows": len(new),
                "passed": True, "max_absolute_errors": errors, "input_sha256": [digest(path) for path in paths]})
    write_json(target, {"passed": True, "scope": "Current environment and current CSV only; not an exact historical environment reproduction", "code_sha256": digest(Path(__file__)), "checks": records})
    print("PASS: all six original/migrated baseline pairs align in margin and probability")


if __name__ == "__main__":
    main()
