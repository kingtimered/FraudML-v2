"""Record isolated-install checks and a synthetic end-to-end run, not a full refit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--smoke-run-id", required=True)
    args = parser.parse_args()
    destination = args.out_dir / "clean_environment.json"
    if destination.exists():
        parser.error("Environment evidence already exists")
    python = args.python.absolute()
    config = python.parent.parent / "pyvenv.cfg"
    if "include-system-site-packages = false" not in config.read_text():
        parser.error("Requires venv without system site packages")
    if (ROOT / "results" / args.smoke_run_id).exists():
        parser.error("Smoke output already exists")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    record = {"status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Isolated dependency installation, correctness tests and synthetic pipeline; no full real-data retraining",
              "code_sha256": digest(Path(__file__)), "pyvenv_cfg": config.read_text(),
              "requirements_sha256": digest(ROOT / "requirements-runtime.txt"), "checks": []}
    operations = [
        ("environment", ["-c", "import json,sys,importlib.metadata as m; print(json.dumps({'python':sys.version,'prefix':sys.prefix,'base_prefix':sys.base_prefix,'path':sys.path,'versions':{d.metadata['Name']:d.version for d in m.distributions()}},indent=2))"]),
        ("pip_check", ["-m", "pip", "check"]),
        ("freeze", ["-m", "pip", "freeze"]),
        ("correctness", ["scripts/python_env.py", "-m", "pytest", "-q"]),
        ("synthetic_fit", ["scripts/python_env.py", "scripts/run_study.py", "--run-id", args.smoke_run_id, "--smoke"]),
        ("synthetic_analysis", ["scripts/python_env.py", "scripts/analyze_results.py", "--run-id", args.smoke_run_id, "--smoke-draws", "30"]),
        ("synthetic_verification", ["scripts/python_env.py", "scripts/verify_study.py", "--run-id", args.smoke_run_id]),
    ]
    for name, arguments in operations:
        print(f"Checking clean environment: {name}", flush=True)
        command = [str(python), *arguments]
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        log = process.stdout + process.stderr
        record["checks"].append({"name": name, "command": command, "exit_code": process.returncode,
                                  "output_sha256": hashlib.sha256(log.encode()).hexdigest(),
                                  "output": log if len(log) < 20000 else log[-20000:]})
        if name == "environment" and process.returncode == 0:
            record["environment"] = json.loads(process.stdout)
            prefix = Path(record["environment"]["prefix"])
            if any("site-packages" in p and not Path(p).is_relative_to(prefix) for p in record["environment"]["path"]):
                raise ValueError("External site-packages found in isolated environment")
        if name == "freeze" and process.returncode == 0:
            (args.out_dir / "requirements-clean-macos-arm64.txt").write_text(process.stdout)
        if process.returncode:
            record["status"] = "failed"
            record["failed_check"] = name
            break
    else:
        smoke = ROOT / "results" / args.smoke_run_id
        run = json.loads((smoke / "run.json").read_text())
        verification = json.loads((smoke / "verification.json").read_text())
        if run["kind"] != "synthetic_smoke" or not verification["passed"]:
            raise ValueError("Synthetic verification failed")
        record["smoke"] = {"run_id": args.smoke_run_id, "kind": run["kind"], "status": run["status"],
                            "verification": verification,
                            "source_sha256": {name: digest(smoke / name) for name in ["run.json", "verification.json", "paired_intervals.json"]}}
        record["status"] = "passed"
    record["elapsed_seconds"] = perf_counter() - started
    destination.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(f"Environment status: {record['status']}; {destination}", flush=True)
    if record["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
