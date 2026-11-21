"""Package code, reports and aggregate evidence; exclude data, row scores and models."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Submission archive already exists")
    supplement = ROOT / "evidence/supplement_20261121_01"
    manifest = json.loads((supplement / "report_manifest.json").read_text())
    for path, expected in {**manifest["source_sha256"], **manifest["outputs_sha256"]}.items():
        if sha((ROOT / path).read_bytes()) != expected:
            raise ValueError(f"Report evidence changed: {path}")
    if not json.loads((supplement / "verification.json").read_text())["passed"]:
        raise ValueError("Supplement has not passed verification")
    files = set()
    for name in ["README.md", ".gitignore", "pyproject.toml", "requirements-runtime.txt"]:
        files.add(ROOT / name)
    for directory in ["fraudml_v2", "scripts", "tests"]:
        files.update((ROOT / directory).rglob("*.py"))
    files.update((ROOT / "configs").glob("*.json"))
    files.update((ROOT / "docs").glob("*.md"))
    files.update(p for p in (ROOT / "evidence").rglob("*") if p.is_file() and p.suffix in {".json", ".md", ".txt", ".png", ".pdf"})
    files.add(ROOT / "results/README.md")
    for run_id in ["study_20261121_01", "legacy_20261121_01"]:
        folder = ROOT / "results" / run_id
        files.update(p for p in folder.iterdir() if p.is_file() and p.suffix in {".json", ".csv", ".md"})
        if (folder / "figures").exists():
            files.update(p for p in (folder / "figures").iterdir() if p.is_file())
    legacy = ROOT.parent / "FraudML-main"
    source_manifest = json.loads((ROOT / "evidence/source_manifest.json").read_text())
    files.update((ROOT / entry["source"]).resolve() for entry in source_manifest["files"])
    files.update((legacy / "fraudml").rglob("*.py"))
    files.add(legacy / "docs/course_proposal.md")
    contents = {}
    for path in sorted(files):
        name = path.relative_to(ROOT.parent).as_posix()
        if path.suffix in {".joblib", ".parquet"} or ".venv" in path.parts or "predictions" in path.parts:
            raise ValueError(f"Disallowed artifact in bundle: {name}")
        if path.name in {"train_transaction.csv", "test_transaction.csv", "train_identity.csv", "test_identity.csv"}:
            raise ValueError("Raw data entered bundle")
        contents[name] = path.read_bytes()
    note = """# 新版作业提交包

优先阅读FraudML-v2/docs/新版作业报告.md、原方案改进对照.md，以及evidence/supplement_20261121_01/补充分析结果.md。

包内含新版代码、配置、测试、文字报告、汇总结果、PNG/PDF研究图、独立核验与环境记录；并保留原项目必要源码、四份指标及来源报告作为复现对照。原DOCX仅为来源归档，新版正式报告为Markdown。

不含原始CSV、训练模型、逐笔分数、审核名单、虚拟环境或依赖下载缓存。results中的CSV为日/周/模型汇总，不能代替逐笔结果。

安装与测试：进入FraudML-v2，按docs/复现指南.md建立环境，再运行正确性测试和合成流程。完整真实数据重训需自行将有标签交易表与identity表放在相邻ieee-fraud-detection目录；旧结果核验和补充重算还需相应本地产物。

主要环境安装、测试、合成流程已经验证；没有在独立环境再次完成真实数据全量训练。正式版Word/PDF、演示与录制尚未制作，按课程要求整理。

原测试已被查看，所有新增比较为回顾性或事后分析。原3日块主结果保留，新增1/7日块和日期省略不用于选赢家或替换主结果。

bundle_manifest.json列出包内每个文件的SHA-256。此包仅在本地生成，未上传或提交。
"""
    contents["提交包说明.md"] = note.encode()
    index = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
             "scope": "Local coursework bundle without raw data, row-level scores, models or runtime",
             "file_count_excluding_manifest": len(contents),
             "files": {name: {"sha256": sha(data), "bytes": len(data)} for name, data in sorted(contents.items())}}
    contents["bundle_manifest.json"] = (json.dumps(index, ensure_ascii=False, indent=2) + "\n").encode()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(contents.items()):
            archive.writestr(name, data)
    with zipfile.ZipFile(args.out) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive CRC verification failed")
        for name, info in index["files"].items():
            if sha(archive.read(name)) != info["sha256"]:
                raise ValueError("Archived file hash mismatch")
    info = {"archive": args.out.name, "bytes": args.out.stat().st_size,
            "sha256": sha(args.out.read_bytes()), "files": len(contents),
            "crc_and_all_file_hashes_checked": True, "uploaded": False}
    sidecar = args.out.with_suffix(".json")
    if sidecar.exists():
        raise ValueError("Archive sidecar already exists")
    sidecar.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
