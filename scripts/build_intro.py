"""Build an offline, self-contained undergraduate project introduction."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]


def main():
    summary_path=ROOT/"evidence/executed_summary.json"
    sensitivity_path=ROOT/"evidence/supplement_20261121_01/sensitivity.json"
    summary=json.loads(summary_path.read_text())
    sensitivity=json.loads(sensitivity_path.read_text())
    omission=next(r for r in sensitivity["periods"]["test"]["omissions"] if r["omitted_days"]==[171,172])
    data={"metrics":summary["metrics"],"omission":omission,
          "periodNames":{"dev_F1":"开发 F1","dev_F2":"开发 F2","validation":"原验证期","test":"原测试期"}}
    template=ROOT/"scripts/project_intro.template.html"
    content=template.read_text()
    if content.count("__STUDY_DATA__")!=1:raise ValueError("Expected one data placeholder")
    content=content.replace("__STUDY_DATA__",json.dumps(data,ensure_ascii=False,separators=(",",":")).replace("</","<\\/"))
    out=ROOT/"presentation"
    out.mkdir(parents=True,exist_ok=True)
    destination=out/"project-intro.html"
    destination.write_text(content)
    manifest={"created_at_utc":datetime.now(timezone.utc).isoformat(),"self_contained":True,"language":"zh-CN",
              "sources_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [summary_path,sensitivity_path,template]},
              "code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "output_sha256":hashlib.sha256(destination.read_bytes()).hexdigest()}
    (out/"intro_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    print(destination)


if __name__=="__main__":main()
