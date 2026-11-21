"""Render publication-exportable figures from saved summary tables only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from run_study import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    out = ROOT / "results" / args.run_id
    run = json.loads((out / "run.json").read_text())
    if run["kind"] != "full_study" or run["status"] not in {"executed", "verified"}:
        parser.error("Figures require a completed real-data study")
    metrics = json.loads((out / "metrics.json").read_text())
    intervals = json.loads((out / "paired_intervals.json").read_text())
    plots = out / "figures"
    if plots.exists():
        parser.error("Figures already exist")
    plots.mkdir()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .2, "savefig.dpi": 180})
    models = ["LR0", "LR1", "LR2", "LGB-H0", "LGB-H1", "LGB-H2", "LGB-H3"]
    colors = dict(zip(models, ["#91a4ba", "#4c7ca6", "#183d69", "#d39018", "#aa5e3b", "#328874", "#602e86"]))
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4), sharey=True)
    for ax, (period, candidates) in zip(axes, metrics.items()):
        for model in models:
            rows = candidates[model]["period_rates"]
            x = sorted(float(rate) for rate in rows)
            y = [rows[f"{rate:g}"]["precision"] for rate in x]
            suffix = "*" if not candidates[model]["converged"] else ""
            ax.plot(x, y, marker="o", markersize=4, color=colors[model], label=model + suffix)
        ax.set_title(period)
        ax.set_xlabel("Fraction of period reviewed")
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xticks([.001, .005, .01, .02])
        ax.set_ylim(0, 1.03)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel("Precision at review budget")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, frameon=False)
    fig.suptitle("Fixed candidate budget curves | retrospective comparison")
    fig.tight_layout(rect=[0, .1, 1, .95])
    fig.savefig(plots / "budget_precision.png")
    fig.savefig(plots / "budget_precision.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    periods = list(metrics)
    for ax, contrast_name in zip(axes, ["LR2 minus LR0", "LGB-H3 minus LGB-H0"]):
        for index, period in enumerate(periods):
            record = intervals["periods"][period]["contrasts"][contrast_name]
            row = record["period_rate"]["precision"]
            center, low, high = [100 * row[name] for name in ["point_difference", "lower", "upper"]]
            color = "#276a60" if record["comparison_eligible"] else "#999999"
            ax.hlines(index, low, high, color=color, linewidth=2)
            ax.scatter([center], [index], color=color, s=35, zorder=3)
        ax.axvline(0, color="black", linewidth=.8)
        ax.set_title(contrast_name)
        ax.set_yticks(range(len(periods)), periods)
        ax.set_xlabel("Precision difference (percentage points)")
    axes[0].invert_yaxis()
    fig.suptitle("0.5% period budget | paired 3-day-block 95% intervals")
    fig.tight_layout()
    fig.savefig(plots / "primary_intervals.png")
    fig.savefig(plots / "primary_intervals.pdf")
    plt.close(fig)

    weekly = pd.read_csv(out / "weekly_metrics.csv")
    weekly = weekly.loc[weekly.is_final_generation & weekly.scenario.eq("fixed_16")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for ax, period in zip(axes, periods):
        for model in ["LR2", "LGB-H0", "LGB-H3"]:
            group = weekly.loc[weekly.period.eq(period) & weekly.model.eq(model)]
            ax.plot(group.week_index + 1, group.precision, marker="o", color=colors[model], label=model)
            partial = group.loc[group.partial_week]
            ax.scatter(partial.week_index + 1, partial.precision, facecolors="white", edgecolors=colors[model], s=65, zorder=4)
        ax.set_title(period)
        ax.set_xlabel("Week in evaluation period")
        ax.set_ylim(0, 1.03)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel("Daily-16 reviews: weekly micro precision")
    axes[-1].legend(frameon=False)
    fig.suptitle("Daily batch supplement | open markers: partial weeks")
    fig.tight_layout()
    fig.savefig(plots / "weekly_daily16.png")
    fig.savefig(plots / "weekly_daily16.pdf")
    plt.close(fig)
    write_json(plots / "manifest.json", {"run_id": args.run_id, "command": sys.argv,
        "code_sha256": digest(Path(__file__)), "inputs_sha256": {name: digest(out / name) for name in ["metrics.json", "paired_intervals.json", "weekly_metrics.csv"]},
        "outputs_sha256": {path.name: digest(path) for path in sorted(plots.iterdir()) if path.is_file()}})
    print(f"Wrote three PNG/PDF figures: {plots}")


if __name__ == "__main__":
    main()
