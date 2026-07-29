from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def summarize_baselines(
    report_paths: list[str | Path], output_path: str | Path
) -> Path:
    reports = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in report_paths
    ]
    if not reports:
        raise ValueError("at least one baseline report is required")
    intervention_model = "intervention_mean"
    model_names = ("state_action_ridge", "state_knn")
    summary: dict[str, Any] = {
        "report_count": len(reports),
        "resplit_seeds": [report["resplit_seed"] for report in reports],
        "reference_model": intervention_model,
        "models": {},
    }
    for model_name in model_names:
        gains = [
            report["models"][intervention_model]["normalized_rmse"]
            - report["models"][model_name]["normalized_rmse"]
            for report in reports
        ]
        per_target = {}
        for target in reports[0]["targets"]:
            reference = [
                report["models"][intervention_model]["per_target"][
                    target
                ]["rmse"]
                for report in reports
            ]
            candidate = [
                report["models"][model_name]["per_target"][target][
                    "rmse"
                ]
                for report in reports
            ]
            ratios = [
                model / baseline
                for model, baseline in zip(
                    candidate, reference, strict=True
                )
                if baseline > 0
            ]
            per_target[target] = {
                "mean_rmse_ratio_to_intervention_mean": (
                    statistics.mean(ratios)
                ),
                "wins": sum(
                    model < baseline
                    for model, baseline in zip(
                        candidate, reference, strict=True
                    )
                ),
            }
        summary["models"][model_name] = {
            "mean_normalized_rmse_gain": statistics.mean(gains),
            "median_normalized_rmse_gain": statistics.median(gains),
            "wins": sum(gain > 0 for gain in gains),
            "gains": gains,
            "per_target": per_target,
        }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize effect baselines across scene-level splits."
    )
    parser.add_argument("report", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = summarize_baselines(args.report, args.output)
    print(f"summary={output}")
