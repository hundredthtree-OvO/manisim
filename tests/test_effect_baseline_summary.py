from __future__ import annotations

import json

from mani_sim.experiments.effect_baseline_summary import (
    summarize_baselines,
)


def test_summary_counts_cross_split_wins(tmp_path) -> None:
    paths = []
    for index, candidate_rmse in enumerate((0.8, 1.2)):
        report = {
            "resplit_seed": index,
            "targets": ["slip"],
            "models": {
                "intervention_mean": {
                    "normalized_rmse": 1.0,
                    "per_target": {"slip": {"rmse": 1.0}},
                },
                "state_action_ridge": {
                    "normalized_rmse": candidate_rmse,
                    "per_target": {
                        "slip": {"rmse": candidate_rmse}
                    },
                },
                "state_knn": {
                    "normalized_rmse": candidate_rmse,
                    "per_target": {
                        "slip": {"rmse": candidate_rmse}
                    },
                },
            },
        }
        path = tmp_path / f"{index}.json"
        path.write_text(json.dumps(report))
        paths.append(path)

    output = summarize_baselines(paths, tmp_path / "summary.json")
    summary = json.loads(output.read_text())

    assert summary["report_count"] == 2
    assert summary["models"]["state_knn"]["wins"] == 1
    assert (
        summary["models"]["state_knn"]["mean_normalized_rmse_gain"]
        == 0.0
    )
