from __future__ import annotations

import json

from mani_sim.experiments.effect_dataset import (
    SCHEMA_VERSION,
    build_effect_dataset,
)


def _branch(name: str, *, slip: float, grasped: bool) -> dict:
    return {
        "intervention": {
            "name": name,
            "dwell_steps": 0,
            "lift_scale": 1.0,
            "xy_offset_m": [0.0, 0.0],
            "gripper_position": -1.0,
        },
        "steps": [
            {
                "step": 0,
                "tcp_position": [0.0, 0.0, 0.1],
                "object_position": [0.0, 0.0, 0.09],
                "grasped": grasped,
            }
        ],
        "maintained_grasp": grasped,
        "final_tcp_position": [0.0, 0.0, 0.2],
        "final_object_position": [0.0, 0.0, 0.19 if grasped else 0.02],
        "maximum_relative_xy_slip_m": slip,
        "maximum_grip_force_n": 10.0,
        "maximum_object_force_n": 1.0,
        "maximum_unintended_force_n": 0.0,
        "grip_force_impulse_ns": 2.0,
        "object_force_impulse_ns": 0.2,
    }


def test_build_effect_dataset_preserves_matched_pairs_and_deltas(
    tmp_path,
) -> None:
    source = tmp_path / "experiment"
    source.mkdir()
    (source / "anchor_sweep_report.json").write_text(
        json.dumps(
            {
                "experiment_id": "exp-1",
                "fixed_dynamics": True,
                "anchors": [
                    {
                        "anchor_id": "x+0_y+0_mm",
                        "grasp_offset_xy_m": [0.0, 0.0],
                    }
                ],
            }
        )
    )
    branches = [
        _branch("base", slip=0.001, grasped=True),
        _branch("release", slip=0.004, grasped=False),
    ]
    group = {
        "checkpoint_id": "x+0_y+0_mm",
        "anchor": "pre_lift",
        "anchor_tcp_position": [0.0, 0.0, 0.1],
        "anchor_object_position": [0.0, 0.0, 0.09],
        "branches": branches,
    }
    (source / "x+0_y+0_mm.json").write_text(json.dumps(group))
    fidelity = tmp_path / "repeats.json"
    fidelity.write_text(json.dumps({"branches": [branches[0], branches[0]]}))

    result = build_effect_dataset(
        source,
        tmp_path / "dataset",
        fidelity_group=fidelity,
    )

    records = [
        json.loads(line)
        for line in (
            result.output_dir / "effects.jsonl"
        ).read_text().splitlines()
    ]
    manifest = json.loads(
        (result.output_dir / "manifest.json").read_text()
    )
    assert result.sample_count == 2
    assert records[0]["schema_version"] == SCHEMA_VERSION
    assert records[0]["matched_group_id"] == records[1][
        "matched_group_id"
    ]
    assert records[1]["labels"]["grasp_transition"] == "lost"
    assert records[1]["labels"]["lost_grasp_step"] == 0
    assert (
        records[1]["effect_delta_from_base"][
            "maximum_relative_xy_slip_m"
        ]
        == 0.003
    )
    assert manifest["grasp_transition_counts"] == {
        "held": 1,
        "lost": 1,
    }
    assert manifest["noise_baseline"]["grasp_outcome_consistent"]


def test_group_split_never_separates_matched_branches(tmp_path) -> None:
    sources = []
    for index in range(3):
        source = tmp_path / f"experiment-{index}"
        source.mkdir()
        experiment_id = f"exp-{index}"
        anchor_id = "x+0_y+0_mm"
        (source / "anchor_sweep_report.json").write_text(
            json.dumps(
                {
                    "experiment_id": experiment_id,
                    "simulation_seed": index,
                    "fixed_dynamics": True,
                    "anchors": [
                        {
                            "anchor_id": anchor_id,
                            "grasp_offset_xy_m": [0.0, 0.0],
                        }
                    ],
                }
            )
        )
        (source / f"{anchor_id}.json").write_text(
            json.dumps(
                {
                    "checkpoint_id": anchor_id,
                    "anchor": "pre_lift",
                    "anchor_tcp_position": [0.0, 0.0, 0.1],
                    "anchor_object_position": [0.0, 0.0, 0.09],
                    "branches": [
                        _branch("base", slip=0.001, grasped=True),
                        _branch("release", slip=0.004, grasped=False),
                    ],
                }
            )
        )
        sources.append(source)

    result = build_effect_dataset(
        sources,
        tmp_path / "split-dataset",
        split_ratios=(0.6, 0.2, 0.2),
        split_seed=7,
    )

    records = [
        json.loads(line)
        for line in (
            result.output_dir / "effects.jsonl"
        ).read_text().splitlines()
    ]
    group_splits = {}
    for record in records:
        group_splits.setdefault(record["matched_group_id"], set()).add(
            record["split"]
        )
    assert all(len(splits) == 1 for splits in group_splits.values())
    assert {next(iter(splits)) for splits in group_splits.values()} == {
        "train",
        "validation",
        "test",
    }


def test_experiment_split_keeps_all_seed_anchors_together(tmp_path) -> None:
    sources = []
    for experiment_index in range(3):
        source = tmp_path / f"scene-{experiment_index}"
        source.mkdir()
        anchors = []
        for anchor_index in range(2):
            anchor_id = f"x+{anchor_index}_y+0_mm"
            anchors.append(
                {
                    "anchor_id": anchor_id,
                    "grasp_offset_xy_m": [anchor_index / 1000, 0.0],
                }
            )
            (source / f"{anchor_id}.json").write_text(
                json.dumps(
                    {
                        "checkpoint_id": anchor_id,
                        "anchor": "pre_lift",
                        "anchor_tcp_position": [0.0, 0.0, 0.1],
                        "anchor_object_position": [0.0, 0.0, 0.09],
                        "branches": [
                            _branch("base", slip=0.001, grasped=True),
                            _branch("release", slip=0.004, grasped=False),
                        ],
                    }
                )
            )
        (source / "anchor_sweep_report.json").write_text(
            json.dumps(
                {
                    "experiment_id": f"scene-{experiment_index}",
                    "simulation_seed": experiment_index,
                    "fixed_dynamics": True,
                    "anchors": anchors,
                }
            )
        )
        sources.append(source)

    result = build_effect_dataset(
        sources,
        tmp_path / "experiment-split",
        split_ratios=(0.6, 0.2, 0.2),
        split_seed=3,
        split_unit="experiment",
    )

    records = [
        json.loads(line)
        for line in (
            result.output_dir / "effects.jsonl"
        ).read_text().splitlines()
    ]
    experiment_splits = {}
    for record in records:
        experiment_splits.setdefault(record["experiment_id"], set()).add(
            record["split"]
        )
    assert all(
        len(splits) == 1 for splits in experiment_splits.values()
    )
    assert len({next(iter(splits)) for splits in experiment_splits.values()}) == 3
