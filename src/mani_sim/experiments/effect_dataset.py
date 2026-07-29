from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = "mani-sim.effect.v1"


def _vector_delta(
    final: Iterable[float], initial: Iterable[float]
) -> list[float]:
    return (
        np.asarray(final, dtype=np.float64)
        - np.asarray(initial, dtype=np.float64)
    ).tolist()


def _lost_step(branch: dict[str, Any]) -> int | None:
    return next(
        (
            int(step["step"])
            for step in branch["steps"]
            if not step["grasped"]
        ),
        None,
    )


def _labels(
    branch: dict[str, Any],
    *,
    anchor_tcp: list[float],
    anchor_object: list[float],
) -> dict[str, Any]:
    return {
        "maintained_grasp": bool(branch["maintained_grasp"]),
        "grasp_transition": (
            "held" if branch["maintained_grasp"] else "lost"
        ),
        "lost_grasp_step": _lost_step(branch),
        "final_tcp_delta_m": _vector_delta(
            branch["final_tcp_position"], anchor_tcp
        ),
        "final_object_delta_m": _vector_delta(
            branch["final_object_position"], anchor_object
        ),
        "maximum_relative_xy_slip_m": float(
            branch["maximum_relative_xy_slip_m"]
        ),
        "maximum_grip_force_n": float(
            branch["maximum_grip_force_n"]
        ),
        "maximum_object_force_n": float(
            branch["maximum_object_force_n"]
        ),
        "maximum_unintended_force_n": float(
            branch["maximum_unintended_force_n"]
        ),
        "grip_force_impulse_ns": float(
            branch["grip_force_impulse_ns"]
        ),
        "object_force_impulse_ns": float(
            branch["object_force_impulse_ns"]
        ),
    }


def _effect_delta(
    labels: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    scalar_names = (
        "maximum_relative_xy_slip_m",
        "maximum_grip_force_n",
        "maximum_object_force_n",
        "maximum_unintended_force_n",
        "grip_force_impulse_ns",
        "object_force_impulse_ns",
    )
    vector_names = ("final_tcp_delta_m", "final_object_delta_m")
    effect = {
        name: float(labels[name] - baseline[name])
        for name in scalar_names
    }
    effect.update(
        {
            name: _vector_delta(labels[name], baseline[name])
            for name in vector_names
        }
    )
    effect["changed_grasp_outcome"] = (
        labels["maintained_grasp"]
        != baseline["maintained_grasp"]
    )
    return effect


def _metric_ranges(branches: list[dict[str, Any]]) -> dict[str, Any]:
    if len(branches) < 2:
        raise ValueError("noise baseline requires at least two repeats")
    first = branches[0]
    anchor_tcp = first["steps"][0]["tcp_position"]
    anchor_object = first["steps"][0]["object_position"]
    values = [
        _labels(
            branch,
            anchor_tcp=anchor_tcp,
            anchor_object=anchor_object,
        )
        for branch in branches
    ]
    scalar_names = (
        "maximum_relative_xy_slip_m",
        "maximum_grip_force_n",
        "maximum_object_force_n",
        "maximum_unintended_force_n",
        "grip_force_impulse_ns",
        "object_force_impulse_ns",
    )
    result: dict[str, Any] = {
        f"{name}_range": float(
            max(value[name] for value in values)
            - min(value[name] for value in values)
        )
        for name in scalar_names
    }
    for name in ("final_tcp_delta_m", "final_object_delta_m"):
        array = np.asarray([value[name] for value in values])
        result[f"{name}_maximum_pairwise_m"] = float(
            np.max(
                np.linalg.norm(
                    array[:, None, :] - array[None, :, :], axis=-1
                )
            )
        )
    result["grasp_outcome_consistent"] = (
        len({value["maintained_grasp"] for value in values}) == 1
    )
    return result


@dataclass(frozen=True)
class EffectDatasetResult:
    output_dir: Path
    sample_count: int
    matched_group_count: int


def _assign_group_splits(
    records: list[dict[str, Any]],
    ratios: tuple[float, float, float],
    *,
    seed: int,
    unit: str,
) -> dict[str, int]:
    if any(ratio < 0 for ratio in ratios) or not np.isclose(
        sum(ratios), 1.0
    ):
        raise ValueError("split ratios must be non-negative and sum to 1")
    if unit not in {"matched_group", "experiment"}:
        raise ValueError("split unit must be matched_group or experiment")
    key = "matched_group_id" if unit == "matched_group" else "experiment_id"
    groups = sorted({record[key] for record in records})
    if len(groups) < 3:
        raise ValueError("group split requires at least three matched groups")
    random.Random(seed).shuffle(groups)
    train_count = max(1, int(len(groups) * ratios[0]))
    validation_count = max(1, int(len(groups) * ratios[1]))
    if train_count + validation_count >= len(groups):
        train_count = len(groups) - 2
        validation_count = 1
    assignments = {
        group: (
            "train"
            if index < train_count
            else (
                "validation"
                if index < train_count + validation_count
                else "test"
            )
        )
        for index, group in enumerate(groups)
    }
    for record in records:
        record["split"] = assignments[record[key]]
    return {
        split: sum(value == split for value in assignments.values())
        for split in ("train", "validation", "test")
    }


def build_effect_dataset(
    experiment_dir: str | Path | Iterable[str | Path],
    output_dir: str | Path,
    *,
    fidelity_group: str | Path | None = None,
    split_ratios: tuple[float, float, float] | None = None,
    split_seed: int = 0,
    split_unit: str = "matched_group",
) -> EffectDatasetResult:
    sources = (
        [Path(experiment_dir)]
        if isinstance(experiment_dir, (str, Path))
        else [Path(item) for item in experiment_dir]
    )
    if not sources:
        raise ValueError("at least one experiment directory is required")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    group_count = 0
    source_reports: list[str] = []
    fixed_dynamics_values: set[bool] = set()
    for source in sources:
        report_path = source / "anchor_sweep_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        source_reports.append(str(report_path))
        fixed_dynamics_values.add(bool(report["fixed_dynamics"]))
        anchor_metadata = {
            item["anchor_id"]: item for item in report["anchors"]
        }
        for group_path in sorted(source.glob("x*.json")):
            group = json.loads(group_path.read_text(encoding="utf-8"))
            anchor_id = group["checkpoint_id"]
            branches = group["branches"]
            baseline_branch = next(
                (
                    branch
                    for branch in branches
                    if branch["intervention"]["name"] == "base"
                ),
                None,
            )
            if baseline_branch is None:
                raise ValueError(f"{group_path} has no base branch")
            baseline = _labels(
                baseline_branch,
                anchor_tcp=group["anchor_tcp_position"],
                anchor_object=group["anchor_object_position"],
            )
            group_count += 1
            for branch in branches:
                labels = _labels(
                    branch,
                    anchor_tcp=group["anchor_tcp_position"],
                    anchor_object=group["anchor_object_position"],
                )
                intervention = branch["intervention"]
                records.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "sample_id": (
                            f"{report['experiment_id']}/{anchor_id}/"
                            f"{intervention['name']}"
                        ),
                        "matched_group_id": (
                            f"{report['experiment_id']}/{anchor_id}"
                        ),
                        "experiment_id": report["experiment_id"],
                        "simulation_seed": report.get(
                            "simulation_seed"
                        ),
                        "anchor": {
                            "type": group["anchor"],
                            "id": anchor_id,
                            "grasp_offset_xy_m": anchor_metadata[
                                anchor_id
                            ]["grasp_offset_xy_m"],
                            "tcp_position": (
                                group["anchor_tcp_position"]
                            ),
                            "object_position": (
                                group["anchor_object_position"]
                            ),
                        },
                        "intervention": intervention,
                        "labels": labels,
                        "effect_delta_from_base": _effect_delta(
                            labels, baseline
                        ),
                        "source": {
                            "experiment_dir": str(source),
                            "branch_group": group_path.name,
                            "checkpoint_id": group["checkpoint_id"],
                        },
                    }
                )

    if not records:
        raise ValueError("experiment directories contain no branch groups")
    if len({record["sample_id"] for record in records}) != len(records):
        raise ValueError("experiment directories contain duplicate sample IDs")
    split_group_counts = (
        _assign_group_splits(
            records,
            split_ratios,
            seed=split_seed,
            unit=split_unit,
        )
        if split_ratios is not None
        else None
    )
    noise_baseline = None
    fidelity_source = None
    if fidelity_group is not None:
        fidelity_path = Path(fidelity_group)
        fidelity = json.loads(fidelity_path.read_text(encoding="utf-8"))
        noise_baseline = _metric_ranges(fidelity["branches"])
        fidelity_source = str(fidelity_path)

    records_path = destination / "effects.jsonl"
    records_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_experiments": [str(source) for source in sources],
        "source_reports": source_reports,
        "fidelity_source": fidelity_source,
        "fixed_dynamics": fixed_dynamics_values == {True},
        "sample_count": len(records),
        "matched_group_count": group_count,
        "interventions": sorted(
            {record["intervention"]["name"] for record in records}
        ),
        "grasp_transition_counts": {
            outcome: sum(
                record["labels"]["grasp_transition"] == outcome
                for record in records
            )
            for outcome in ("held", "lost")
        },
        "noise_baseline": noise_baseline,
        "split_seed": split_seed if split_ratios is not None else None,
        "split_ratios": split_ratios,
        "split_unit": split_unit if split_ratios is not None else None,
        "split_unit_counts": split_group_counts,
        "split_matched_group_counts": (
            {
                split: len(
                    {
                        record["matched_group_id"]
                        for record in records
                        if record["split"] == split
                    }
                )
                for split in ("train", "validation", "test")
            }
            if split_ratios is not None
            else None
        ),
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return EffectDatasetResult(
        output_dir=destination,
        sample_count=len(records),
        matched_group_count=group_count,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a matched-branch effect dataset."
    )
    parser.add_argument("experiment_dir", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fidelity-group", type=Path)
    parser.add_argument(
        "--split-ratios",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VALIDATION", "TEST"),
    )
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--split-unit",
        choices=("matched_group", "experiment"),
        default="matched_group",
    )
    args = parser.parse_args()
    result = build_effect_dataset(
        args.experiment_dir,
        args.output_dir,
        fidelity_group=args.fidelity_group,
        split_ratios=(
            tuple(args.split_ratios)
            if args.split_ratios is not None
            else None
        ),
        split_seed=args.split_seed,
        split_unit=args.split_unit,
    )
    print(f"dataset_dir={result.output_dir}")
    print(f"samples={result.sample_count}")
    print(f"matched_groups={result.matched_group_count}")
