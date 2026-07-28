from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _force_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "maximum": 0.0, "p95": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "maximum": float(np.max(array)),
        "p95": float(np.percentile(array, 95)),
    }


def summarize_session(session_dir: str | Path) -> dict[str, Any]:
    session = Path(session_dir)
    manifest = _read_jsonl(session / "episodes.jsonl")
    end_reasons = Counter(
        row.get("end_reason", "unknown") for row in manifest
    )
    phase_totals: dict[str, list[float]] = defaultdict(list)
    forces: dict[str, list[float]] = {
        "grip": [],
        "object": [],
        "unintended": [],
    }

    for episode in manifest:
        rows = _read_jsonl(session / episode["file"])
        episode_phases: dict[str, float] = defaultdict(float)
        for index, row in enumerate(rows):
            phase = row.get("policy_phase")
            if phase is not None and index + 1 < len(rows):
                duration = max(
                    0.0,
                    float(rows[index + 1]["timestamp"])
                    - float(row["timestamp"]),
                )
                episode_phases[str(phase)] += duration
            forces["grip"].append(float(row.get("force_grip_n", 0.0)))
            forces["object"].append(
                float(row.get("force_object_net_n", 0.0))
            )
            forces["unintended"].append(
                float(row.get("unintended_contact_force_n", 0.0))
            )
        for phase, duration in episode_phases.items():
            phase_totals[phase].append(duration)

    episode_count = len(manifest)
    success_count = end_reasons["success"]
    return {
        "session_dir": str(session),
        "episode_count": episode_count,
        "success_count": success_count,
        "success_rate": (
            success_count / episode_count if episode_count else 0.0
        ),
        "end_reasons": dict(sorted(end_reasons.items())),
        "mean_episode_steps": (
            float(np.mean([row["step_count"] for row in manifest]))
            if manifest
            else 0.0
        ),
        "phase_mean_duration_s": {
            phase: round(float(np.mean(durations)), 6)
            for phase, durations in sorted(phase_totals.items())
        },
        "force_n": {
            name: _force_summary(values)
            for name, values in forces.items()
        },
    }


def write_session_report(session_dir: str | Path) -> dict[str, Any]:
    session = Path(session_dir)
    report = summarize_session(session)
    (session / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize a recorded mani-sim session."
    )
    parser.add_argument("session_dir", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            write_session_report(args.session_dir),
            ensure_ascii=False,
            indent=2,
        )
    )
