import json
from pathlib import Path

import numpy as np

from mani_sim.recording.episode_recorder import EpisodeRecorder, SCHEMA_VERSION


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_episode_recorder_creates_session_and_reset_bounded_episodes(
    tmp_path: Path,
) -> None:
    with EpisodeRecorder(
        tmp_path,
        metadata={"config": {"seed": 7}},
        session_id="test-session",
    ) as recorder:
        recorder.write({"step": 0, "position": np.array([1.0, 2.0, 3.0])})
        recorder.write({"step": 1})
        recorder.rotate_episode(
            "manual_reset", final_fields={"task_phase": "grasped"}
        )
        recorder.write({"step": 2})

    session = tmp_path / "test-session"
    metadata = json.loads(
        (session / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["schema_version"] == SCHEMA_VERSION
    assert metadata["config"] == {"seed": 7}

    first = _jsonl(session / "episodes/episode_000000.jsonl")
    second = _jsonl(session / "episodes/episode_000001.jsonl")
    assert [row["episode_step"] for row in first] == [0, 1]
    assert [row["global_step"] for row in first] == [0, 1]
    assert first[0]["position"] == [1.0, 2.0, 3.0]
    assert second[0]["episode_step"] == 0
    assert second[0]["global_step"] == 2

    manifest = _jsonl(session / "episodes.jsonl")
    assert [row["end_reason"] for row in manifest] == [
        "manual_reset",
        "session_end",
    ]
    assert [row["step_count"] for row in manifest] == [2, 1]
    assert manifest[0]["final"] == {"task_phase": "grasped"}


def test_episode_recorder_does_not_overwrite_existing_session(
    tmp_path: Path,
) -> None:
    first = EpisodeRecorder(tmp_path, session_id="same")
    second = EpisodeRecorder(tmp_path, session_id="same")
    first.close()
    second.close()

    assert first.session_id == "same"
    assert second.session_id == "same-001"
