import json
from pathlib import Path

from mani_sim.recording.session_report import summarize_session


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_session_report_summarizes_outcomes_phases_and_forces(
    tmp_path: Path,
) -> None:
    session = tmp_path / "session"
    episodes = session / "episodes"
    episodes.mkdir(parents=True)
    _write_jsonl(
        session / "episodes.jsonl",
        [
            {
                "episode_id": 0,
                "file": "episodes/episode_000000.jsonl",
                "end_reason": "success",
                "step_count": 3,
            },
            {
                "episode_id": 1,
                "file": "episodes/episode_000001.jsonl",
                "end_reason": "policy_timeout",
                "step_count": 2,
            },
        ],
    )
    _write_jsonl(
        episodes / "episode_000000.jsonl",
        [
            {
                "timestamp": 1.0,
                "policy_phase": "approach",
                "force_grip_n": 0.0,
                "force_object_net_n": 0.6,
                "unintended_contact_force_n": 0.0,
            },
            {
                "timestamp": 1.1,
                "policy_phase": "close",
                "force_grip_n": 20.0,
                "force_object_net_n": 1.0,
                "unintended_contact_force_n": 0.5,
            },
            {
                "timestamp": 1.2,
                "policy_phase": "close",
                "force_grip_n": 25.0,
                "force_object_net_n": 0.8,
                "unintended_contact_force_n": 0.0,
            },
        ],
    )
    _write_jsonl(
        episodes / "episode_000001.jsonl",
        [
            {
                "timestamp": 2.0,
                "policy_phase": "approach",
                "force_grip_n": 0.0,
                "force_object_net_n": 0.6,
                "unintended_contact_force_n": 0.0,
            },
            {
                "timestamp": 2.1,
                "policy_phase": "descend",
                "force_grip_n": 0.0,
                "force_object_net_n": 0.7,
                "unintended_contact_force_n": 0.0,
            },
        ],
    )

    report = summarize_session(session)

    assert report["episode_count"] == 2
    assert report["success_count"] == 1
    assert report["success_rate"] == 0.5
    assert report["end_reasons"] == {"policy_timeout": 1, "success": 1}
    assert report["force_n"]["grip"]["maximum"] == 25.0
    assert report["force_n"]["unintended"]["maximum"] == 0.5
    assert report["phase_mean_duration_s"]["approach"] == 0.1
