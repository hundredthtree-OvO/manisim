import json
from pathlib import Path

import numpy as np

from mani_sim.recording.episode_recorder import EpisodeRecorder


def test_episode_recorder_preserves_current_jsonl_behavior(
    tmp_path: Path,
) -> None:
    path = tmp_path / "episode.jsonl"
    with EpisodeRecorder(path) as recorder:
        recorder.write({"step": 0, "position": np.array([1.0, 2.0, 3.0])})

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "step": 0,
        "position": [1.0, 2.0, 3.0],
    }
