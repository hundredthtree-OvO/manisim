from __future__ import annotations

from pathlib import Path
from typing import Any

from mani_sim.recording.jsonl_recorder import JsonlRecorder


class EpisodeRecorder:
    """Runtime-facing recorder boundary backed by the current JSONL writer."""

    def __init__(self, path: str | Path):
        self._writer = JsonlRecorder(path)

    def write(self, record: dict[str, Any]) -> None:
        self._writer.write(record)

    def close(self) -> None:
        self._writer.close()

    def __enter__(self) -> "EpisodeRecorder":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
