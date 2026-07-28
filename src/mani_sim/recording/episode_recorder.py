from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mani_sim.recording.jsonl_recorder import JsonlRecorder, _json_value


SCHEMA_VERSION = "mani-sim.session.v1"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


class EpisodeRecorder:
    """Write one launch as a session containing reset-bounded episodes."""

    def __init__(
        self,
        root: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
    ):
        started_at = _utc_now()
        base_id = session_id or started_at.strftime("%Y%m%d-%H%M%S-%f")
        self.session_id, self.session_dir = self._create_session_dir(
            Path(root), base_id
        )
        self._episodes_dir = self.session_dir / "episodes"
        self._episodes_dir.mkdir()
        self._manifest = JsonlRecorder(self.session_dir / "episodes.jsonl")
        self._episode_writer: JsonlRecorder | None = None
        self._episode_id = -1
        self._episode_step = 0
        self._episode_started_at: datetime | None = None
        self._global_step = 0
        self._closed = False

        session_metadata = {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "created_at": _iso_utc(started_at),
            **(metadata or {}),
        }
        (self.session_dir / "metadata.json").write_text(
            json.dumps(
                session_metadata,
                default=_json_value,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self.start_episode()

    @staticmethod
    def _create_session_dir(root: Path, base_id: str) -> tuple[str, Path]:
        root.mkdir(parents=True, exist_ok=True)
        for suffix in range(1000):
            candidate_id = base_id if suffix == 0 else f"{base_id}-{suffix:03d}"
            candidate = root / candidate_id
            try:
                candidate.mkdir()
            except FileExistsError:
                continue
            return candidate_id, candidate
        raise RuntimeError(f"could not allocate a session directory under {root}")

    @property
    def episode_id(self) -> int:
        return self._episode_id

    def start_episode(self) -> None:
        if self._closed:
            raise RuntimeError("cannot start an episode on a closed recorder")
        if self._episode_writer is not None:
            raise RuntimeError("the current episode must be ended first")
        self._episode_id += 1
        self._episode_step = 0
        self._episode_started_at = _utc_now()
        self._episode_writer = JsonlRecorder(
            self._episodes_dir / f"episode_{self._episode_id:06d}.jsonl"
        )

    def write(self, record: dict[str, Any]) -> None:
        if self._episode_writer is None:
            raise RuntimeError("no active episode")
        enriched = {
            **record,
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "episode_id": self._episode_id,
            "episode_step": self._episode_step,
            "global_step": self._global_step,
        }
        self._episode_writer.write(enriched)
        self._episode_step += 1
        self._global_step += 1

    def end_episode(
        self,
        reason: str,
        *,
        final_fields: dict[str, Any] | None = None,
    ) -> None:
        if self._episode_writer is None:
            return
        ended_at = _utc_now()
        self._episode_writer.close()
        self._episode_writer = None
        self._manifest.write(
            {
                "schema_version": SCHEMA_VERSION,
                "session_id": self.session_id,
                "episode_id": self._episode_id,
                "file": f"episodes/episode_{self._episode_id:06d}.jsonl",
                "started_at": _iso_utc(self._episode_started_at or ended_at),
                "ended_at": _iso_utc(ended_at),
                "end_reason": reason,
                "step_count": self._episode_step,
                "final": final_fields or {},
            }
        )

    def rotate_episode(
        self,
        reason: str,
        *,
        final_fields: dict[str, Any] | None = None,
    ) -> None:
        self.end_episode(reason, final_fields=final_fields)
        self.start_episode()

    def close(self, reason: str = "session_end") -> None:
        if self._closed:
            return
        self.end_episode(reason)
        self._manifest.close()
        self._closed = True

    def __enter__(self) -> "EpisodeRecorder":
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: object,
    ) -> None:
        self.close("exception" if exception_type is not None else "session_end")
