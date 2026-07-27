from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class JsonlRecorder:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("w", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self._stream.write(
            json.dumps(record, default=_json_value, separators=(",", ":")) + "\n"
        )
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()

    def __enter__(self) -> "JsonlRecorder":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
