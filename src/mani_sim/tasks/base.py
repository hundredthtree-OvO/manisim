from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class TaskObservation:
    tcp_position: np.ndarray
    object_positions: dict[str, np.ndarray]
    grasped_objects: frozenset[str] = frozenset()


class Task(Protocol):
    def reset(self) -> None: ...

    def update(self, observation: TaskObservation) -> Any: ...

    def record_fields(
        self, state: Any, observation: TaskObservation
    ) -> dict[str, Any]: ...
