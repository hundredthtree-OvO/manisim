from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TaskSpaceCommand:
    target_position: np.ndarray
    gripper_position: float
    timestamp: float
    source: str
    valid: bool = True
    target_orientation: np.ndarray | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def create(
        cls,
        *,
        target_position: np.ndarray | list[float] | tuple[float, ...],
        gripper_position: float,
        timestamp: float,
        source: str,
        valid: bool = True,
        target_orientation: (
            np.ndarray | list[float] | tuple[float, ...] | None
        ) = None,
        metadata: dict[str, Any] | None = None,
    ) -> "TaskSpaceCommand":
        return cls(
            target_position=np.asarray(
                target_position, dtype=np.float64
            ).copy(),
            gripper_position=float(gripper_position),
            timestamp=float(timestamp),
            source=source,
            valid=valid,
            target_orientation=(
                None
                if target_orientation is None
                else np.asarray(
                    target_orientation, dtype=np.float64
                ).copy()
            ),
            metadata=None if metadata is None else dict(metadata),
        )
