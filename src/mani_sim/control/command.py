from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TaskSpaceCommand:
    target_position: np.ndarray
    target_orientation: np.ndarray
    gripper_position: float
    timestamp: float
    valid: bool = True
