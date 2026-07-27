from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


class EEServo:
    def __init__(self, gain: float, max_delta_m: float, deadband_m: float):
        if gain <= 0 or max_delta_m <= 0 or deadband_m < 0:
            raise ValueError("invalid servo parameters")
        self.gain = gain
        self.max_delta_m = max_delta_m
        self.deadband_m = deadband_m

    def metric_delta(
        self, target_position: ArrayLike, actual_position: ArrayLike
    ) -> NDArray[np.float32]:
        error = np.asarray(target_position, dtype=np.float64) - np.asarray(
            actual_position, dtype=np.float64
        )
        distance = float(np.linalg.norm(error))
        if distance <= self.deadband_m:
            return np.zeros(3, dtype=np.float32)
        delta = self.gain * error
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > self.max_delta_m:
            delta *= self.max_delta_m / delta_norm
        return delta.astype(np.float32)


def build_normalized_panda_action(
    metric_delta: ArrayLike,
    gripper_target: float,
    controller_delta_limit_m: float,
) -> NDArray[np.float32]:
    """Map metric Panda EE delta and absolute gripper state to normalized action."""

    if controller_delta_limit_m <= 0:
        raise ValueError("controller delta limit must be positive")
    arm = np.asarray(metric_delta, dtype=np.float32)
    if arm.shape != (3,):
        raise ValueError("metric_delta must have shape (3,)")
    normalized_arm = np.clip(
        arm / float(controller_delta_limit_m), -1.0, 1.0
    )
    return np.concatenate(
        [normalized_arm, np.array([np.clip(gripper_target, -1.0, 1.0)])]
    ).astype(np.float32)
