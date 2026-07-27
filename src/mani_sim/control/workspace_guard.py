from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class GuardResult:
    target: NDArray[np.float64]
    saturated: bool
    reason: str


class WorkspaceGuard:
    """Coarse bounds plus progress-based saturation hysteresis."""

    def __init__(
        self,
        x_bounds_m: tuple[float, float],
        y_bounds_m: tuple[float, float],
        work_height_m: float,
        progress_epsilon_m: float,
        saturation_steps: int,
        saturation_distance_m: float,
        release_target_delta_m: float,
        z_bounds_m: tuple[float, float] | None = None,
    ):
        self.x_bounds_m = x_bounds_m
        self.y_bounds_m = y_bounds_m
        self.work_height_m = work_height_m
        self.z_bounds_m = z_bounds_m or (work_height_m, work_height_m)
        self.progress_epsilon_m = progress_epsilon_m
        self.saturation_steps = saturation_steps
        self.saturation_distance_m = saturation_distance_m
        self.release_target_delta_m = release_target_delta_m
        self._previous_tcp: NDArray[np.float64] | None = None
        self._saturation_target: NDArray[np.float64] | None = None
        self._low_progress_steps = 0
        self._saturated = False

    def reset(self) -> None:
        self._previous_tcp = None
        self._saturation_target = None
        self._low_progress_steps = 0
        self._saturated = False

    def update(self, requested_target: ArrayLike, tcp_position: ArrayLike) -> GuardResult:
        target = np.asarray(requested_target, dtype=np.float64).copy()
        tcp = np.asarray(tcp_position, dtype=np.float64)
        target[0] = np.clip(target[0], *self.x_bounds_m)
        target[1] = np.clip(target[1], *self.y_bounds_m)
        target[2] = np.clip(target[2], *self.z_bounds_m)

        if self._saturated:
            assert self._saturation_target is not None
            target_shift = np.linalg.norm(target - self._saturation_target)
            if target_shift < self.release_target_delta_m:
                self._previous_tcp = tcp.copy()
                return GuardResult(tcp.copy(), True, "ik_stalled")
            self._saturated = False
            self._low_progress_steps = 0
            self._saturation_target = None
            self._previous_tcp = tcp.copy()
            return GuardResult(target, False, "tracking")

        distance = float(np.linalg.norm(target - tcp))
        if self._previous_tcp is not None and distance > self.saturation_distance_m:
            direction = target - self._previous_tcp
            norm = float(np.linalg.norm(direction))
            progress = 0.0
            if norm > 1e-12:
                progress = float(np.dot(tcp - self._previous_tcp, direction / norm))
            if progress < self.progress_epsilon_m:
                self._low_progress_steps += 1
            else:
                self._low_progress_steps = 0
        else:
            self._low_progress_steps = 0

        self._previous_tcp = tcp.copy()
        if self._low_progress_steps >= self.saturation_steps:
            self._saturated = True
            self._saturation_target = target.copy()
            return GuardResult(tcp.copy(), True, "ik_stalled")
        return GuardResult(target, False, "tracking")
