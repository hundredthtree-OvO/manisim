from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class ResetTargetState:
    target: NDArray[np.float64]
    target_height_m: float


class ResetManager:
    """Synchronize command state to the real TCP and gate pointer re-entry."""

    def __init__(
        self, pointer_rearm_pixels: float, pointer_settle_steps: int = 0
    ):
        self.pointer_rearm_pixels = pointer_rearm_pixels
        self.pointer_settle_steps = pointer_settle_steps
        self._mouse_anchor = np.zeros(2, dtype=np.float64)
        self._pointer_armed = False
        self._settle_steps_remaining = 0

    @property
    def pointer_armed(self) -> bool:
        return self._pointer_armed

    def reset(
        self, tcp_position: ArrayLike, mouse_pixel: ArrayLike
    ) -> ResetTargetState:
        tcp = np.asarray(tcp_position, dtype=np.float64).copy()
        self._mouse_anchor = np.asarray(mouse_pixel, dtype=np.float64).copy()
        self._pointer_armed = self.pointer_rearm_pixels == 0
        self._settle_steps_remaining = self.pointer_settle_steps
        return ResetTargetState(tcp, float(tcp[2]))

    def accepts_pointer(self, mouse_pixel: ArrayLike) -> bool:
        if self._pointer_armed:
            return True
        pixel = np.asarray(mouse_pixel, dtype=np.float64)
        if self._settle_steps_remaining > 0:
            self._mouse_anchor = pixel.copy()
            self._settle_steps_remaining -= 1
            return False
        if (
            np.linalg.norm(pixel - self._mouse_anchor)
            >= self.pointer_rearm_pixels
        ):
            self._pointer_armed = True
        return self._pointer_armed

    @staticmethod
    def vertical_target(
        previous_target: ArrayLike, target_height_m: float
    ) -> NDArray[np.float64]:
        target = np.asarray(previous_target, dtype=np.float64).copy()
        target[2] = target_height_m
        return target

    @staticmethod
    def axis_target(
        previous_target: ArrayLike, *, axis: int, value: float
    ) -> NDArray[np.float64]:
        target = np.asarray(previous_target, dtype=np.float64).copy()
        target[axis] = value
        return target
