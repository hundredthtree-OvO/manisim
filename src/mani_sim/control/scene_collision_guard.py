from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class AxisAlignedBox:
    center: NDArray[np.float64]
    size: NDArray[np.float64]
    name: str

    @classmethod
    def create(
        cls, center: ArrayLike, size: ArrayLike, name: str
    ) -> "AxisAlignedBox":
        return cls(
            np.asarray(center, dtype=np.float64),
            np.asarray(size, dtype=np.float64),
            name,
        )


@dataclass(frozen=True)
class CollisionGuardResult:
    target: NDArray[np.float64]
    protected: bool
    reason: str


class SceneCollisionGuard:
    """TCP-level geometric guard for the ground, obstacles and grasp object."""

    def __init__(
        self,
        *,
        ground_clearance_m: float,
        obstacle_margin_m: float,
    ):
        self.ground_clearance_m = ground_clearance_m
        self.obstacle_margin_m = obstacle_margin_m

    def protect(
        self,
        requested_target: ArrayLike,
        *,
        obstacles: tuple[AxisAlignedBox, ...] = (),
    ) -> CollisionGuardResult:
        target = np.asarray(requested_target, dtype=np.float64).copy()
        if target[2] < self.ground_clearance_m:
            target[2] = self.ground_clearance_m
            return CollisionGuardResult(target, True, "ground_clearance")

        for obstacle in obstacles:
            half = obstacle.size / 2 + self.obstacle_margin_m
            offset = target - obstacle.center
            if np.all(np.abs(offset) < half):
                target[2] = obstacle.center[2] + half[2]
                return CollisionGuardResult(
                    target, True, f"obstacle_clearance:{obstacle.name}"
                )

        return CollisionGuardResult(target, False, "clear")
