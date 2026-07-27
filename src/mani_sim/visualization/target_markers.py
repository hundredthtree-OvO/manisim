from __future__ import annotations

import numpy as np
import sapien
from numpy.typing import ArrayLike

from mani_skill.utils.building import actors


class TargetMarkers:
    def __init__(self, scene: object, initial_position: ArrayLike):
        initial_pose = sapien.Pose(p=np.asarray(initial_position, dtype=float))
        self.raw = actors.build_sphere(
            scene,
            radius=0.018,
            color=[1.0, 0.1, 0.1, 0.8],
            name="mouse-target-raw",
            body_type="kinematic",
            add_collision=False,
            initial_pose=initial_pose,
        )
        self.safe = actors.build_sphere(
            scene,
            radius=0.014,
            color=[0.1, 1.0, 0.1, 0.9],
            name="mouse-target-safe",
            body_type="kinematic",
            add_collision=False,
            initial_pose=initial_pose,
        )

    def update(self, raw_target: ArrayLike, safe_target: ArrayLike) -> None:
        self.raw.set_pose(sapien.Pose(p=np.asarray(raw_target, dtype=float)))
        self.safe.set_pose(sapien.Pose(p=np.asarray(safe_target, dtype=float)))
