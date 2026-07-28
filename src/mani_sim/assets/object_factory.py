from __future__ import annotations

from typing import Any

import numpy as np
import sapien
from mani_skill.utils.building import actors

from mani_sim.assets.object_spec import BoxObjectSpec


class ObjectFactory:
    """Build scene actors from task-independent asset specifications."""

    def __init__(self, scene: Any):
        self.scene = scene

    def build_box(self, spec: BoxObjectSpec) -> Any:
        size = np.asarray(spec.size_m, dtype=np.float64)
        return actors.build_box(
            self.scene,
            half_sizes=size / 2,
            color=spec.color_rgba,
            name=spec.name,
            body_type=spec.body_type,
            add_collision=spec.add_collision,
            initial_pose=sapien.Pose(p=spec.position_m),
        )
