"""Compatibility adapter for the original PickProgress API."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from mani_sim.tasks.base import TaskObservation
from mani_sim.tasks.pick_place import PickPlaceState, PickPlaceTask


PickProgressState = PickPlaceState


class PickProgress(PickPlaceTask):
    def __init__(
        self,
        *,
        initial_cube_height_m: float,
        approach_clearance_m: float,
        lift_height_m: float,
        goal_position_xy_m: ArrayLike,
        goal_tolerance_m: float,
        place_height_tolerance_m: float,
    ):
        super().__init__(
            initial_object_height_m=initial_cube_height_m,
            approach_clearance_m=approach_clearance_m,
            lift_height_m=lift_height_m,
            goal_position_xy_m=tuple(goal_position_xy_m),
            goal_tolerance_m=goal_tolerance_m,
            place_height_tolerance_m=place_height_tolerance_m,
        )

    def update(
        self,
        tcp_position: ArrayLike,
        cube_position: ArrayLike,
        is_grasped: bool,
    ) -> PickPlaceState:
        return super().update(
            TaskObservation(
                tcp_position=np.asarray(tcp_position, dtype=np.float64),
                object_positions={
                    "target": np.asarray(cube_position, dtype=np.float64)
                },
                grasped_objects=(
                    frozenset({"target"}) if is_grasped else frozenset()
                ),
            )
        )


__all__ = ["PickProgress", "PickProgressState"]
