from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mani_sim.config import CubeTaskConfig
from mani_sim.tasks.base import TaskObservation


@dataclass(frozen=True)
class PickPlaceState:
    phase: str
    approached: bool
    grasped: bool
    lifted: bool
    transported: bool
    released: bool
    placed: bool


class PickPlaceTask:
    """Task state and success logic without input or simulator ownership."""

    def __init__(
        self,
        *,
        initial_object_height_m: float,
        approach_clearance_m: float,
        lift_height_m: float,
        goal_position_xy_m: tuple[float, float],
        goal_tolerance_m: float,
        place_height_tolerance_m: float,
        object_name: str = "target",
    ):
        self.initial_object_height_m = initial_object_height_m
        self.approach_clearance_m = approach_clearance_m
        self.lift_height_m = lift_height_m
        self.goal_position_xy_m = np.asarray(
            goal_position_xy_m, dtype=np.float64
        )
        self.goal_tolerance_m = goal_tolerance_m
        self.place_height_tolerance_m = place_height_tolerance_m
        self.object_name = object_name
        self.reset()

    @classmethod
    def from_config(
        cls, config: CubeTaskConfig, initial_object_height_m: float
    ) -> "PickPlaceTask":
        return cls(
            initial_object_height_m=initial_object_height_m,
            approach_clearance_m=config.approach_clearance_m,
            lift_height_m=config.lift_height_m,
            goal_position_xy_m=config.goal_position_xy_m,
            goal_tolerance_m=config.goal_tolerance_m,
            place_height_tolerance_m=config.place_height_tolerance_m,
        )

    def reset(
        self,
        *,
        initial_object_height_m: float | None = None,
        goal_position_xy_m: tuple[float, float] | np.ndarray | None = None,
    ) -> None:
        if initial_object_height_m is not None:
            self.initial_object_height_m = float(initial_object_height_m)
        if goal_position_xy_m is not None:
            self.goal_position_xy_m = np.asarray(
                goal_position_xy_m, dtype=np.float64
            ).copy()
        self.approached = False
        self.ever_grasped = False
        self.lifted = False
        self.transported = False
        self.released = False
        self.placed = False

    def update(self, observation: TaskObservation) -> PickPlaceState:
        tcp = observation.tcp_position
        obj = observation.object_positions[self.object_name]
        is_grasped = self.object_name in observation.grasped_objects
        xy_aligned = np.linalg.norm(tcp[:2] - obj[:2]) <= 0.03
        near_from_above = (
            obj[2] <= tcp[2] <= obj[2] + self.approach_clearance_m
        )
        self.approached = self.approached or bool(xy_aligned and near_from_above)
        self.ever_grasped = self.ever_grasped or is_grasped
        self.lifted = self.lifted or bool(
            self.ever_grasped
            and obj[2] >= self.initial_object_height_m + self.lift_height_m
        )
        at_goal = bool(
            np.linalg.norm(obj[:2] - self.goal_position_xy_m)
            <= self.goal_tolerance_m
        )
        self.transported = self.transported or bool(self.lifted and at_goal)
        self.released = self.released or bool(
            self.transported and not is_grasped
        )
        at_place_height = (
            abs(obj[2] - self.initial_object_height_m)
            <= self.place_height_tolerance_m
        )
        self.placed = self.placed or bool(
            self.released and at_goal and at_place_height
        )
        phase = (
            "placed"
            if self.placed
            else "released"
            if self.released
            else "transported"
            if self.transported
            else "lifted"
            if self.lifted
            else "grasped"
            if is_grasped
            else "approached"
            if self.approached
            else "approaching"
        )
        return PickPlaceState(
            phase,
            self.approached,
            is_grasped,
            self.lifted,
            self.transported,
            self.released,
            self.placed,
        )

    def record_fields(
        self, state: PickPlaceState, observation: TaskObservation
    ) -> dict[str, Any]:
        obj = observation.object_positions[self.object_name]
        return {
            "task_phase": state.phase,
            "task_transported": state.transported,
            "task_released": state.released,
            "task_placed": state.placed,
            "cube_goal_xy_distance_m": float(
                np.linalg.norm(obj[:2] - self.goal_position_xy_m)
            ),
        }

    def ui_fields(
        self, state: PickPlaceState, observation: TaskObservation
    ) -> tuple[tuple[str, str], ...]:
        obj = observation.object_positions[self.object_name]
        goal_distance = float(
            np.linalg.norm(obj[:2] - self.goal_position_xy_m)
        )
        return (
            ("task", "pick_place"),
            ("phase", state.phase),
            ("grasped", "yes" if state.grasped else "no"),
            ("goal distance", f"{goal_distance:.3f} m"),
            ("success", "yes" if state.placed else "no"),
        )
