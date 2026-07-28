from __future__ import annotations

import time

import numpy as np

from mani_sim.control.command import TaskSpaceCommand
from mani_sim.runtime.observation import RuntimeObservation


class ScriptedPickPlaceSource:
    """Deterministic single-environment pick-and-place waypoint policy."""

    def __init__(
        self,
        *,
        approach_clearance_m: float = 0.08,
        lift_height_m: float = 0.10,
        lift_overshoot_m: float = 0.03,
        position_tolerance_m: float = 0.01,
        release_settle_steps: int = 10,
        object_name: str = "target",
        goal_name: str = "goal",
    ):
        self.approach_clearance_m = approach_clearance_m
        self.lift_height_m = lift_height_m
        self.lift_overshoot_m = lift_overshoot_m
        self.position_tolerance_m = position_tolerance_m
        self.release_settle_steps = release_settle_steps
        self.object_name = object_name
        self.goal_name = goal_name
        self.reset()

    def reset(self) -> None:
        self.phase = "approach"
        self._initial_object_height_m: float | None = None
        self._release_steps = 0

    def _reached(
        self, observation: RuntimeObservation, target: np.ndarray
    ) -> bool:
        return bool(
            np.linalg.norm(observation.tcp_position - target)
            <= self.position_tolerance_m
        )

    def act(self, observation: RuntimeObservation) -> TaskSpaceCommand:
        if (
            self.object_name not in observation.object_positions
            or self.goal_name not in observation.object_positions
        ):
            return TaskSpaceCommand.create(
                target_position=observation.tcp_position,
                gripper_position=1.0,
                timestamp=time.monotonic(),
                source="scripted_pick_place",
                valid=False,
                metadata={"policy_phase": "missing_scene_actor"},
            )

        obj = observation.object_positions[self.object_name]
        goal = observation.object_positions[self.goal_name]
        grasped = self.object_name in observation.grasped_objects
        if self._initial_object_height_m is None:
            self._initial_object_height_m = float(obj[2])
        object_height = self._initial_object_height_m
        carry_height = (
            object_height + self.lift_height_m + self.lift_overshoot_m
        )
        approach = np.array(
            [obj[0], obj[1], object_height + self.approach_clearance_m]
        )
        grasp = np.array([obj[0], obj[1], object_height])
        lift = np.array([obj[0], obj[1], carry_height])
        transport = np.array([goal[0], goal[1], carry_height])
        place = np.array([goal[0], goal[1], object_height])
        retreat = np.array(
            [goal[0], goal[1], object_height + self.approach_clearance_m]
        )

        target = approach
        gripper = 1.0
        if self.phase == "approach":
            if self._reached(observation, approach):
                self.phase = "descend"
                target = grasp
        elif self.phase == "descend":
            target = grasp
            if self._reached(observation, grasp):
                self.phase = "close"
                gripper = -1.0
        elif self.phase == "close":
            target = grasp
            gripper = -1.0
            if grasped:
                self.phase = "lift"
                target = lift
        elif self.phase == "lift":
            target = lift
            gripper = -1.0
            if grasped and self._reached(observation, lift):
                self.phase = "transport"
                target = transport
        elif self.phase == "transport":
            target = transport
            gripper = -1.0
            if grasped and self._reached(observation, transport):
                self.phase = "lower"
                target = place
        elif self.phase == "lower":
            target = place
            gripper = -1.0
            if self._reached(observation, place):
                self.phase = "open"
                gripper = 1.0
        elif self.phase == "open":
            target = place
            if not grasped:
                self._release_steps += 1
            if self._release_steps >= self.release_settle_steps:
                self.phase = "retreat"
                target = retreat
        elif self.phase == "retreat":
            target = retreat
            if self._reached(observation, retreat):
                self.phase = "complete"
        elif self.phase == "complete":
            target = retreat
        else:
            raise RuntimeError(f"unknown scripted policy phase: {self.phase}")

        return TaskSpaceCommand.create(
            target_position=target,
            gripper_position=gripper,
            timestamp=time.monotonic(),
            source="scripted_pick_place",
            metadata={"policy_phase": self.phase},
        )
