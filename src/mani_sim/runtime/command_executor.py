from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mani_sim.control.command import TaskSpaceCommand
from mani_sim.control.ee_servo import (
    EEServo,
    build_normalized_panda_action,
)
from mani_sim.control.scene_collision_guard import (
    AxisAlignedBox,
    CollisionGuardResult,
    SceneCollisionGuard,
)
from mani_sim.control.workspace_guard import GuardResult, WorkspaceGuard
from mani_sim.reachability import ReachabilityMap
from mani_sim.runtime.observation import RuntimeObservation


@dataclass(frozen=True)
class CommandExecution:
    command: TaskSpaceCommand
    raw_target: np.ndarray
    safe_target: np.ndarray
    action: np.ndarray
    guard_result: GuardResult | None
    collision_result: CollisionGuardResult | None
    reachability_projection: Any | None
    projection_origin: np.ndarray
    projection_suppressed: bool


class CommandExecutor:
    """Apply the shared safety and controller chain to canonical commands."""

    def __init__(
        self,
        *,
        servo: EEServo,
        workspace_guard: WorkspaceGuard,
        scene_guard: SceneCollisionGuard,
        reachability: ReachabilityMap | None,
        controller_delta_limit_m: float,
        previous_safe_target_weight: float,
        maximum_projected_target_step_m: float,
        collision_protection_enabled: bool,
    ):
        self.servo = servo
        self.workspace_guard = workspace_guard
        self.scene_guard = scene_guard
        self.reachability = reachability
        self.controller_delta_limit_m = controller_delta_limit_m
        self.previous_safe_target_weight = previous_safe_target_weight
        self.maximum_projected_target_step_m = (
            maximum_projected_target_step_m
        )
        self.collision_protection_enabled = collision_protection_enabled
        self.last_safe_target = np.zeros(3, dtype=np.float64)

    def reset(self, tcp_position: np.ndarray) -> None:
        self.workspace_guard.reset()
        self.last_safe_target = np.asarray(
            tcp_position, dtype=np.float64
        ).copy()

    def emergency_stop(self, tcp_position: np.ndarray) -> None:
        self.reset(tcp_position)

    def prepare(
        self,
        command: TaskSpaceCommand,
        observation: RuntimeObservation,
        *,
        obstacles: tuple[AxisAlignedBox, ...] = (),
    ) -> CommandExecution:
        tcp = observation.tcp_position
        if not command.valid:
            safe_target = self.last_safe_target.copy()
            metric_delta = self.servo.metric_delta(safe_target, tcp)
            action = build_normalized_panda_action(
                metric_delta,
                command.gripper_position,
                self.controller_delta_limit_m,
            )
            return CommandExecution(
                command=command,
                raw_target=safe_target.copy(),
                safe_target=safe_target,
                action=action,
                guard_result=None,
                collision_result=None,
                reachability_projection=None,
                projection_origin=tcp.copy(),
                projection_suppressed=False,
            )

        raw_target = command.target_position.copy()
        bounded_candidate = self.workspace_guard.clip_target(raw_target)
        previous_on_plane = self.last_safe_target.copy()
        previous_on_plane[2] = bounded_candidate[2]
        projection_origin = (
            self.previous_safe_target_weight * previous_on_plane
            + (1.0 - self.previous_safe_target_weight) * tcp
        )
        projection = (
            self.reachability.project_continuous(
                projection_origin, bounded_candidate
            )
            if self.reachability is not None
            else None
        )
        requested_target = (
            projection.target if projection is not None else bounded_candidate
        )
        projection_suppressed = False
        if (
            self.reachability is not None
            and projection is not None
            and projection.projected
        ):
            (
                requested_target,
                projection_suppressed,
            ) = self.reachability.limit_projected_target_step(
                previous_on_plane,
                requested_target,
                self.maximum_projected_target_step_m,
            )

        collision_result = None
        if self.collision_protection_enabled:
            collision_result = self.scene_guard.protect(
                requested_target, obstacles=obstacles
            )
            requested_target = collision_result.target

        guard_result = self.workspace_guard.update(requested_target, tcp)
        safe_target = guard_result.target
        self.last_safe_target = safe_target.copy()
        metric_delta = self.servo.metric_delta(safe_target, tcp)
        action = build_normalized_panda_action(
            metric_delta,
            command.gripper_position,
            self.controller_delta_limit_m,
        )
        return CommandExecution(
            command=command,
            raw_target=raw_target,
            safe_target=safe_target,
            action=action,
            guard_result=guard_result,
            collision_result=collision_result,
            reachability_projection=projection,
            projection_origin=projection_origin,
            projection_suppressed=projection_suppressed,
        )
