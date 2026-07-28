from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SimulationConfig:
    env_id: str = "Empty-v1"
    sim_backend: str = "cpu"
    control_mode: str = "pd_ee_delta_pos"
    robot_uid: str = "panda"
    seed: int = 0


@dataclass(frozen=True)
class WorkspaceConfig:
    work_height_m: float = 0.45
    z_bounds_m: tuple[float, float] = (0.02, 0.65)
    x_bounds_m: tuple[float, float] = (0.15, 0.75)
    y_bounds_m: tuple[float, float] = (-0.55, 0.55)


@dataclass(frozen=True)
class ServoConfig:
    gain: float = 0.5
    max_delta_m: float = 0.01
    deadband_m: float = 0.002
    progress_epsilon_m: float = 0.0005
    saturation_steps: int = 12
    saturation_distance_m: float = 0.03
    release_target_delta_m: float = 0.04


@dataclass(frozen=True)
class InputConfig:
    gripper_toggle_key: str = "space"
    quit_key: str = "q"
    reset_key: str = "r"
    vertical_up_key: str = "u"
    vertical_down_key: str = "j"
    vertical_speed_mps: float = 0.12
    top_view_key: str = "1"
    front_view_key: str = "2"
    wrist_view_key: str = "3"


@dataclass(frozen=True)
class CameraConfig:
    center_x_m: float = 0.45
    center_y_m: float = 0.0
    height_above_work_plane_m: float = 0.75
    vertical_fov_rad: float = 0.8


@dataclass(frozen=True)
class RecordingConfig:
    enabled: bool = True
    path: str = "runs/demo0.jsonl"


@dataclass(frozen=True)
class ReachabilityConfig:
    enabled: bool = True
    path: str = "calibrations/panda_fixed_orientation.json"
    maximum_height_delta_m: float = 0.001
    previous_safe_target_weight: float = 0.7
    maximum_projected_target_step_m: float = 0.03


@dataclass(frozen=True)
class ResetConfig:
    policy: str = "hold_tcp"
    pointer_rearm_pixels: float = 3.0
    pointer_settle_steps: int = 2


@dataclass(frozen=True)
class CubeTaskConfig:
    enabled: bool = True
    position_xy_m: tuple[float, float] = (0.45, 0.0)
    size_m: float = 0.04
    approach_clearance_m: float = 0.08
    lift_height_m: float = 0.10
    goal_position_xy_m: tuple[float, float] = (0.30, 0.30)
    goal_tolerance_m: float = 0.04
    place_height_tolerance_m: float = 0.015


@dataclass(frozen=True)
class CollisionProtectionConfig:
    enabled: bool = True
    ground_tcp_clearance_m: float = 0.015
    obstacle_margin_m: float = 0.02
    maximum_unintended_contact_force_n: float = 8.0
    obstacle_enabled: bool = False
    obstacle_center_m: tuple[float, float, float] = (0.58, 0.25, 0.05)
    obstacle_size_m: tuple[float, float, float] = (0.10, 0.10, 0.10)


@dataclass(frozen=True)
class AppConfig:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    servo: ServoConfig = field(default_factory=ServoConfig)
    input: InputConfig = field(default_factory=InputConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    reachability: ReachabilityConfig = field(default_factory=ReachabilityConfig)
    reset: ResetConfig = field(default_factory=ResetConfig)
    cube_task: CubeTaskConfig = field(default_factory=CubeTaskConfig)
    collision_protection: CollisionProtectionConfig = field(
        default_factory=CollisionProtectionConfig
    )


def _pair(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two numbers")
    pair = (float(value[0]), float(value[1]))
    if pair[0] >= pair[1]:
        raise ValueError(f"{name} lower bound must be smaller than upper bound")
    return pair


def _vector(value: Any, name: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} numbers")
    return tuple(float(item) for item in value)


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}

    sim = raw.get("simulation", {})
    workspace = raw.get("workspace", {})
    servo = raw.get("servo", {})
    input_config = raw.get("input", {})
    camera = raw.get("camera", {})
    recording = raw.get("recording", {})
    reachability = raw.get("reachability", {})
    reset = raw.get("reset", {})
    cube_task = raw.get("cube_task", {})
    collision = raw.get("collision_protection", {})

    config = AppConfig(
        simulation=SimulationConfig(**sim),
        workspace=WorkspaceConfig(
            work_height_m=float(
                workspace.get("work_height_m", WorkspaceConfig.work_height_m)
            ),
            z_bounds_m=_pair(
                workspace.get("z_bounds_m", WorkspaceConfig.z_bounds_m),
                "workspace.z_bounds_m",
            ),
            x_bounds_m=_pair(
                workspace.get("x_bounds_m", WorkspaceConfig.x_bounds_m),
                "workspace.x_bounds_m",
            ),
            y_bounds_m=_pair(
                workspace.get("y_bounds_m", WorkspaceConfig.y_bounds_m),
                "workspace.y_bounds_m",
            ),
        ),
        servo=ServoConfig(**servo),
        input=InputConfig(**input_config),
        camera=CameraConfig(**camera),
        recording=RecordingConfig(**recording),
        reachability=ReachabilityConfig(**reachability),
        reset=ResetConfig(**reset),
        cube_task=CubeTaskConfig(
            enabled=bool(cube_task.get("enabled", CubeTaskConfig.enabled)),
            position_xy_m=_vector(
                cube_task.get("position_xy_m", CubeTaskConfig.position_xy_m),
                "cube_task.position_xy_m",
                2,
            ),
            size_m=float(cube_task.get("size_m", CubeTaskConfig.size_m)),
            approach_clearance_m=float(
                cube_task.get(
                    "approach_clearance_m", CubeTaskConfig.approach_clearance_m
                )
            ),
            lift_height_m=float(
                cube_task.get("lift_height_m", CubeTaskConfig.lift_height_m)
            ),
            goal_position_xy_m=_vector(
                cube_task.get(
                    "goal_position_xy_m",
                    CubeTaskConfig.goal_position_xy_m,
                ),
                "cube_task.goal_position_xy_m",
                2,
            ),
            goal_tolerance_m=float(
                cube_task.get(
                    "goal_tolerance_m", CubeTaskConfig.goal_tolerance_m
                )
            ),
            place_height_tolerance_m=float(
                cube_task.get(
                    "place_height_tolerance_m",
                    CubeTaskConfig.place_height_tolerance_m,
                )
            ),
        ),
        collision_protection=CollisionProtectionConfig(
            enabled=bool(
                collision.get("enabled", CollisionProtectionConfig.enabled)
            ),
            ground_tcp_clearance_m=float(
                collision.get(
                    "ground_tcp_clearance_m",
                    CollisionProtectionConfig.ground_tcp_clearance_m,
                )
            ),
            obstacle_margin_m=float(
                collision.get(
                    "obstacle_margin_m",
                    CollisionProtectionConfig.obstacle_margin_m,
                )
            ),
            maximum_unintended_contact_force_n=float(
                collision.get(
                    "maximum_unintended_contact_force_n",
                    CollisionProtectionConfig.maximum_unintended_contact_force_n,
                )
            ),
            obstacle_enabled=bool(
                collision.get(
                    "obstacle_enabled",
                    CollisionProtectionConfig.obstacle_enabled,
                )
            ),
            obstacle_center_m=_vector(
                collision.get(
                    "obstacle_center_m",
                    CollisionProtectionConfig.obstacle_center_m,
                ),
                "collision_protection.obstacle_center_m",
                3,
            ),
            obstacle_size_m=_vector(
                collision.get(
                    "obstacle_size_m",
                    CollisionProtectionConfig.obstacle_size_m,
                ),
                "collision_protection.obstacle_size_m",
                3,
            ),
        ),
    )
    _validate(config)
    return config


def _validate(config: AppConfig) -> None:
    if config.servo.gain <= 0:
        raise ValueError("servo.gain must be positive")
    if config.servo.max_delta_m <= 0:
        raise ValueError("servo.max_delta_m must be positive")
    if config.servo.deadband_m < 0:
        raise ValueError("servo.deadband_m must not be negative")
    if config.servo.saturation_steps < 1:
        raise ValueError("servo.saturation_steps must be at least 1")
    if config.reachability.maximum_height_delta_m < 0:
        raise ValueError("reachability.maximum_height_delta_m must not be negative")
    if not 0 <= config.reachability.previous_safe_target_weight <= 1:
        raise ValueError(
            "reachability.previous_safe_target_weight must be within [0, 1]"
        )
    if config.reachability.maximum_projected_target_step_m <= 0:
        raise ValueError(
            "reachability.maximum_projected_target_step_m must be positive"
        )
    if config.input.vertical_speed_mps <= 0:
        raise ValueError("input.vertical_speed_mps must be positive")
    if config.reset.policy != "hold_tcp":
        raise ValueError("reset.policy currently only supports hold_tcp")
    if config.reset.pointer_rearm_pixels < 0:
        raise ValueError("reset.pointer_rearm_pixels must not be negative")
    if config.reset.pointer_settle_steps < 0:
        raise ValueError("reset.pointer_settle_steps must not be negative")
    if not (
        config.workspace.z_bounds_m[0]
        <= config.workspace.work_height_m
        <= config.workspace.z_bounds_m[1]
    ):
        raise ValueError("workspace.work_height_m must lie within z_bounds_m")
    if config.camera.height_above_work_plane_m <= 0:
        raise ValueError("camera.height_above_work_plane_m must be positive")
    if not 0 < config.camera.vertical_fov_rad < 3.14:
        raise ValueError("camera.vertical_fov_rad must be between 0 and pi")
    if config.cube_task.size_m <= 0:
        raise ValueError("cube_task.size_m must be positive")
    if config.cube_task.approach_clearance_m <= 0:
        raise ValueError("cube_task.approach_clearance_m must be positive")
    if config.cube_task.lift_height_m <= 0:
        raise ValueError("cube_task.lift_height_m must be positive")
    if config.cube_task.goal_tolerance_m <= 0:
        raise ValueError("cube_task.goal_tolerance_m must be positive")
    if config.cube_task.place_height_tolerance_m <= 0:
        raise ValueError(
            "cube_task.place_height_tolerance_m must be positive"
        )
    collision = config.collision_protection
    if collision.ground_tcp_clearance_m < 0:
        raise ValueError(
            "collision_protection.ground_tcp_clearance_m must not be negative"
        )
    if collision.obstacle_margin_m < 0:
        raise ValueError(
            "collision_protection.obstacle_margin_m must not be negative"
        )
    if collision.maximum_unintended_contact_force_n <= 0:
        raise ValueError(
            "collision_protection.maximum_unintended_contact_force_n "
            "must be positive"
        )
    if any(size <= 0 for size in collision.obstacle_size_m):
        raise ValueError(
            "collision_protection.obstacle_size_m values must be positive"
        )
