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
    z_bounds_m: tuple[float, float] = (0.05, 0.65)
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
class AppConfig:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    servo: ServoConfig = field(default_factory=ServoConfig)
    input: InputConfig = field(default_factory=InputConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    reachability: ReachabilityConfig = field(default_factory=ReachabilityConfig)


def _pair(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two numbers")
    pair = (float(value[0]), float(value[1]))
    if pair[0] >= pair[1]:
        raise ValueError(f"{name} lower bound must be smaller than upper bound")
    return pair


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
