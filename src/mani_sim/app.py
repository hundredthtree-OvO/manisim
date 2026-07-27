from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

import mani_skill.envs  # noqa: F401 - imports register ManiSkill environments
from mani_skill.utils import sapien_utils
from mani_sim.config import AppConfig, load_config
from mani_sim.control.ee_servo import EEServo, build_normalized_panda_action
from mani_sim.control.workspace_guard import WorkspaceGuard
from mani_sim.input.sapien_pointer import SapienPointer
from mani_sim.input.sapien_pointer import PointerSample
from mani_sim.input.height import update_height
from mani_sim.input.view_selection import ViewSelection
from mani_sim.reachability import ReachabilityMap
from mani_sim.recording.jsonl_recorder import JsonlRecorder
from mani_sim.robot_setup import initialize_panda
from mani_sim.visualization.target_markers import TargetMarkers
from mani_sim.visualization.camera_views import AuxiliaryCameraPanel


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=np.float64)


def _single_env_vector(value: Any) -> np.ndarray:
    array = _to_numpy(value)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    return array


def _controller_delta_limit(env: gym.Env) -> float:
    controller = env.unwrapped.agent.controller.controllers["arm"]
    lower = np.broadcast_to(controller.config.pos_lower, 3)
    upper = np.broadcast_to(controller.config.pos_upper, 3)
    limit = float(min(np.min(np.abs(lower)), np.min(np.abs(upper))))
    if limit <= 0:
        raise RuntimeError("Panda arm controller has an invalid position range")
    return limit


def _validate_runtime(env: gym.Env, config: AppConfig) -> float:
    if config.simulation.control_mode != "pd_ee_delta_pos":
        raise RuntimeError("Demo 0 currently requires pd_ee_delta_pos")
    if env.action_space.shape != (4,):
        raise RuntimeError(
            f"expected Panda arm+gripper action shape (4,), got {env.action_space.shape}"
        )
    if not np.allclose(env.action_space.low, -1.0) or not np.allclose(
        env.action_space.high, 1.0
    ):
        raise RuntimeError(f"expected normalized action space, got {env.action_space}")
    return _controller_delta_limit(env)


def _make_guard(config: AppConfig) -> WorkspaceGuard:
    workspace = config.workspace
    servo = config.servo
    return WorkspaceGuard(
        x_bounds_m=workspace.x_bounds_m,
        y_bounds_m=workspace.y_bounds_m,
        work_height_m=workspace.work_height_m,
        progress_epsilon_m=servo.progress_epsilon_m,
        saturation_steps=servo.saturation_steps,
        saturation_distance_m=servo.saturation_distance_m,
        release_target_delta_m=servo.release_target_delta_m,
        z_bounds_m=workspace.z_bounds_m,
    )


def _configure_camera(viewer: Any, config: AppConfig) -> Any:
    camera = config.camera
    work_height = config.workspace.work_height_m
    eye = [
        camera.center_x_m,
        camera.center_y_m,
        work_height + camera.height_above_work_plane_m,
    ]
    target = [camera.center_x_m, camera.center_y_m, work_height]
    pose = sapien_utils.look_at(eye, target, up=[1, 0, 0]).sp
    viewer.window.set_camera_parameters(
        0.05, 100.0, camera.vertical_fov_rad
    )
    viewer.set_camera_pose(pose)
    return pose


def _add_front_camera(base_env: Any) -> Any:
    pose = sapien_utils.look_at(
        [1.15, -1.15, 0.95], [0.42, 0.0, 0.48], up=[0, 0, 1]
    )
    return base_env.scene.add_camera(
        name="front_observer",
        pose=pose,
        width=320,
        height=240,
        fovy=0.9,
        near=0.05,
        far=10.0,
    )


def run(config: AppConfig, *, max_steps: int | None = None) -> None:
    simulation = config.simulation
    env = gym.make(
        simulation.env_id,
        obs_mode="none",
        reward_mode="none",
        render_mode="human",
        control_mode=simulation.control_mode,
        robot_uids=simulation.robot_uid,
        sim_backend=simulation.sim_backend,
    )
    recorder_context: Any
    if config.recording.enabled:
        recorder_context = JsonlRecorder(config.recording.path)
    else:
        recorder_context = contextlib.nullcontext(None)

    try:
        env.reset(seed=simulation.seed)
        initialize_panda(env.unwrapped)
        _add_front_camera(env.unwrapped)
        controller_limit = _validate_runtime(env, config)
        viewer = env.unwrapped.render_human()
        fixed_camera_pose = _configure_camera(viewer, config)
        env.unwrapped.render_human()
        pointer = SapienPointer(viewer.window, config.workspace.work_height_m)
        camera_panel = AuxiliaryCameraPanel()
        camera_panel.init(viewer)
        viewer.plugins.append(camera_panel)
        reachability = None
        if config.reachability.enabled:
            reachability = ReachabilityMap.load(
                config.reachability.path,
                config.reachability.maximum_height_delta_m,
            )
        servo = EEServo(
            config.servo.gain,
            config.servo.max_delta_m,
            config.servo.deadband_m,
        )
        guard = _make_guard(config)
        tcp = _single_env_vector(env.unwrapped.agent.tcp_pose.p)
        markers = TargetMarkers(env.unwrapped.scene, tcp)
        gripper_target = 1.0
        last_safe_target = tcp.copy()
        previous_recorded_safe_target = tcp.copy()
        target_height = config.workspace.work_height_m
        view_selection = ViewSelection()
        control_dt = 1.0 / float(env.unwrapped.control_freq)

        print(
            "Demo 0 ready | move mouse: TCP target | space: gripper | "
            "U/J: height | 1: top control | 2: front reserve | "
            "3: wrist observe | r: reset | q: quit"
        )
        print(
            f"action_space={env.action_space}, controller_delta_limit="
            f"{controller_limit:.3f} m"
        )
        if reachability is not None:
            print(
                f"reachability_map={config.reachability.path}, "
                f"layers={reachability.heights}"
            )

        with recorder_context as recorder:
            step = 0
            while max_steps is None or step < max_steps:
                env.unwrapped.render_human()
                if viewer.closed:
                    break
                viewer.set_camera_pose(fixed_camera_pose)

                window = viewer.window
                if window.key_press(config.input.quit_key):
                    break
                if window.key_press(config.input.gripper_toggle_key):
                    gripper_target *= -1.0
                requested_view = view_selection.active_view
                if window.key_press(config.input.top_view_key):
                    requested_view = 1
                elif window.key_press(config.input.front_view_key):
                    requested_view = 2
                elif window.key_press(config.input.wrist_view_key):
                    requested_view = 3
                view_selection.switch(
                    requested_view, tuple(window.mouse_position)
                )
                camera_panel.set_active_view(view_selection.active_view)
                if window.key_press(config.input.reset_key):
                    env.reset(seed=simulation.seed)
                    initialize_panda(env.unwrapped)
                    guard.reset()
                    tcp = _single_env_vector(env.unwrapped.agent.tcp_pose.p)
                    last_safe_target = tcp.copy()
                    target_height = config.workspace.work_height_m

                if view_selection.active_view == 1:
                    target_height = update_height(
                        target_height,
                        up_pressed=window.key_down(
                            config.input.vertical_up_key
                        ),
                        down_pressed=window.key_down(
                            config.input.vertical_down_key
                        ),
                        speed_mps=config.input.vertical_speed_mps,
                        dt_seconds=control_dt,
                        bounds_m=config.workspace.z_bounds_m,
                    )
                pointer.work_height_m = target_height

                sample = pointer.sample()
                if camera_panel.pointer_over_panel(*sample.pixel):
                    sample = PointerSample(sample.pixel, None, False)
                if not view_selection.accepts_pointer(sample.pixel):
                    sample = PointerSample(sample.pixel, None, False)
                tcp = _single_env_vector(env.unwrapped.agent.tcp_pose.p)
                if sample.valid and sample.world_target is not None:
                    previous_on_plane = last_safe_target.copy()
                    previous_on_plane[2] = target_height
                    origin_weight = (
                        config.reachability.previous_safe_target_weight
                    )
                    projection_origin = (
                        origin_weight * previous_on_plane
                        + (1.0 - origin_weight) * tcp
                    )
                    reachability_projection = (
                        reachability.project_continuous(
                            projection_origin, sample.world_target
                        )
                        if reachability is not None
                        else None
                    )
                    requested_target = (
                        reachability_projection.target
                        if reachability_projection is not None
                        else sample.world_target
                    )
                    projection_suppressed = False
                    if (
                        reachability is not None
                        and reachability_projection is not None
                        and reachability_projection.projected
                    ):
                        (
                            requested_target,
                            projection_suppressed,
                        ) = reachability.limit_projected_target_step(
                            previous_on_plane,
                            requested_target,
                            config.reachability.maximum_projected_target_step_m,
                        )
                    result = guard.update(requested_target, tcp)
                    safe_target = result.target
                    last_safe_target = safe_target.copy()
                    raw_target = sample.world_target
                else:
                    result = None
                    reachability_projection = None
                    projection_origin = tcp.copy()
                    projection_suppressed = False
                    safe_target = last_safe_target
                    raw_target = last_safe_target

                metric_delta = servo.metric_delta(safe_target, tcp)
                action = build_normalized_panda_action(
                    metric_delta, gripper_target, controller_limit
                )
                _, _, terminated, truncated, _ = env.step(action)
                markers.update(raw_target, safe_target)

                if recorder is not None:
                    recorder.write(
                        {
                            "step": step,
                            "timestamp": time.monotonic(),
                            "mouse_pixel": sample.pixel,
                            "input_valid": sample.valid,
                            "active_view": view_selection.active_view,
                            "target_height_m": target_height,
                            "raw_target_world": raw_target,
                            "safe_target_world": safe_target,
                            "actual_tcp_position": tcp,
                            "action": action,
                            "gripper_target": gripper_target,
                            "saturation_state": (
                                result.reason if result is not None else "input_invalid"
                            ),
                            "reachability_projected": (
                                reachability_projection.projected
                                if reachability_projection is not None
                                else False
                            ),
                            "projection_distance_m": (
                                reachability_projection.projection_distance_m
                                if reachability_projection is not None
                                else 0.0
                            ),
                            "projection_method": (
                                reachability_projection.method
                                if reachability_projection is not None
                                else "none"
                            ),
                            "projection_origin_world": projection_origin,
                            "projection_suppressed": projection_suppressed,
                            "tracking_error_m": float(
                                np.linalg.norm(safe_target - tcp)
                            ),
                            "safe_target_step_m": float(
                                np.linalg.norm(
                                    safe_target - previous_recorded_safe_target
                                )
                            ),
                            "qpos": _single_env_vector(
                                env.unwrapped.agent.robot.get_qpos()
                            ),
                            "qvel": _single_env_vector(
                                env.unwrapped.agent.robot.get_qvel()
                            ),
                        }
                    )
                previous_recorded_safe_target = safe_target.copy()

                step += 1
                if bool(np.asarray(terminated).any()) or bool(
                    np.asarray(truncated).any()
                ):
                    env.reset(seed=simulation.seed)
                    initialize_panda(env.unwrapped)
                    guard.reset()
    finally:
        env.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/demo0.yaml"),
        help="Path to the Demo 0 YAML configuration.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional finite run length, useful for smoke tests.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    run(load_config(args.config), max_steps=args.max_steps)


if __name__ == "__main__":
    main()
