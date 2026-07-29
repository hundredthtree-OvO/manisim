from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

import mani_skill.envs  # noqa: F401 - imports register ManiSkill environments
from mani_skill.utils import sapien_utils
from mani_sim.action_sources.mouse import MouseActionSource
from mani_sim.action_sources.scripted_pick_place import (
    ScriptedPickPlaceSource,
)
from mani_sim.config import AppConfig, load_config
from mani_sim.control.command import TaskSpaceCommand
from mani_sim.control.ee_servo import EEServo
from mani_sim.control.workspace_guard import WorkspaceGuard
from mani_sim.control.scene_collision_guard import SceneCollisionGuard
from mani_sim.input.sapien_pointer import SapienPointer
from mani_sim.input.sapien_pointer import PointerSample
from mani_sim.input.height import update_height
from mani_sim.input.view_selection import ViewSelection
from mani_sim.reachability import ReachabilityMap
from mani_sim.recording.episode_recorder import EpisodeRecorder
from mani_sim.recording.session_report import write_session_report
from mani_sim.robot_setup import initialize_panda
from mani_sim.environments.scenario import Scenario, build_scenario
from mani_sim.runtime.command_executor import CommandExecutor
from mani_sim.runtime.reset_manager import ResetManager
from mani_sim.runtime.observation import (
    RuntimeObservation,
    capture_runtime_observation,
    single_env_vector,
)
from mani_sim.tasks.pick_place import PickPlaceTask
from mani_sim.visualization.target_markers import TargetMarkers
from mani_sim.visualization.camera_views import AuxiliaryCameraPanel
from mani_sim.visualization.status_panel import (
    RuntimeStatus,
    RuntimeStatusPanel,
)
from mani_sim.visualization.force_monitor import (
    ForceDisplaySample,
    ForceMonitorPanel,
)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


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


def _top_camera_pose(config: AppConfig) -> Any:
    camera = config.camera
    work_height = config.workspace.work_height_m
    eye = [
        camera.center_x_m,
        camera.center_y_m,
        work_height + camera.height_above_work_plane_m,
    ]
    target = [camera.center_x_m, camera.center_y_m, work_height]
    return sapien_utils.look_at(eye, target, up=[1, 0, 0]).sp


def _front_camera_pose() -> Any:
    return sapien_utils.look_at(
        [0.45, -1.20, 0.35], [0.45, 0.0, 0.35], up=[0, 0, 1]
    ).sp


def _set_main_camera(
    viewer: Any, config: AppConfig, view: int, top_pose: Any, front_pose: Any
) -> None:
    if view == 1:
        viewer.window.set_camera_parameters(
            0.05, 100.0, config.camera.vertical_fov_rad
        )
        viewer.set_camera_pose(top_pose)
    elif view == 2:
        viewer.window.set_camera_parameters(0.05, 100.0, 0.9)
        viewer.set_camera_pose(front_pose)


def _configure_camera(viewer: Any, config: AppConfig) -> Any:
    pose = _top_camera_pose(config)
    viewer.window.set_camera_parameters(
        0.05, 100.0, config.camera.vertical_fov_rad
    )
    viewer.set_camera_pose(pose)
    return pose


def _add_front_camera(base_env: Any) -> Any:
    return base_env.scene.add_camera(
        name="front_observer",
        pose=_front_camera_pose(),
        width=320,
        height=240,
        fovy=0.9,
        near=0.05,
        far=10.0,
    )


def _add_top_camera(base_env: Any, config: AppConfig) -> Any:
    return base_env.scene.add_camera(
        name="top_observer",
        pose=_top_camera_pose(config),
        width=320,
        height=240,
        fovy=config.camera.vertical_fov_rad,
        near=0.05,
        far=10.0,
    )


@dataclass(frozen=True)
class EpisodeReset:
    observation: RuntimeObservation
    target_height_m: float
    target_depth_y_m: float
    gripper_target: float
    previous_safe_target: np.ndarray


def _reset_episode(
    env: gym.Env,
    scenario: Scenario,
    executor: CommandExecutor,
    task: Any,
    reset_manager: ResetManager,
    *,
    seed: int,
    pointer_position: Any,
) -> EpisodeReset:
    env.reset(seed=seed)
    initialize_panda(env.unwrapped)
    scenario.reset(np.random.default_rng(seed))
    if task is not None:
        target_position = scenario.initial_position("target")
        goal_position = scenario.initial_position("goal")
        task.reset(
            initial_object_height_m=(
                None
                if target_position is None
                else float(target_position[2])
            ),
            goal_position_xy_m=(
                None if goal_position is None else goal_position[:2]
            ),
        )
    tcp = single_env_vector(env.unwrapped.agent.tcp_pose.p)
    reset_state = reset_manager.reset(tcp, pointer_position)
    executor.reset(reset_state.target)
    return EpisodeReset(
        observation=capture_runtime_observation(
            env.unwrapped, scenario
        ),
        target_height_m=reset_state.target_height_m,
        target_depth_y_m=float(reset_state.target[1]),
        gripper_target=1.0,
        previous_safe_target=reset_state.target.copy(),
    )


def run(
    config: AppConfig,
    *,
    max_steps: int | None = None,
    max_episodes: int | None = None,
) -> None:
    if max_episodes is not None:
        if max_episodes < 1:
            raise ValueError("max_episodes must be at least 1")
        if config.collection.source != "scripted_pick_place":
            raise ValueError(
                "max_episodes requires collection.source=scripted_pick_place"
            )
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
        recorder_context = EpisodeRecorder(
            config.recording.path,
            metadata={
                "config": asdict(config),
                "runtime": {
                    "python": platform.python_version(),
                    "mani_skill": _package_version("mani-skill"),
                    "sapien": _package_version("sapien"),
                    "torch": _package_version("torch"),
                },
                "frame_semantics": (
                    "command/action are computed from the pre-step TCP; "
                    "object, task, qpos and qvel fields are sampled post-step"
                ),
            },
        )
    else:
        recorder_context = contextlib.nullcontext(None)

    try:
        env.reset(seed=simulation.seed)
        initialize_panda(env.unwrapped)
        scenario = build_scenario(env.unwrapped, config)
        scenario.reset(np.random.default_rng(simulation.seed))
        _add_top_camera(env.unwrapped, config)
        _add_front_camera(env.unwrapped)
        controller_limit = _validate_runtime(env, config)
        viewer = env.unwrapped.render_human()
        fixed_camera_pose = _configure_camera(viewer, config)
        front_camera_pose = _front_camera_pose()
        env.unwrapped.render_human()
        pointer = SapienPointer(viewer.window, config.workspace.work_height_m)
        camera_panel = AuxiliaryCameraPanel()
        camera_panel.init(viewer)
        viewer.plugins.append(camera_panel)
        status_panel = RuntimeStatusPanel()
        status_panel.init(viewer)
        viewer.plugins.append(status_panel)
        force_panel = ForceMonitorPanel(
            env.unwrapped,
            history_capacity=int(env.unwrapped.control_freq * 5),
        )
        force_panel.init(viewer)
        viewer.plugins.append(force_panel)
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
        scene_guard = SceneCollisionGuard(
            ground_clearance_m=(
                config.collision_protection.ground_tcp_clearance_m
            ),
            obstacle_margin_m=config.collision_protection.obstacle_margin_m,
        )
        executor = CommandExecutor(
            servo=servo,
            workspace_guard=guard,
            scene_guard=scene_guard,
            reachability=reachability,
            controller_delta_limit_m=controller_limit,
            previous_safe_target_weight=(
                config.reachability.previous_safe_target_weight
            ),
            maximum_projected_target_step_m=(
                config.reachability.maximum_projected_target_step_m
            ),
            collision_protection_enabled=(
                config.collision_protection.enabled
            ),
        )
        mouse_source = MouseActionSource()
        tcp = single_env_vector(env.unwrapped.agent.tcp_pose.p)
        executor.reset(tcp)
        markers = TargetMarkers(env.unwrapped.scene, tcp)
        reset_manager = ResetManager(
            config.reset.pointer_rearm_pixels,
            config.reset.pointer_settle_steps,
        )
        reset_state = reset_manager.reset(tcp, viewer.window.mouse_position)
        gripper_target = 1.0
        executor.last_safe_target = reset_state.target.copy()
        previous_recorded_safe_target = reset_state.target.copy()
        target_height = reset_state.target_height_m
        target_depth_y = float(reset_state.target[1])
        view_selection = ViewSelection()
        control_dt = 1.0 / float(env.unwrapped.control_freq)
        task = None
        if scenario.cube_initial_position is not None:
            task = PickPlaceTask.from_config(
                config.cube_task,
                float(scenario.cube_initial_position[2]),
            )
            goal_position = scenario.goal_position
            task.reset(
                goal_position_xy_m=(
                    None if goal_position is None else goal_position[:2]
                )
            )
        scripted_source = ScriptedPickPlaceSource(
            approach_clearance_m=config.cube_task.approach_clearance_m,
            lift_height_m=config.cube_task.lift_height_m,
        )
        action_source = (
            mouse_source
            if config.collection.source == "mouse"
            else scripted_source
        )
        observation = capture_runtime_observation(env.unwrapped, scenario)

        print(
            f"Demo 0 ready | source={config.collection.source} | "
            "1: top XY | 2: front XZ | 3: wrist observe | "
            "r: reset | q: quit"
        )
        if config.collection.source == "mouse":
            print("move mouse: TCP target | space: gripper | U/J: height/depth")
        else:
            print(
                "scripted policy is driving the robot; camera keys remain active"
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
        if scenario.cube is not None:
            print(
                f"pick_cube_xy={config.cube_task.position_xy_m}, "
                f"size={config.cube_task.size_m:.3f} m | "
                f"place_goal_xy={config.cube_task.goal_position_xy_m} | "
                "approach, close, lift, move, lower, open"
            )
            if config.cube_task.randomize_positions:
                print(
                    "position_randomization=enabled | "
                    f"target_x={config.cube_task.target_x_bounds_m}, "
                    f"target_y={config.cube_task.target_y_bounds_m}, "
                    f"goal_x={config.cube_task.goal_x_bounds_m}, "
                    f"goal_y={config.cube_task.goal_y_bounds_m}"
                )

        with recorder_context as recorder:
            if recorder is not None:
                print(f"recording_session={recorder.session_dir}")
            step = 0
            episode_step = 0
            completed_episodes = 0
            episode_seed = simulation.seed
            next_episode_seed = simulation.seed + 1
            success_settle_steps = 0
            last_task_record: dict[str, Any] = {"task_phase": "not_started"}
            run_end_reason = "max_steps" if max_steps is not None else "session_end"
            while max_steps is None or step < max_steps:
                env.unwrapped.render_human()
                if viewer.closed:
                    run_end_reason = "window_closed"
                    break
                _set_main_camera(
                    viewer,
                    config,
                    view_selection.active_view,
                    fixed_camera_pose,
                    front_camera_pose,
                )

                window = viewer.window
                if window.key_press(config.input.quit_key):
                    run_end_reason = "quit"
                    break
                if (
                    config.collection.source == "mouse"
                    and window.key_press(config.input.gripper_toggle_key)
                ):
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
                _set_main_camera(
                    viewer,
                    config,
                    view_selection.active_view,
                    fixed_camera_pose,
                    front_camera_pose,
                )
                camera_panel.set_active_view(view_selection.active_view)
                if window.key_press(config.input.reset_key):
                    if recorder is not None:
                        recorder.rotate_episode(
                            "manual_reset", final_fields=last_task_record
                        )
                    reset = _reset_episode(
                        env,
                        scenario,
                        executor,
                        task,
                        reset_manager,
                        seed=next_episode_seed,
                        pointer_position=window.mouse_position,
                    )
                    episode_seed = next_episode_seed
                    next_episode_seed += 1
                    observation = reset.observation
                    previous_recorded_safe_target = (
                        reset.previous_safe_target
                    )
                    target_height = reset.target_height_m
                    target_depth_y = reset.target_depth_y_m
                    gripper_target = reset.gripper_target
                    scripted_source.reset()
                    episode_step = 0
                    success_settle_steps = 0

                up_pressed = (
                    config.collection.source == "mouse"
                    and window.key_down(config.input.vertical_up_key)
                )
                down_pressed = (
                    config.collection.source == "mouse"
                    and window.key_down(config.input.vertical_down_key)
                )
                vertical_active = up_pressed != down_pressed
                if view_selection.active_view == 1:
                    target_height = update_height(
                        target_height,
                        up_pressed=up_pressed,
                        down_pressed=down_pressed,
                        speed_mps=config.input.vertical_speed_mps,
                        dt_seconds=control_dt,
                        bounds_m=config.workspace.z_bounds_m,
                    )
                elif view_selection.active_view == 2:
                    target_depth_y = update_height(
                        target_depth_y,
                        up_pressed=up_pressed,
                        down_pressed=down_pressed,
                        speed_mps=config.input.vertical_speed_mps,
                        dt_seconds=control_dt,
                        bounds_m=config.workspace.y_bounds_m,
                    )
                pointer.work_height_m = target_height

                if view_selection.active_view == 2:
                    sample = pointer.sample_axis_plane(
                        plane_axis=1,
                        plane_value=target_depth_y,
                    )
                else:
                    sample = pointer.sample()
                if camera_panel.pointer_over_panel(*sample.pixel):
                    sample = PointerSample(sample.pixel, None, False)
                if status_panel.pointer_over_panel(*sample.pixel):
                    sample = PointerSample(sample.pixel, None, False)
                if force_panel.pointer_over_panel(*sample.pixel):
                    sample = PointerSample(sample.pixel, None, False)
                if not view_selection.accepts_pointer(sample.pixel):
                    sample = PointerSample(sample.pixel, None, False)
                if sample.valid and not reset_manager.accepts_pointer(
                    sample.pixel
                ):
                    sample = PointerSample(sample.pixel, None, False)
                tcp = observation.tcp_position
                command_candidate = (
                    sample.world_target
                    if sample.valid and sample.world_target is not None
                    else ResetManager.vertical_target(
                        executor.last_safe_target, target_height
                    )
                    if vertical_active and view_selection.active_view == 1
                    else ResetManager.axis_target(
                        executor.last_safe_target,
                        axis=1,
                        value=target_depth_y,
                    )
                    if vertical_active and view_selection.active_view == 2
                    else None
                )
                if config.collection.source == "mouse":
                    mouse_source.update(
                        TaskSpaceCommand.create(
                            target_position=(
                                command_candidate
                                if command_candidate is not None
                                else executor.last_safe_target
                            ),
                            gripper_position=gripper_target,
                            timestamp=time.monotonic(),
                            source="human",
                            valid=command_candidate is not None,
                        )
                    )
                command = action_source.act(observation)
                gripper_target = command.gripper_position
                execution = executor.prepare(
                    command,
                    observation,
                    obstacles=scenario.obstacles,
                )
                raw_target = execution.raw_target
                safe_target = execution.safe_target
                action = execution.action
                result = execution.guard_result
                collision_result = execution.collision_result
                reachability_projection = execution.reachability_projection
                projection_origin = execution.projection_origin
                projection_suppressed = execution.projection_suppressed
                target_height = float(safe_target[2])
                target_depth_y = float(safe_target[1])
                _, _, terminated, truncated, _ = env.step(action)
                markers.update(raw_target, safe_target)
                next_observation = capture_runtime_observation(
                    env.unwrapped, scenario
                )
                force_sample = next_observation.contact_forces
                contact_force_n = force_sample.maximum_unintended_n
                contact_emergency_stop = (
                    contact_force_n
                    > config.collision_protection.maximum_unintended_contact_force_n
                )
                if contact_emergency_stop:
                    tcp = next_observation.tcp_position
                    executor.emergency_stop(tcp)

                cube_position = next_observation.object_positions.get(
                    "target"
                )
                is_grasped = (
                    "target" in next_observation.grasped_objects
                )
                task_observation = (
                    next_observation.task_observation(
                        tcp_position=tcp
                    )
                    if cube_position is not None
                    else None
                )
                progress_state = (
                    task.update(task_observation)
                    if task is not None and task_observation is not None
                    else None
                )
                task_record = (
                    task.record_fields(progress_state, task_observation)
                    if task is not None
                    and progress_state is not None
                    and task_observation is not None
                    else {
                        "task_phase": "disabled",
                        "task_transported": False,
                        "task_released": False,
                        "task_placed": False,
                        "cube_goal_xy_distance_m": None,
                    }
                )
                last_task_record = task_record
                task_fields = (
                    task.ui_fields(progress_state, task_observation)
                    if task is not None
                    and progress_state is not None
                    and task_observation is not None
                    else ()
                )
                task_fields = task_fields + (
                    ("source", command.source),
                    (
                        "policy phase",
                        str(
                            (command.metadata or {}).get(
                                "policy_phase", "manual"
                            )
                        ),
                    ),
                )
                status_panel.update(
                    RuntimeStatus.create(
                        active_view=view_selection.active_view,
                        tcp_position=next_observation.tcp_position,
                        contact_force_n=contact_force_n,
                        contact_threshold_n=(
                            config.collision_protection
                            .maximum_unintended_contact_force_n
                        ),
                        emergency_stop=contact_emergency_stop,
                        recording=config.recording.enabled,
                        episode_seed=episode_seed,
                        randomized_scene=(
                            config.cube_task.randomize_positions
                        ),
                        grip_force_n=force_sample.grip_n,
                        left_finger_force_n=force_sample.left_finger_n,
                        right_finger_force_n=force_sample.right_finger_n,
                        object_force_n=force_sample.object_net_n,
                        task_fields=task_fields,
                    )
                )
                force_panel.update(
                    ForceDisplaySample(
                        grip_n=force_sample.grip_n,
                        object_n=force_sample.object_net_n,
                        unintended_n=force_sample.maximum_unintended_n,
                        left_finger_n=force_sample.left_finger_n,
                        right_finger_n=force_sample.right_finger_n,
                        threshold_n=(
                            config.collision_protection
                            .maximum_unintended_contact_force_n
                        ),
                    )
                )

                if recorder is not None:
                    recorder.write(
                        {
                            "step": step,
                            "episode_seed": episode_seed,
                            "timestamp": time.monotonic(),
                            "mouse_pixel": sample.pixel,
                            "input_valid": sample.valid,
                            "active_view": view_selection.active_view,
                            "target_height_m": target_height,
                            "target_depth_y_m": target_depth_y,
                            "raw_target_world": raw_target,
                            "safe_target_world": safe_target,
                            "actual_tcp_position": tcp,
                            "action": action,
                            "gripper_target": gripper_target,
                            "action_source": command.source,
                            "policy_phase": (
                                (command.metadata or {}).get("policy_phase")
                            ),
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
                            "collision_protected": (
                                collision_result.protected
                                if collision_result is not None
                                else False
                            ),
                            "collision_reason": (
                                collision_result.reason
                                if collision_result is not None
                                else "none"
                            ),
                            "unintended_contact_force_n": contact_force_n,
                            "force_left_finger_world_n": (
                                force_sample.left_finger_world_n
                            ),
                            "force_right_finger_world_n": (
                                force_sample.right_finger_world_n
                            ),
                            "force_grip_n": force_sample.grip_n,
                            "force_object_net_world_n": (
                                force_sample.object_net_world_n
                            ),
                            "force_object_net_n": force_sample.object_net_n,
                            "force_unintended_by_pair_world_n": (
                                force_sample.unintended_by_pair_world_n
                            ),
                            "contact_emergency_stop": contact_emergency_stop,
                            "cube_position": cube_position,
                            "cube_initial_position": (
                                scenario.initial_position("target")
                            ),
                            "is_grasped": is_grasped,
                            **task_record,
                            "goal_position": scenario.goal_position,
                            "tracking_error_m": float(
                                np.linalg.norm(safe_target - tcp)
                            ),
                            "safe_target_step_m": float(
                                np.linalg.norm(
                                    safe_target - previous_recorded_safe_target
                                )
                            ),
                            "qpos": next_observation.qpos,
                            "qvel": next_observation.qvel,
                        }
                    )
                previous_recorded_safe_target = safe_target.copy()
                observation = next_observation

                step += 1
                episode_step += 1
                if task_record.get("task_placed", False):
                    success_settle_steps += 1
                else:
                    success_settle_steps = 0
                automatic_reset_reason = None
                if config.collection.source == "scripted_pick_place":
                    if (
                        success_settle_steps
                        >= config.collection.success_settle_steps
                    ):
                        automatic_reset_reason = "success"
                    elif (
                        episode_step
                        >= config.collection.max_episode_steps
                    ):
                        automatic_reset_reason = "policy_timeout"
                terminated_now = bool(np.asarray(terminated).any())
                truncated_now = bool(np.asarray(truncated).any())
                if (
                    automatic_reset_reason is not None
                    or terminated_now
                    or truncated_now
                ):
                    episode_end_reason = automatic_reset_reason or (
                        "terminated" if terminated_now else "truncated"
                    )
                    if automatic_reset_reason is not None:
                        completed_episodes += 1
                    episode_limit_reached = (
                        max_episodes is not None
                        and completed_episodes >= max_episodes
                    )
                    if recorder is not None:
                        if episode_limit_reached:
                            recorder.end_episode(
                                episode_end_reason,
                                final_fields=task_record,
                            )
                        else:
                            recorder.rotate_episode(
                                episode_end_reason,
                                final_fields=task_record,
                            )
                    if episode_limit_reached:
                        run_end_reason = "episode_limit"
                        break
                    reset = _reset_episode(
                        env,
                        scenario,
                        executor,
                        task,
                        reset_manager,
                        seed=next_episode_seed,
                        pointer_position=window.mouse_position,
                    )
                    episode_seed = next_episode_seed
                    next_episode_seed += 1
                    observation = reset.observation
                    previous_recorded_safe_target = (
                        reset.previous_safe_target
                    )
                    target_height = reset.target_height_m
                    target_depth_y = reset.target_depth_y_m
                    gripper_target = reset.gripper_target
                    scripted_source.reset()
                    episode_step = 0
                    success_settle_steps = 0
            if recorder is not None:
                recorder.end_episode(
                    run_end_reason, final_fields=last_task_record
                )
                session_dir = recorder.session_dir
            else:
                session_dir = None
        if (
            session_dir is not None
            and config.collection.source == "scripted_pick_place"
        ):
            report = write_session_report(session_dir)
            print(
                "session_summary="
                f"{report['success_count']}/{report['episode_count']} success "
                f"({report['success_rate']:.1%}), "
                f"reasons={report['end_reasons']}"
            )
            print(f"session_report={session_dir / 'summary.json'}")
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
        "--episodes",
        type=int,
        default=None,
        help=(
            "Stop after N scripted episodes and write summary.json."
        ),
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
    run(
        load_config(args.config),
        max_steps=args.max_steps,
        max_episodes=args.episodes,
    )


if __name__ == "__main__":
    main()
