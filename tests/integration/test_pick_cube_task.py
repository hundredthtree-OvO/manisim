from dataclasses import replace

import numpy as np
import pytest
import torch

from mani_sim.config import load_config
from mani_sim.control.ee_servo import EEServo, build_normalized_panda_action
from mani_sim.control.scene_collision_guard import SceneCollisionGuard
from mani_sim.robot_setup import initialize_panda
from mani_sim.reachability import ReachabilityMap
from mani_sim.runtime.contact_forces import sample_contact_forces
from mani_sim.task_progress import PickProgress
from mani_sim.task_scene import build_task_scene


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="pick regression needs the external NVIDIA/Vulkan device",
    ),
]


def _vector(value) -> np.ndarray:
    return value[0].detach().cpu().numpy()


def _move(env, servo, target, gripper, steps) -> float:
    for _ in range(steps):
        tcp = _vector(env.unwrapped.agent.tcp_pose.p)
        delta = servo.metric_delta(target, tcp)
        action = build_normalized_panda_action(delta, gripper, 0.1)
        env.step(action)
    return float(np.linalg.norm(target - _vector(env.unwrapped.agent.tcp_pose.p)))


def test_approach_grasp_and_lift_cube() -> None:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    config = load_config("configs/demo0.yaml")
    env = gym.make(
        "Empty-v1",
        obs_mode="none",
        reward_mode="none",
        render_mode=None,
        control_mode="pd_ee_delta_pos",
        robot_uids="panda_wristcam",
        sim_backend="cpu",
    )
    servo = EEServo(gain=0.8, max_delta_m=0.01, deadband_m=0.001)
    try:
        env.reset(seed=0)
        initialize_panda(env.unwrapped)
        scene = build_task_scene(env.unwrapped, config)
        assert scene.cube is not None
        cube_start = _vector(scene.cube.pose.p)
        reachability = ReachabilityMap.load(config.reachability.path)
        projection = reachability.project_continuous(
            np.array([0.30, 0.0, cube_start[2]]), cube_start
        )
        assert not projection.projected
        assert np.allclose(projection.target, cube_start)
        progress = PickProgress(
            initial_cube_height_m=float(cube_start[2]),
            approach_clearance_m=config.cube_task.approach_clearance_m,
            lift_height_m=config.cube_task.lift_height_m,
            goal_position_xy_m=config.cube_task.goal_position_xy_m,
            goal_tolerance_m=config.cube_task.goal_tolerance_m,
            place_height_tolerance_m=(
                config.cube_task.place_height_tolerance_m
            ),
        )

        above = cube_start + np.array([0.0, 0.0, 0.07])
        approach_error = _move(env, servo, above, 1.0, 240)
        state = progress.update(
            _vector(env.unwrapped.agent.tcp_pose.p),
            _vector(scene.cube.pose.p),
            False,
        )
        assert state.approached

        grasp_target = cube_start.copy()
        grasp_error = _move(env, servo, grasp_target, 1.0, 180)
        _move(env, servo, grasp_target, -1.0, 120)
        grasped = bool(
            env.unwrapped.agent.is_grasping(scene.cube)[0].detach().cpu()
        )
        state = progress.update(
            _vector(env.unwrapped.agent.tcp_pose.p),
            _vector(scene.cube.pose.p),
            grasped,
        )
        assert state.grasped
        force_sample = sample_contact_forces(env.unwrapped, scene)
        print(
            "grasp_forces_N=",
            [
                round(force_sample.left_finger_n, 6),
                round(force_sample.right_finger_n, 6),
                round(force_sample.object_net_n, 6),
            ],
        )
        assert force_sample.grip_n >= 0.5
        assert force_sample.object_net_n > 0.0

        lift_target = grasp_target + np.array(
            [0.0, 0.0, config.cube_task.lift_height_m + 0.04]
        )
        lift_error = _move(env, servo, lift_target, -1.0, 300)
        cube_end = _vector(scene.cube.pose.p)
        state = progress.update(
            _vector(env.unwrapped.agent.tcp_pose.p), cube_end, True
        )
        assert state.lifted
        assert cube_end[2] - cube_start[2] >= config.cube_task.lift_height_m

        goal_xy = np.asarray(config.cube_task.goal_position_xy_m)
        transport_target = lift_target.copy()
        transport_target[:2] = goal_xy
        transport_error = _move(
            env, servo, transport_target, -1.0, 300
        )
        transported_cube = _vector(scene.cube.pose.p)
        grasped = bool(
            env.unwrapped.agent.is_grasping(scene.cube)[0].detach().cpu()
        )
        state = progress.update(
            _vector(env.unwrapped.agent.tcp_pose.p),
            transported_cube,
            grasped,
        )
        assert state.transported

        place_target = np.array([goal_xy[0], goal_xy[1], cube_start[2]])
        projection = reachability.project_continuous(
            np.array([0.30, 0.0, place_target[2]]), place_target
        )
        assert not projection.projected
        place_error = _move(env, servo, place_target, -1.0, 300)
        _move(env, servo, place_target, 1.0, 180)
        retreat_target = place_target + np.array([0.0, 0.0, 0.08])
        retreat_error = _move(env, servo, retreat_target, 1.0, 240)
        cube_final = _vector(scene.cube.pose.p)
        grasped = bool(
            env.unwrapped.agent.is_grasping(scene.cube)[0].detach().cpu()
        )
        state = progress.update(
            _vector(env.unwrapped.agent.tcp_pose.p), cube_final, grasped
        )

        print(
            "pick_stage_errors_m=",
            [
                round(approach_error, 6),
                round(grasp_error, 6),
                round(lift_error, 6),
                round(transport_error, 6),
                round(place_error, 6),
                round(retreat_error, 6),
            ],
        )
        print("cube_lift_m=", round(float(cube_end[2] - cube_start[2]), 6))
        print("cube_final_position=", cube_final.tolist())
        print(
            "cube_goal_xy_error_m=",
            round(float(np.linalg.norm(cube_final[:2] - goal_xy)), 6),
        )
        assert not grasped
        assert state.released
        assert state.placed
    finally:
        env.close()


def test_obstacle_guard_prevents_physical_contact() -> None:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    config = load_config("configs/demo0.yaml")
    config = replace(
        config,
        collision_protection=replace(
            config.collision_protection, obstacle_enabled=True
        ),
    )
    env = gym.make(
        "Empty-v1",
        obs_mode="none",
        reward_mode="none",
        render_mode=None,
        control_mode="pd_ee_delta_pos",
        robot_uids="panda_wristcam",
        sim_backend="cpu",
    )
    servo = EEServo(gain=0.8, max_delta_m=0.01, deadband_m=0.001)
    try:
        env.reset(seed=0)
        initialize_panda(env.unwrapped)
        scene = build_task_scene(env.unwrapped, config)
        assert scene.obstacle is not None
        guard = SceneCollisionGuard(
            ground_clearance_m=(
                config.collision_protection.ground_tcp_clearance_m
            ),
            obstacle_margin_m=config.collision_protection.obstacle_margin_m,
        )
        raw_target = np.asarray(
            config.collision_protection.obstacle_center_m
        )
        protected = guard.protect(
            raw_target, obstacles=scene.obstacles
        )
        assert protected.protected

        peak_force = 0.0
        for _ in range(300):
            tcp = _vector(env.unwrapped.agent.tcp_pose.p)
            delta = servo.metric_delta(protected.target, tcp)
            action = build_normalized_panda_action(delta, 1.0, 0.1)
            env.step(action)
            for link in env.unwrapped.agent.robot.links:
                force = env.unwrapped.scene.get_pairwise_contact_forces(
                    scene.obstacle, link
                )
                peak_force = max(
                    peak_force,
                    float(torch.linalg.norm(force).detach().cpu()),
                )
        print("protected_obstacle_target=", protected.target.tolist())
        print("obstacle_contact_force_N=", round(peak_force, 6))
        assert peak_force < 0.01
    finally:
        env.close()
