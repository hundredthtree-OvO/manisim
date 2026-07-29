from __future__ import annotations

import numpy as np
import pytest
import torch

from mani_sim.config import load_config
from mani_sim.control.command import TaskSpaceCommand
from mani_sim.control.ee_servo import EEServo
from mani_sim.control.scene_collision_guard import SceneCollisionGuard
from mani_sim.control.workspace_guard import WorkspaceGuard
from mani_sim.environments.scenario import build_scenario
from mani_sim.experiments.checkpoint import ExperimentCheckpoint
from mani_sim.experiments.pre_lift import (
    PreLiftBranchCollector,
    PreLiftIntervention,
    default_pre_lift_interventions,
)
from mani_sim.robot_setup import initialize_panda
from mani_sim.runtime.command_executor import CommandExecutor
from mani_sim.runtime.observation import capture_runtime_observation


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="branch regression needs the external NVIDIA/Vulkan device",
    ),
]


def _executor(config) -> CommandExecutor:
    return CommandExecutor(
        servo=EEServo(gain=0.8, max_delta_m=0.01, deadband_m=0.001),
        workspace_guard=WorkspaceGuard(
            x_bounds_m=config.workspace.x_bounds_m,
            y_bounds_m=config.workspace.y_bounds_m,
            z_bounds_m=config.workspace.z_bounds_m,
            work_height_m=config.workspace.work_height_m,
            progress_epsilon_m=config.servo.progress_epsilon_m,
            saturation_steps=config.servo.saturation_steps,
            saturation_distance_m=config.servo.saturation_distance_m,
            release_target_delta_m=config.servo.release_target_delta_m,
        ),
        scene_guard=SceneCollisionGuard(
            ground_clearance_m=(
                config.collision_protection.ground_tcp_clearance_m
            ),
            obstacle_margin_m=(
                config.collision_protection.obstacle_margin_m
            ),
        ),
        reachability=None,
        controller_delta_limit_m=0.1,
        previous_safe_target_weight=(
            config.reachability.previous_safe_target_weight
        ),
        maximum_projected_target_step_m=(
            config.reachability.maximum_projected_target_step_m
        ),
        collision_protection_enabled=True,
    )


def _move_to(
    env,
    scenario,
    executor: CommandExecutor,
    target: np.ndarray,
    gripper: float,
    steps: int,
) -> None:
    for step in range(steps):
        observation = capture_runtime_observation(
            env.unwrapped, scenario
        )
        execution = executor.prepare(
            TaskSpaceCommand.create(
                target_position=target,
                gripper_position=gripper,
                timestamp=float(step),
                source="branch_test_setup",
            ),
            observation,
            obstacles=scenario.obstacles,
        )
        env.step(execution.action)


def test_checkpoint_fidelity_and_pre_lift_branch_collection(
    tmp_path,
) -> None:
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
    try:
        env.reset(seed=0)
        initialize_panda(env.unwrapped)
        scenario = build_scenario(env.unwrapped, config)
        executor = _executor(config)
        executor.reset(
            capture_runtime_observation(
                env.unwrapped, scenario
            ).tcp_position
        )
        cube = scenario.initial_position("target")
        assert cube is not None
        _move_to(
            env,
            scenario,
            executor,
            cube + np.array([0.0, 0.0, 0.07]),
            1.0,
            180,
        )
        _move_to(env, scenario, executor, cube, 1.0, 140)
        _move_to(env, scenario, executor, cube, -1.0, 100)
        anchor = capture_runtime_observation(env.unwrapped, scenario)
        assert "target" in anchor.grasped_objects

        checkpoint = ExperimentCheckpoint.capture(
            env.unwrapped,
            components={"executor": executor},
            user_state={"anchor": "pre_lift", "gripper_target": -1.0},
        )
        collector = PreLiftBranchCollector(
            base_env=env.unwrapped,
            scenario=scenario,
            executor=executor,
            horizon_steps=70,
        )

        repeats = collector.collect(
            checkpoint,
            (
                PreLiftIntervention("repeat_0"),
                PreLiftIntervention("repeat_1"),
                PreLiftIntervention("repeat_2"),
            ),
            checkpoint_id="pre_lift_fidelity",
        )
        final_tcp = np.asarray(
            [branch.final_tcp_position for branch in repeats.branches]
        )
        final_object = np.asarray(
            [
                branch.final_object_position
                for branch in repeats.branches
            ]
        )
        tcp_spread = float(
            np.max(np.linalg.norm(final_tcp - final_tcp[0], axis=1))
        )
        object_spread = float(
            np.max(
                np.linalg.norm(final_object - final_object[0], axis=1)
            )
        )
        tcp_trajectories = np.asarray(
            [
                [step.tcp_position for step in branch.steps]
                for branch in repeats.branches
            ]
        )
        object_trajectories = np.asarray(
            [
                [step.object_position for step in branch.steps]
                for branch in repeats.branches
            ]
        )
        tcp_trajectory_spread = float(
            np.max(
                np.linalg.norm(
                    tcp_trajectories - tcp_trajectories[0:1],
                    axis=-1,
                )
            )
        )
        object_trajectory_spread = float(
            np.max(
                np.linalg.norm(
                    object_trajectories - object_trajectories[0:1],
                    axis=-1,
                )
            )
        )
        assert tcp_spread <= 0.002
        assert object_spread <= 0.002
        assert tcp_trajectory_spread <= 0.002
        assert object_trajectory_spread <= 0.002
        assert all(
            branch.maintained_grasp for branch in repeats.branches
        )

        group = collector.collect(
            checkpoint,
            default_pre_lift_interventions(),
            checkpoint_id="pre_lift_group_0",
        )
        output = group.write_json(tmp_path / "branch_group.json")
        assert output.exists()
        assert len(group.branches) == 13
        assert all(len(branch.steps) == 70 for branch in group.branches)
        assert group.fixed_dynamics
        slow = next(
            branch
            for branch in group.branches
            if branch.intervention.name == "slow"
        )
        fast = next(
            branch
            for branch in group.branches
            if branch.intervention.name == "fast"
        )
        assert (
            fast.steps[20].tcp_position[2]
            > slow.steps[20].tcp_position[2]
        )
        print(
            "checkpoint_final_spread_m=",
            {
                "tcp": tcp_spread,
                "object": object_spread,
                "tcp_trajectory": tcp_trajectory_spread,
                "object_trajectory": object_trajectory_spread,
            },
        )
        print(
            "pre_lift_branch_summary=",
            [
                {
                    "name": branch.intervention.name,
                    "grasped": branch.maintained_grasp,
                    "slip_m": round(
                        branch.maximum_relative_xy_slip_m, 6
                    ),
                    "max_grip_n": round(
                        branch.maximum_grip_force_n, 6
                    ),
                    "max_object_n": round(
                        branch.maximum_object_force_n, 6
                    ),
                }
                for branch in group.branches
            ],
        )
    finally:
        env.close()
