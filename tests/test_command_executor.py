import numpy as np

from mani_sim.control.command import TaskSpaceCommand
from mani_sim.control.ee_servo import EEServo
from mani_sim.control.scene_collision_guard import SceneCollisionGuard
from mani_sim.control.workspace_guard import WorkspaceGuard
from mani_sim.runtime.command_executor import CommandExecutor
from mani_sim.runtime.observation import RuntimeObservation


def _executor() -> CommandExecutor:
    executor = CommandExecutor(
        servo=EEServo(gain=0.5, max_delta_m=0.01, deadband_m=0.002),
        workspace_guard=WorkspaceGuard(
            x_bounds_m=(0.15, 0.75),
            y_bounds_m=(-0.55, 0.55),
            z_bounds_m=(0.02, 0.65),
            work_height_m=0.45,
            progress_epsilon_m=0.0005,
            saturation_steps=12,
            saturation_distance_m=0.03,
            release_target_delta_m=0.04,
        ),
        scene_guard=SceneCollisionGuard(
            ground_clearance_m=0.015,
            obstacle_margin_m=0.02,
        ),
        reachability=None,
        controller_delta_limit_m=0.1,
        previous_safe_target_weight=0.7,
        maximum_projected_target_step_m=0.03,
        collision_protection_enabled=True,
    )
    executor.reset(np.array([0.6, 0.0, 0.45]))
    return executor


def _observation() -> RuntimeObservation:
    return RuntimeObservation.create(
        tcp_position=[0.6, 0.0, 0.45],
        qpos=[],
        qvel=[],
        object_positions={},
    )


def test_invalid_command_holds_previous_safe_target() -> None:
    executor = _executor()
    command = TaskSpaceCommand.create(
        target_position=[0.0, 0.0, 0.0],
        gripper_position=1.0,
        timestamp=0.0,
        source="human",
        valid=False,
    )

    execution = executor.prepare(command, _observation())

    assert np.allclose(execution.safe_target, [0.6, 0.0, 0.45])
    assert np.allclose(execution.action, [0.0, 0.0, 0.0, 1.0])
    assert execution.guard_result is None


def test_valid_command_uses_shared_guard_and_servo_chain() -> None:
    executor = _executor()
    command = TaskSpaceCommand.create(
        target_position=[0.8, 0.1, 0.7],
        gripper_position=-1.0,
        timestamp=1.0,
        source="scripted",
    )

    execution = executor.prepare(command, _observation())

    assert np.allclose(execution.raw_target, [0.8, 0.1, 0.7])
    assert np.allclose(execution.safe_target, [0.75, 0.1, 0.65])
    assert execution.guard_result is not None
    assert execution.command.source == "scripted"
    assert execution.action.shape == (4,)
    assert execution.action[-1] == -1.0
