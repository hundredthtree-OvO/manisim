import numpy as np

from mani_sim.action_sources.base import ActionSource
from mani_sim.action_sources.mouse import MouseActionSource
from mani_sim.action_sources.scripted_pick_place import (
    ScriptedPickPlaceSource,
)
from mani_sim.control.command import TaskSpaceCommand
from mani_sim.runtime.observation import RuntimeObservation


def test_mouse_action_source_returns_latest_copied_command() -> None:
    source = MouseActionSource()
    target = np.array([0.45, 0.10, 0.30])
    source.update(
        TaskSpaceCommand.create(
            target_position=target,
            gripper_position=-1.0,
            timestamp=12.5,
            source="human",
        )
    )
    target[:] = 0.0

    command = source.act(RuntimeObservation.empty())

    assert isinstance(source, ActionSource)
    assert np.allclose(command.target_position, [0.45, 0.10, 0.30])
    assert command.gripper_position == -1.0
    assert command.source == "human"


def test_mouse_action_source_can_publish_invalid_hold() -> None:
    source = MouseActionSource()
    source.update(
        TaskSpaceCommand.create(
            target_position=[0.6, 0.0, 0.45],
            gripper_position=1.0,
            timestamp=1.0,
            valid=False,
            source="human",
        )
    )

    assert not source.act(RuntimeObservation.empty()).valid


def _pick_place_observation(
    tcp: list[float],
    *,
    target: list[float] = [0.45, 0.0, 0.02],
    grasped: bool = False,
) -> RuntimeObservation:
    return RuntimeObservation.create(
        tcp_position=tcp,
        qpos=[],
        qvel=[],
        object_positions={
            "target": target,
            "goal": [0.30, 0.30, 0.001],
        },
        grasped_objects={"target"} if grasped else (),
    )


def test_scripted_pick_place_source_runs_canonical_phase_sequence() -> None:
    source = ScriptedPickPlaceSource(
        approach_clearance_m=0.08,
        lift_height_m=0.10,
        position_tolerance_m=0.01,
        release_settle_steps=2,
    )

    command = source.act(_pick_place_observation([0.45, 0.0, 0.45]))
    assert isinstance(source, ActionSource)
    assert command.source == "scripted_pick_place"
    assert command.metadata == {"policy_phase": "approach"}
    assert np.allclose(command.target_position, [0.45, 0.0, 0.10])
    assert command.gripper_position == 1.0

    source.act(_pick_place_observation([0.45, 0.0, 0.10]))
    command = source.act(_pick_place_observation([0.45, 0.0, 0.02]))
    assert command.metadata == {"policy_phase": "close"}
    assert command.gripper_position == -1.0

    command = source.act(
        _pick_place_observation([0.45, 0.0, 0.02], grasped=True)
    )
    assert command.metadata == {"policy_phase": "lift"}
    assert np.allclose(command.target_position, [0.45, 0.0, 0.15])

    source.act(
        _pick_place_observation(
            [0.45, 0.0, 0.15],
            target=[0.45, 0.0, 0.15],
            grasped=True,
        )
    )
    command = source.act(
        _pick_place_observation(
            [0.30, 0.30, 0.15],
            target=[0.30, 0.30, 0.15],
            grasped=True,
        )
    )
    assert command.metadata == {"policy_phase": "lower"}
    assert np.allclose(command.target_position, [0.30, 0.30, 0.02])

    command = source.act(
        _pick_place_observation(
            [0.30, 0.30, 0.02],
            target=[0.30, 0.30, 0.02],
            grasped=True,
        )
    )
    assert command.metadata == {"policy_phase": "open"}
    assert command.gripper_position == 1.0

    source.act(
        _pick_place_observation(
            [0.30, 0.30, 0.02],
            target=[0.30, 0.30, 0.02],
        )
    )
    command = source.act(
        _pick_place_observation(
            [0.30, 0.30, 0.02],
            target=[0.30, 0.30, 0.02],
        )
    )
    assert command.metadata == {"policy_phase": "retreat"}


def test_scripted_pick_place_source_reset_restarts_at_approach() -> None:
    source = ScriptedPickPlaceSource()
    source.act(_pick_place_observation([0.45, 0.0, 0.10]))

    source.reset()
    command = source.act(_pick_place_observation([0.45, 0.0, 0.45]))

    assert source.phase == "approach"
    assert command.metadata == {"policy_phase": "approach"}


def test_scripted_source_experiment_state_round_trip() -> None:
    source = ScriptedPickPlaceSource()
    source.act(_pick_place_observation([0.45, 0.0, 0.10]))
    expected = source.get_experiment_state()
    source.reset()

    source.set_experiment_state(expected)

    assert source.get_experiment_state() == expected
