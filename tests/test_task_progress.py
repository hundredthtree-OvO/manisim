import numpy as np

from mani_sim.tasks.base import TaskObservation
from mani_sim.tasks.pick_place import PickPlaceTask


def _observation(tcp, cube, grasped=False) -> TaskObservation:
    return TaskObservation(
        tcp_position=np.asarray(tcp, dtype=np.float64),
        object_positions={
            "target": np.asarray(cube, dtype=np.float64),
        },
        grasped_objects=(
            frozenset({"target"}) if grasped else frozenset()
        ),
    )


def test_pick_progress_advances_monotonically() -> None:
    task = PickPlaceTask(
        initial_object_height_m=0.02,
        approach_clearance_m=0.08,
        lift_height_m=0.10,
        goal_position_xy_m=(0.30, 0.30),
        goal_tolerance_m=0.04,
        place_height_tolerance_m=0.015,
    )

    state = task.update(_observation([0.2, 0.0, 0.3], [0.45, 0.0, 0.02]))
    assert state.phase == "approaching"

    state = task.update(_observation([0.45, 0.0, 0.08], [0.45, 0.0, 0.02]))
    assert state.phase == "approached"

    state = task.update(
        _observation([0.45, 0.0, 0.02], [0.45, 0.0, 0.02], True)
    )
    assert state.phase == "grasped"

    state = task.update(
        _observation([0.45, 0.0, 0.15], [0.45, 0.0, 0.13], True)
    )
    assert state.phase == "lifted"

    state = task.update(
        _observation([0.30, 0.30, 0.15], [0.30, 0.30, 0.13], True)
    )
    assert state.phase == "transported"

    state = task.update(
        _observation([0.30, 0.30, 0.08], [0.30, 0.30, 0.04])
    )
    assert state.phase == "released"

    state = task.update(
        _observation([0.30, 0.30, 0.08], [0.30, 0.30, 0.02])
    )
    assert state.phase == "placed"
    assert task.record_fields(
        state,
        _observation([0.30, 0.30, 0.08], [0.30, 0.30, 0.02]),
    )["task_placed"]
    fields = dict(
        task.ui_fields(
            state,
            _observation([0.30, 0.30, 0.08], [0.30, 0.30, 0.02]),
        )
    )
    assert fields["phase"] == "placed"
    assert fields["success"] == "yes"

    task.reset()
    state = task.update(_observation([0.2, 0.0, 0.3], [0.45, 0.0, 0.02]))
    assert state.phase == "approaching"
    assert not state.placed


def test_pick_place_experiment_state_round_trip() -> None:
    task = PickPlaceTask(
        initial_object_height_m=0.02,
        approach_clearance_m=0.08,
        lift_height_m=0.10,
        goal_position_xy_m=(0.30, 0.30),
        goal_tolerance_m=0.04,
        place_height_tolerance_m=0.015,
    )
    task.update(
        _observation(
            [0.45, 0.0, 0.15],
            [0.45, 0.0, 0.13],
            True,
        )
    )
    expected = task.get_experiment_state()
    task.reset()

    task.set_experiment_state(expected)

    restored = task.get_experiment_state()
    assert restored["ever_grasped"]
    assert restored["lifted"]
    assert np.allclose(
        restored["goal_position_xy_m"],
        expected["goal_position_xy_m"],
    )
