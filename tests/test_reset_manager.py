import numpy as np

from mani_sim.runtime.reset_manager import ResetManager


def test_reset_holds_actual_tcp_until_pointer_moves() -> None:
    manager = ResetManager(pointer_rearm_pixels=3.0)
    state = manager.reset([0.615, 0.0, 0.17], [100.0, 200.0])

    assert np.allclose(state.target, [0.615, 0.0, 0.17])
    assert state.target_height_m == 0.17
    assert not manager.accepts_pointer([102.0, 200.0])
    assert manager.accepts_pointer([103.0, 200.0])
    assert manager.accepts_pointer([100.0, 200.0])


def test_vertical_target_changes_only_height_before_pointer_rearm() -> None:
    target = ResetManager.vertical_target([0.615, 0.0, 0.17], 0.18)
    assert np.allclose(target, [0.615, 0.0, 0.18])
    depth_target = ResetManager.axis_target(
        [0.615, 0.0, 0.17], axis=1, value=0.02
    )
    assert np.allclose(depth_target, [0.615, 0.02, 0.17])


def test_pointer_settle_period_ignores_window_initialization_motion() -> None:
    manager = ResetManager(
        pointer_rearm_pixels=3.0, pointer_settle_steps=2
    )
    manager.reset([0.615, 0.0, 0.45], [0.0, 0.0])

    assert not manager.accepts_pointer([100.0, 100.0])
    assert not manager.accepts_pointer([110.0, 100.0])
    assert not manager.accepts_pointer([112.0, 100.0])
    assert manager.accepts_pointer([113.0, 100.0])
