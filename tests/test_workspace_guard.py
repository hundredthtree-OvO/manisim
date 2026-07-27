import numpy as np

from mani_sim.control.workspace_guard import WorkspaceGuard


def make_guard(saturation_steps: int = 2) -> WorkspaceGuard:
    return WorkspaceGuard(
        x_bounds_m=(0.1, 0.8),
        y_bounds_m=(-0.5, 0.5),
        work_height_m=0.4,
        progress_epsilon_m=0.001,
        saturation_steps=saturation_steps,
        saturation_distance_m=0.03,
        release_target_delta_m=0.04,
    )


def test_guard_clips_coarse_workspace() -> None:
    result = make_guard().update([2, -2, 9], [0.2, 0, 0.4])
    np.testing.assert_allclose(result.target, [0.8, -0.5, 0.4])


def test_guard_saturates_after_repeated_stall() -> None:
    guard = make_guard()
    target = [0.8, 0, 0.4]
    tcp = [0.2, 0, 0.4]
    assert not guard.update(target, tcp).saturated
    assert not guard.update(target, tcp).saturated
    result = guard.update(target, tcp)
    assert result.saturated
    np.testing.assert_allclose(result.target, tcp)


def test_guard_releases_after_material_target_change() -> None:
    guard = make_guard(saturation_steps=1)
    guard.update([0.8, 0, 0.4], [0.2, 0, 0.4])
    assert guard.update([0.8, 0, 0.4], [0.2, 0, 0.4]).saturated
    result = guard.update([0.7, 0.2, 0.4], [0.2, 0, 0.4])
    assert not result.saturated
