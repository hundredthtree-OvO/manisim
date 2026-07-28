import numpy as np

from mani_sim.control.scene_collision_guard import (
    AxisAlignedBox,
    SceneCollisionGuard,
)


def _guard() -> SceneCollisionGuard:
    return SceneCollisionGuard(
        ground_clearance_m=0.015,
        obstacle_margin_m=0.02,
    )


def test_ground_target_is_clamped_to_clearance() -> None:
    result = _guard().protect([0.4, 0.0, 0.0])
    assert result.protected
    assert result.reason == "ground_clearance"
    assert np.allclose(result.target, [0.4, 0.0, 0.015])


def test_target_inside_static_obstacle_is_projected_to_nearest_face() -> None:
    obstacle = AxisAlignedBox.create(
        [0.58, 0.25, 0.05], [0.10, 0.10, 0.10], "block"
    )
    result = _guard().protect(
        [0.58, 0.25, 0.10], obstacles=(obstacle,)
    )
    assert result.protected
    assert result.reason == "obstacle_clearance:block"
    assert np.isclose(result.target[2], 0.12)
    assert not np.allclose(result.target, [0.58, 0.25, 0.10])


def test_free_space_target_is_not_clamped() -> None:
    result = _guard().protect([0.48, 0.0, 0.02])
    assert not result.protected
