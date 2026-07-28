import numpy as np

from mani_sim.mapping.screen_to_plane import (
    intersect_axis_plane,
    intersect_horizontal_plane,
    screen_to_world_ray,
)


def test_center_pixel_follows_camera_minus_z() -> None:
    model = np.eye(4)
    model[:3, 3] = [1.0, 2.0, 3.0]
    projection = np.diag([2.0, -2.0, -1.0, 1.0])

    origin, direction = screen_to_world_ray(
        (50, 25), (100, 50), model, projection
    )

    np.testing.assert_allclose(origin, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(direction, [0.0, 0.0, -1.0])


def test_singleton_batched_camera_matrices_are_supported() -> None:
    projection = np.diag([2.0, -2.0, -1.0, 1.0])[None, ...]
    model = np.eye(4)[None, ...]
    _, direction = screen_to_world_ray(
        (50, 25), (100, 50), model, projection
    )
    np.testing.assert_allclose(direction, [0.0, 0.0, -1.0])


def test_top_pixel_points_along_camera_positive_y() -> None:
    projection = np.diag([1.0, -1.0, -1.0, 1.0])
    _, direction = screen_to_world_ray(
        (50, 0), (100, 100), np.eye(4), projection
    )
    assert direction[1] > 0


def test_horizontal_plane_intersection() -> None:
    point = intersect_horizontal_plane([0, 0, 2], [0, 0, -1], 0.5)
    np.testing.assert_allclose(point, [0, 0, 0.5])


def test_parallel_and_backward_rays_do_not_intersect() -> None:
    assert intersect_horizontal_plane([0, 0, 1], [1, 0, 0], 0.5) is None
    assert intersect_horizontal_plane([0, 0, 1], [0, 0, 1], 0.5) is None


def test_front_ray_intersects_constant_y_depth_plane() -> None:
    point = intersect_axis_plane(
        [0.5, -1.0, 0.4], [0.1, 1.0, -0.2], axis=1, value=0.0
    )
    np.testing.assert_allclose(point, [0.6, 0.0, 0.2])


def test_axis_plane_rejects_invalid_axis_and_parallel_ray() -> None:
    assert (
        intersect_axis_plane([0, 0, 0], [1, 0, 0], axis=1, value=1)
        is None
    )
    try:
        intersect_axis_plane([0, 0, 0], [1, 0, 0], axis=3, value=1)
    except ValueError as error:
        assert "axis" in str(error)
    else:
        raise AssertionError("invalid axis should fail")
