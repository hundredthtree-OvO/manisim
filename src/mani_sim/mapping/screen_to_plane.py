from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def screen_to_world_ray(
    pixel: tuple[float, float],
    viewport_size: tuple[int, int],
    camera_model_matrix: ArrayLike,
    projection_matrix: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert a SAPIEN viewer pixel to a world-space ray.

    SAPIEN's viewer uses a camera looking along local ``-Z`` and a Vulkan-style
    projection whose Y scale is negative. Window pixels have their origin at
    the top-left, hence both NDC axes use ``2 * pixel / size - 1`` here.
    """

    width, height = viewport_size
    if width <= 0 or height <= 0:
        raise ValueError("viewport dimensions must be positive")

    x, y = pixel
    projection = np.asarray(projection_matrix, dtype=np.float64)
    model = np.asarray(camera_model_matrix, dtype=np.float64)
    if projection.shape == (1, 4, 4):
        projection = projection[0]
    if model.shape == (1, 4, 4):
        model = model[0]
    if projection.shape != (4, 4) or model.shape != (4, 4):
        raise ValueError("camera matrices must both have shape (4, 4)")
    if abs(projection[0, 0]) < 1e-12 or abs(projection[1, 1]) < 1e-12:
        raise ValueError("invalid projection scale")

    ndc_x = 2.0 * float(x) / float(width) - 1.0
    ndc_y = 2.0 * float(y) / float(height) - 1.0
    direction_camera = np.array(
        [
            ndc_x / projection[0, 0],
            ndc_y / projection[1, 1],
            -1.0,
        ],
        dtype=np.float64,
    )
    direction_world = model[:3, :3] @ direction_camera
    norm = np.linalg.norm(direction_world)
    if norm < 1e-12:
        raise ValueError("camera produced a zero-length ray")
    return model[:3, 3].copy(), direction_world / norm


def intersect_horizontal_plane(
    origin: ArrayLike,
    direction: ArrayLike,
    height_m: float,
    *,
    parallel_epsilon: float = 1e-8,
) -> NDArray[np.float64] | None:
    """Return the forward ray intersection with ``z = height_m``."""

    origin_array = np.asarray(origin, dtype=np.float64)
    direction_array = np.asarray(direction, dtype=np.float64)
    if origin_array.shape != (3,) or direction_array.shape != (3,):
        raise ValueError("origin and direction must have shape (3,)")
    if abs(direction_array[2]) <= parallel_epsilon:
        return None
    distance = (float(height_m) - origin_array[2]) / direction_array[2]
    if distance <= 0:
        return None
    return origin_array + distance * direction_array


def intersect_axis_plane(
    origin: ArrayLike,
    direction: ArrayLike,
    *,
    axis: int,
    value: float,
    parallel_epsilon: float = 1e-8,
) -> NDArray[np.float64] | None:
    """Return the forward ray intersection with one world-axis plane."""

    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    origin_array = np.asarray(origin, dtype=np.float64)
    direction_array = np.asarray(direction, dtype=np.float64)
    if origin_array.shape != (3,) or direction_array.shape != (3,):
        raise ValueError("origin and direction must have shape (3,)")
    if abs(direction_array[axis]) <= parallel_epsilon:
        return None
    distance = (float(value) - origin_array[axis]) / direction_array[axis]
    if distance <= 0:
        return None
    return origin_array + distance * direction_array
