from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mani_sim.mapping.screen_to_plane import (
    intersect_horizontal_plane,
    screen_to_world_ray,
)


@dataclass(frozen=True)
class PointerSample:
    pixel: tuple[float, float]
    world_target: np.ndarray | None
    valid: bool


class SapienPointer:
    """Thin adapter around the public surface exposed by SAPIEN RenderWindow."""

    def __init__(self, window: Any, work_height_m: float):
        self.window = window
        self.work_height_m = work_height_m

    def sample(self) -> PointerSample:
        x, y = (float(value) for value in self.window.mouse_position)
        width, height = (int(value) for value in self.window.size)
        pixel = (x, y)
        if x < 0 or y < 0 or x >= width or y >= height:
            return PointerSample(pixel, None, False)

        origin, direction = screen_to_world_ray(
            pixel,
            (width, height),
            self.window.get_camera_model_matrix(),
            self.window.get_camera_projection_matrix(),
        )
        target = intersect_horizontal_plane(
            origin, direction, height_m=self.work_height_m
        )
        return PointerSample(pixel, target, target is not None)
