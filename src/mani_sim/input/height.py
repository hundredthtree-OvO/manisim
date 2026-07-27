from __future__ import annotations

import numpy as np


def update_height(
    current_height_m: float,
    *,
    up_pressed: bool,
    down_pressed: bool,
    speed_mps: float,
    dt_seconds: float,
    bounds_m: tuple[float, float],
) -> float:
    direction = float(up_pressed) - float(down_pressed)
    return float(
        np.clip(
            current_height_m + direction * speed_mps * dt_seconds,
            *bounds_m,
        )
    )
