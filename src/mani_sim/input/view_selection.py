from __future__ import annotations

import numpy as np


class ViewSelection:
    """Tracks active control/observe views and safely rearms pointer input."""

    def __init__(self, rearm_distance_px: float = 3.0):
        self.active_view = 1
        self.rearm_distance_px = rearm_distance_px
        self._rearm_pixel: np.ndarray | None = None

    def switch(self, view: int, mouse_pixel: tuple[float, float]) -> None:
        if view not in (1, 2, 3):
            raise ValueError("view must be 1, 2, or 3")
        if view == self.active_view:
            return
        self.active_view = view
        if view in (1, 2):
            self._rearm_pixel = np.asarray(mouse_pixel, dtype=np.float64)

    def accepts_pointer(self, mouse_pixel: tuple[float, float]) -> bool:
        if self.active_view not in (1, 2):
            return False
        if self._rearm_pixel is None:
            return True
        pixel = np.asarray(mouse_pixel, dtype=np.float64)
        if np.linalg.norm(pixel - self._rearm_pixel) < self.rearm_distance_px:
            return False
        self._rearm_pixel = None
        return True
