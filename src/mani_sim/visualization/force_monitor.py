from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import sapien
from mani_skill.utils import sapien_utils
from sapien import internal_renderer as R
from sapien.utils.viewer.plugin import Plugin


@dataclass(frozen=True)
class ForceDisplaySample:
    grip_n: float
    object_n: float
    unintended_n: float
    left_finger_n: float
    right_finger_n: float
    threshold_n: float


class ForceHistory:
    def __init__(self, capacity: int):
        self.samples: deque[tuple[float, float, float]] = deque(
            maxlen=capacity
        )

    def append(self, sample: ForceDisplaySample) -> None:
        self.samples.append(
            (sample.grip_n, sample.object_n, sample.unintended_n)
        )


class ForceChartRasterizer:
    COLORS = (
        np.array([55, 145, 255, 255], dtype=np.uint8),
        np.array([65, 210, 105, 255], dtype=np.uint8),
        np.array([245, 75, 75, 255], dtype=np.uint8),
    )

    def __init__(
        self,
        width: int = 320,
        height: int = 190,
        maximum_n: float = 40.0,
    ):
        self.width = width
        self.height = height
        self.maximum_n = maximum_n

    def render(
        self,
        samples: tuple[tuple[float, float, float], ...],
        threshold_n: float,
    ) -> np.ndarray:
        image = np.full(
            (self.height, self.width, 4),
            [24, 27, 32, 255],
            dtype=np.uint8,
        )
        left, right, top, bottom = 34, self.width - 8, 8, self.height - 22
        for fraction in np.linspace(0.0, 1.0, 5):
            y = round(bottom - fraction * (bottom - top))
            image[y : y + 1, left:right] = [64, 68, 76, 255]
        image[top : bottom + 1, left : left + 1] = [140, 145, 155, 255]
        image[bottom : bottom + 1, left:right] = [140, 145, 155, 255]

        threshold_y = self._y(threshold_n, top, bottom)
        for x in range(left, right, 6):
            image[threshold_y : threshold_y + 1, x : x + 3] = [
                245,
                170,
                45,
                255,
            ]

        if samples:
            sample_indices = np.linspace(
                0, len(samples) - 1, min(right - left, len(samples))
            ).astype(int)
            x_values = np.linspace(
                left, right - 1, len(sample_indices)
            ).astype(int)
            values = np.asarray(samples, dtype=np.float64)[sample_indices]
            for channel, color in enumerate(self.COLORS):
                y_values = np.array(
                    [
                        self._y(value, top, bottom)
                        for value in values[:, channel]
                    ]
                )
                for index in range(1, len(x_values)):
                    self._line(
                        image,
                        x_values[index - 1],
                        y_values[index - 1],
                        x_values[index],
                        y_values[index],
                        color,
                    )
        return image

    def _y(self, value: float, top: int, bottom: int) -> int:
        fraction = float(np.clip(value / self.maximum_n, 0.0, 1.0))
        return round(bottom - fraction * (bottom - top))

    @staticmethod
    def _line(
        image: np.ndarray,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        color: np.ndarray,
    ) -> None:
        count = max(abs(x1 - x0), abs(y1 - y0), 1) + 1
        xs = np.linspace(x0, x1, count).astype(int)
        ys = np.linspace(y0, y1, count).astype(int)
        for offset in (-1, 0, 1):
            clipped = np.clip(ys + offset, 0, image.shape[0] - 1)
            image[clipped, xs] = color


class ForceChartSurface:
    """Render an uploaded chart texture through a dedicated scene camera."""

    CAMERA_NAME = "force_chart_camera"

    @staticmethod
    def _orient_for_plane(image: np.ndarray) -> np.ndarray:
        """Map raster X/Y onto the plane's screen-horizontal/screen-vertical."""

        return np.ascontiguousarray(np.transpose(image, (1, 0, 2)))

    def __init__(self, base_env: Any, rasterizer: ForceChartRasterizer):
        self.rasterizer = rasterizer
        initial = self._orient_for_plane(rasterizer.render((), 8.0))
        self.texture = sapien.render.RenderTexture2D(
            initial,
            "R8G8B8A8Unorm",
            filter_mode="linear",
            address_mode="edge",
            srgb=True,
        )
        material = sapien.render.RenderMaterial()
        material.set_base_color_texture(self.texture)
        material.set_emission_texture(self.texture)
        material.set_emission([1.0, 1.0, 1.0, 1.0])

        chart_center = np.array([50.0, 0.0, 0.0])
        aspect = rasterizer.width / rasterizer.height
        shape = sapien.render.RenderShapePlane(
            # RenderShapePlane lies in local YZ with normal +X.
            np.array([1.0, aspect, 1.0], dtype=np.float32),
            material,
        )
        body = sapien.render.RenderBodyComponent().attach(shape)
        entity = sapien.Entity()
        entity.name = "force_chart_surface"
        entity.pose = sapien.Pose(p=chart_center)
        entity.add_component(body)
        entity.add_to_scene(base_env.scene.sub_scenes[0])
        self.entity = entity

        eye = chart_center + np.array([2.0, 0.0, 0.0])
        self.camera = base_env.scene.add_camera(
            name=self.CAMERA_NAME,
            pose=sapien_utils.look_at(
                eye, chart_center, up=[0.0, 0.0, 1.0]
            ).sp,
            width=rasterizer.width,
            height=rasterizer.height,
            fovy=0.93,
            near=0.05,
            far=5.0,
        )

    def update(
        self,
        samples: tuple[tuple[float, float, float], ...],
        threshold_n: float,
    ) -> None:
        image = self.rasterizer.render(samples, threshold_n)
        self.texture.upload(self._orient_for_plane(image))


class ForceMonitorPanel(Plugin):
    """Independent force-history window with a camera-backed color chart."""

    def __init__(
        self,
        base_env: Any,
        *,
        history_capacity: int,
        panel_width: int = 360,
        panel_height: int = 285,
        runtime_panel_width: int = 360,
    ):
        self.panel_width = panel_width
        self.panel_height = panel_height
        self.runtime_panel_width = runtime_panel_width
        self.history = ForceHistory(history_capacity)
        self.rasterizer = ForceChartRasterizer()
        self.surface = ForceChartSurface(base_env, self.rasterizer)
        self.sample = ForceDisplaySample(0.0, 0.0, 0.0, 0.0, 0.0, 8.0)
        self.ui_window = None
        self.ui_picture = R.UIPicture()

    def update(self, sample: ForceDisplaySample) -> None:
        self.sample = sample
        self.history.append(sample)
        self.surface.update(tuple(self.history.samples), sample.threshold_n)

    def panel_rect(self) -> tuple[float, float, float, float]:
        width, height = self.viewer.window.size
        right = max(0.0, width - self.runtime_panel_width - 20.0)
        left = max(0.0, right - self.panel_width - 10.0)
        top = max(10.0, height - self.panel_height - 10.0)
        return left, top, right, top + self.panel_height

    def pointer_over_panel(self, x: float, y: float) -> bool:
        left, top, right, bottom = self.panel_rect()
        return left <= x <= right and top <= y <= bottom

    def _camera(self):
        return next(
            (
                camera
                for camera in self.viewer.cameras
                if camera.name.endswith(self.surface.CAMERA_NAME)
            ),
            None,
        )

    def get_ui_windows(self):
        if self.ui_window is None:
            self.ui_window = (
                R.UIWindow()
                .Label("Force monitor")
                .Size(self.panel_width, self.panel_height)
            )
        left, top, _, _ = self.panel_rect()
        self.ui_window.Pos(left, top)
        self.ui_window.remove_children()
        self.ui_window.append(
            R.UIDisplayText().Text(
                "BLUE grip | GREEN object | RED unintended | ORANGE 8 N"
            )
        )
        camera = self._camera()
        if camera is not None:
            camera.take_picture()
            self.ui_picture.Size(320, 190)
            self.ui_picture.Picture(camera._internal_renderer, "Color")
            self.ui_window.append(self.ui_picture)
        else:
            self.ui_window.append(
                R.UIDisplayText().Text("force chart camera unavailable")
            )
        sample = self.sample
        self.ui_window.append(
            R.UIDisplayText().Text(
                f"L/R {sample.left_finger_n:.2f}/{sample.right_finger_n:.2f} N"
                f" | object {sample.object_n:.2f} N"
                f" | unintended {sample.unintended_n:.2f} N"
            )
        )
        return [self.ui_window]
