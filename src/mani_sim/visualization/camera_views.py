from __future__ import annotations

from sapien import internal_renderer as R
from sapien.utils.viewer.plugin import Plugin


class AuxiliaryCameraPanel(Plugin):
    """Single-window front/wrist monitor; only the main viewport accepts input."""

    def __init__(
        self,
        front_camera_name: str = "front_observer",
        wrist_camera_name: str = "hand_camera",
        panel_width: int = 360,
    ):
        self.camera_names = (front_camera_name, wrist_camera_name)
        self.panel_width = panel_width
        self.panel_height = 610
        self.active_view = 1
        self.ui_window = None
        self.ui_pictures = None

    def set_active_view(self, view: int) -> None:
        if view not in (1, 2, 3):
            raise ValueError("active view must be 1, 2, or 3")
        self.active_view = view

    def pointer_over_panel(self, x: float, y: float) -> bool:
        width, _ = self.viewer.window.size
        return x >= width - self.panel_width - 10 and 10 <= y <= self.panel_height

    def _camera(self, name: str):
        return next(
            (
                camera
                for camera in self.viewer.cameras
                if camera.name == name or camera.name.endswith(f"_{name}")
            ),
            None,
        )

    def get_ui_windows(self):
        if self.ui_window is None:
            self.ui_window = (
                R.UIWindow()
                .Label("Auxiliary views")
                .Size(self.panel_width, self.panel_height)
            )
            self.ui_pictures = [R.UIPicture(), R.UIPicture()]

        width, _ = self.viewer.window.size
        self.ui_window.Pos(max(0, width - self.panel_width - 10), 10)
        self.ui_window.Label(f"Views | active={self.active_view}")
        self.ui_window.remove_children()
        labels = (
            "2: FRONT XZ (reserved)",
            "3: WRIST (observe only)",
        )
        for index, (name, label) in enumerate(zip(self.camera_names, labels)):
            camera = self._camera(name)
            active_index = index + 2
            prefix = "ACTIVE | " if self.active_view == active_index else ""
            self.ui_window.append(R.UIDisplayText().Text(prefix + label))
            if camera is None:
                self.ui_window.append(
                    R.UIDisplayText().Text(f"camera unavailable: {name}")
                )
                continue
            camera.take_picture()
            assert self.ui_pictures is not None
            self.ui_pictures[index].Size(320, 240)
            self.ui_pictures[index].Picture(camera._internal_renderer, "Color")
            self.ui_window.append(self.ui_pictures[index])
        return [self.ui_window]
