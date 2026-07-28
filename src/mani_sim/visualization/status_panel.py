from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike
from sapien import internal_renderer as R
from sapien.utils.viewer.plugin import Plugin


@dataclass(frozen=True)
class RuntimeStatus:
    active_view: int
    tcp_position: np.ndarray
    contact_force_n: float
    contact_threshold_n: float
    emergency_stop: bool
    recording: bool
    grip_force_n: float = 0.0
    left_finger_force_n: float = 0.0
    right_finger_force_n: float = 0.0
    object_force_n: float = 0.0
    task_fields: tuple[tuple[str, str], ...] = ()

    @classmethod
    def create(
        cls,
        *,
        active_view: int,
        tcp_position: ArrayLike,
        contact_force_n: float,
        contact_threshold_n: float,
        emergency_stop: bool,
        recording: bool,
        grip_force_n: float = 0.0,
        left_finger_force_n: float = 0.0,
        right_finger_force_n: float = 0.0,
        object_force_n: float = 0.0,
        task_fields: tuple[tuple[str, str], ...] = (),
    ) -> "RuntimeStatus":
        return cls(
            active_view=active_view,
            tcp_position=np.asarray(tcp_position, dtype=np.float64).copy(),
            contact_force_n=contact_force_n,
            contact_threshold_n=contact_threshold_n,
            emergency_stop=emergency_stop,
            recording=recording,
            grip_force_n=grip_force_n,
            left_finger_force_n=left_finger_force_n,
            right_finger_force_n=right_finger_force_n,
            object_force_n=object_force_n,
            task_fields=task_fields,
        )


class RuntimeStatusPanel(Plugin):
    """Task-independent runtime and task status in the SAPIEN window."""

    VIEW_LABELS = {
        1: ("TOP XY", "mouse X/Y | U/J Z"),
        2: ("FRONT XZ", "mouse X/Z | U/J Y"),
        3: ("WRIST", "observe only"),
    }

    def __init__(
        self,
        panel_width: int = 360,
        panel_height: int = 260,
    ):
        self.panel_width = panel_width
        self.panel_height = panel_height
        self.ui_window = None
        self.status = RuntimeStatus.create(
            active_view=1,
            tcp_position=[0.0, 0.0, 0.0],
            contact_force_n=0.0,
            contact_threshold_n=0.0,
            emergency_stop=False,
            recording=False,
        )

    def update(self, status: RuntimeStatus) -> None:
        self.status = status

    def panel_rect(self) -> tuple[float, float, float, float]:
        width, height = self.viewer.window.size
        left = max(0.0, width - self.panel_width - 10.0)
        top = max(10.0, height - self.panel_height - 10.0)
        return left, top, left + self.panel_width, top + self.panel_height

    def pointer_over_panel(self, x: float, y: float) -> bool:
        left, top, right, bottom = self.panel_rect()
        return left <= x <= right and top <= y <= bottom

    def lines(self) -> tuple[str, ...]:
        status = self.status
        view, mapping = self.VIEW_LABELS[status.active_view]
        tcp = status.tcp_position
        safety = "STOP" if status.emergency_stop else "OK"
        lines = [
            f"ACTIVE VIEW: {view}",
            f"CONTROL: {mapping}",
            f"TCP: ({tcp[0]:.3f}, {tcp[1]:.3f}, {tcp[2]:.3f}) m",
            (
                f"UNINTENDED CONTACT: {status.contact_force_n:.2f} / "
                f"{status.contact_threshold_n:.2f} N"
            ),
            f"SAFETY: {safety}",
            f"RECORDING: {'ON' if status.recording else 'OFF'}",
            "--- FORCE ---",
            f"F L/R: {status.left_finger_force_n:.2f}/"
            f"{status.right_finger_force_n:.2f} N | "
            f"O: {status.object_force_n:.2f} N | "
            f"X: {status.contact_force_n:.2f} N",
        ]
        if status.task_fields:
            lines.append("--- TASK ---")
            lines.extend(
                f"{name.upper()}: {value}"
                for name, value in status.task_fields
            )
        return tuple(lines)

    def get_ui_windows(self):
        if self.ui_window is None:
            self.ui_window = (
                R.UIWindow()
                .Label("Runtime status")
                .Size(self.panel_width, self.panel_height)
            )
        left, top, _, _ = self.panel_rect()
        self.ui_window.Pos(left, top)
        self.ui_window.remove_children()
        for line in self.lines():
            self.ui_window.append(R.UIDisplayText().Text(line))
        return [self.ui_window]
