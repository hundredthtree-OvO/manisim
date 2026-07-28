from __future__ import annotations

from mani_sim.control.command import TaskSpaceCommand
from mani_sim.runtime.observation import RuntimeObservation


class MouseActionSource:
    """Expose the latest UI-produced command through the ActionSource API."""

    def __init__(self):
        self._command = TaskSpaceCommand.create(
            target_position=[0.0, 0.0, 0.0],
            gripper_position=1.0,
            timestamp=0.0,
            source="human",
            valid=False,
        )

    def update(self, command: TaskSpaceCommand) -> None:
        self._command = command

    def act(self, _observation: RuntimeObservation) -> TaskSpaceCommand:
        return self._command
