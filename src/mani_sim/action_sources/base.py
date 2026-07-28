from __future__ import annotations

from typing import Protocol, runtime_checkable

from mani_sim.control.command import TaskSpaceCommand
from mani_sim.runtime.observation import RuntimeObservation


@runtime_checkable
class ActionSource(Protocol):
    def act(self, observation: RuntimeObservation) -> TaskSpaceCommand: ...
