from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class ExperimentStateful(Protocol):
    def get_experiment_state(self) -> Any: ...

    def set_experiment_state(self, state: Any) -> None: ...


def _copied(value: Any) -> Any:
    return copy.deepcopy(value)


@dataclass(frozen=True)
class ExperimentCheckpoint:
    """Complete restorable state for a fixed-dynamics branch experiment."""

    environment_state: dict[str, Any]
    controller_state: Any
    component_states: dict[str, Any]
    user_state: dict[str, Any]
    python_random_state: object
    numpy_random_state: tuple[Any, ...]
    torch_random_state: torch.Tensor
    torch_cuda_random_states: tuple[torch.Tensor, ...]

    @classmethod
    def capture(
        cls,
        base_env: Any,
        *,
        components: Mapping[str, ExperimentStateful] | None = None,
        user_state: Mapping[str, Any] | None = None,
    ) -> "ExperimentCheckpoint":
        controller_state = base_env.agent.get_controller_state()
        return cls(
            environment_state=_copied(base_env.get_state_dict()),
            controller_state=_copied(controller_state),
            component_states={
                name: _copied(component.get_experiment_state())
                for name, component in (components or {}).items()
            },
            user_state=_copied(dict(user_state or {})),
            python_random_state=_copied(random.getstate()),
            numpy_random_state=_copied(np.random.get_state()),
            torch_random_state=torch.random.get_rng_state().clone(),
            torch_cuda_random_states=(
                tuple(state.clone() for state in torch.cuda.get_rng_state_all())
                if torch.cuda.is_available()
                else ()
            ),
        )

    def restore(
        self,
        base_env: Any,
        *,
        components: Mapping[str, ExperimentStateful] | None = None,
    ) -> dict[str, Any]:
        supplied = components or {}
        missing = set(self.component_states) - set(supplied)
        if missing:
            raise KeyError(
                "missing checkpoint components: "
                + ", ".join(sorted(missing))
            )

        base_env.set_state_dict(_copied(self.environment_state))
        base_env.agent.set_controller_state(
            _copied(self.controller_state)
        )
        for name, state in self.component_states.items():
            supplied[name].set_experiment_state(_copied(state))

        random.setstate(_copied(self.python_random_state))
        np.random.set_state(_copied(self.numpy_random_state))
        torch.random.set_rng_state(self.torch_random_state.clone())
        if self.torch_cuda_random_states and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(
                [state.clone() for state in self.torch_cuda_random_states]
            )
        return _copied(self.user_state)
