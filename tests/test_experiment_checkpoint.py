from __future__ import annotations

import numpy as np
import torch

from mani_sim.experiments.checkpoint import ExperimentCheckpoint


class _Agent:
    def __init__(self) -> None:
        self.controller_state = {"target": np.array([1.0, 2.0])}

    def get_controller_state(self):
        return self.controller_state

    def set_controller_state(self, state) -> None:
        self.controller_state = state


class _Environment:
    def __init__(self) -> None:
        self.value = np.array([3.0, 4.0])
        self.agent = _Agent()

    def get_state_dict(self):
        return {
            "actors": {"target": self.value.copy()},
            "controller": self.agent.get_controller_state(),
        }

    def set_state_dict(self, state) -> None:
        self.value = state["actors"]["target"].copy()


class _Component:
    def __init__(self) -> None:
        self.value = np.array([5.0])

    def get_experiment_state(self):
        return {"value": self.value.copy()}

    def set_experiment_state(self, state) -> None:
        self.value = state["value"].copy()


def test_experiment_checkpoint_restores_environment_controller_and_component():
    env = _Environment()
    component = _Component()
    checkpoint = ExperimentCheckpoint.capture(
        env,
        components={"component": component},
        user_state={"gripper_target": -1.0},
    )

    env.value[:] = 99.0
    env.agent.controller_state["target"][:] = 88.0
    component.value[:] = 77.0

    user_state = checkpoint.restore(
        env, components={"component": component}
    )

    assert np.allclose(env.value, [3.0, 4.0])
    assert np.allclose(
        env.agent.controller_state["target"], [1.0, 2.0]
    )
    assert np.allclose(component.value, [5.0])
    assert user_state == {"gripper_target": -1.0}


def test_experiment_checkpoint_restores_random_generators():
    env = _Environment()
    np.random.seed(7)
    torch.manual_seed(7)
    checkpoint = ExperimentCheckpoint.capture(env)
    expected_numpy = np.random.random(3)
    expected_torch = torch.rand(3)

    np.random.random(10)
    torch.rand(10)
    checkpoint.restore(env)

    assert np.allclose(np.random.random(3), expected_numpy)
    assert torch.allclose(torch.rand(3), expected_torch)
