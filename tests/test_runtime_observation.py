import numpy as np

from mani_sim.runtime.contact_forces import ContactForceSample
from mani_sim.runtime.observation import RuntimeObservation


def test_runtime_observation_converts_to_task_observation() -> None:
    observation = RuntimeObservation.create(
        tcp_position=[0.4, 0.1, 0.2],
        qpos=np.arange(9),
        qvel=np.arange(9) * 0.1,
        object_positions={"target": [0.45, 0.0, 0.02]},
        grasped_objects={"target"},
        contact_forces=ContactForceSample(),
    )

    task_observation = observation.task_observation()

    assert np.allclose(task_observation.tcp_position, [0.4, 0.1, 0.2])
    assert np.allclose(
        task_observation.object_positions["target"], [0.45, 0.0, 0.02]
    )
    assert task_observation.grasped_objects == frozenset({"target"})


def test_runtime_observation_owns_mutable_inputs() -> None:
    target = np.array([0.45, 0.0, 0.02])
    observation = RuntimeObservation.create(
        tcp_position=[0.4, 0.1, 0.2],
        qpos=[0.0],
        qvel=[0.0],
        object_positions={"target": target},
    )
    target[:] = 1.0

    assert np.allclose(
        observation.object_positions["target"], [0.45, 0.0, 0.02]
    )
