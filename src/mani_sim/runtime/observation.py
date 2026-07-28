from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np
from numpy.typing import ArrayLike

from mani_sim.runtime.contact_forces import ContactForceSample
from mani_sim.runtime.contact_forces import sample_contact_forces
from mani_sim.tasks.base import TaskObservation


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=np.float64)


def single_env_vector(value: Any) -> np.ndarray:
    array = to_numpy(value)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    return array


def single_env_bool(value: Any) -> bool:
    return bool(to_numpy(value).reshape(-1)[0])


@dataclass(frozen=True)
class RuntimeObservation:
    tcp_position: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    object_positions: dict[str, np.ndarray]
    grasped_objects: frozenset[str] = frozenset()
    contact_forces: ContactForceSample = field(
        default_factory=ContactForceSample
    )

    @classmethod
    def create(
        cls,
        *,
        tcp_position: ArrayLike,
        qpos: ArrayLike,
        qvel: ArrayLike,
        object_positions: Mapping[str, ArrayLike],
        grasped_objects: Iterable[str] = (),
        contact_forces: ContactForceSample | None = None,
    ) -> "RuntimeObservation":
        return cls(
            tcp_position=np.asarray(
                tcp_position, dtype=np.float64
            ).copy(),
            qpos=np.asarray(qpos, dtype=np.float64).copy(),
            qvel=np.asarray(qvel, dtype=np.float64).copy(),
            object_positions={
                name: np.asarray(position, dtype=np.float64).copy()
                for name, position in object_positions.items()
            },
            grasped_objects=frozenset(grasped_objects),
            contact_forces=contact_forces or ContactForceSample(),
        )

    @classmethod
    def empty(cls) -> "RuntimeObservation":
        return cls.create(
            tcp_position=np.zeros(3),
            qpos=np.zeros(0),
            qvel=np.zeros(0),
            object_positions={},
        )

    def task_observation(
        self, *, tcp_position: ArrayLike | None = None
    ) -> TaskObservation:
        return TaskObservation(
            tcp_position=(
                self.tcp_position
                if tcp_position is None
                else np.asarray(tcp_position, dtype=np.float64).copy()
            ),
            object_positions=self.object_positions,
            grasped_objects=self.grasped_objects,
        )


def capture_runtime_observation(
    base_env: Any,
    scenario: Any,
    *,
    contact_forces: ContactForceSample | None = None,
) -> RuntimeObservation:
    target = scenario.actor("target")
    grasped_objects = (
        {"target"}
        if target is not None
        and single_env_bool(base_env.agent.is_grasping(target))
        else set()
    )
    positions = {
        name: single_env_vector(actor.pose.p)
        for name, actor in scenario.actors.items()
    }
    return RuntimeObservation.create(
        tcp_position=single_env_vector(base_env.agent.tcp_pose.p),
        qpos=single_env_vector(base_env.agent.robot.get_qpos()),
        qvel=single_env_vector(base_env.agent.robot.get_qvel()),
        object_positions=positions,
        grasped_objects=grasped_objects,
        contact_forces=(
            contact_forces
            if contact_forces is not None
            else sample_contact_forces(base_env, scenario)
        ),
    )
