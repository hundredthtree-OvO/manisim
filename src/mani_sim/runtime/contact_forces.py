from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mani_sim.environments.scenario import Scenario


def _vector(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    return array


@dataclass(frozen=True)
class ContactForceSample:
    left_finger_world_n: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    right_finger_world_n: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    object_net_world_n: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    unintended_by_pair_world_n: dict[str, np.ndarray] = field(
        default_factory=dict
    )

    @property
    def left_finger_n(self) -> float:
        return float(np.linalg.norm(self.left_finger_world_n))

    @property
    def right_finger_n(self) -> float:
        return float(np.linalg.norm(self.right_finger_world_n))

    @property
    def grip_n(self) -> float:
        return min(self.left_finger_n, self.right_finger_n)

    @property
    def object_net_n(self) -> float:
        return float(np.linalg.norm(self.object_net_world_n))

    @property
    def maximum_unintended_n(self) -> float:
        return max(
            (
                float(np.linalg.norm(force))
                for force in self.unintended_by_pair_world_n.values()
            ),
            default=0.0,
        )


def _named_link(robot: Any, suffix: str) -> Any | None:
    return next(
        (link for link in robot.links if link.name.endswith(suffix)),
        None,
    )


def sample_contact_forces(
    base_env: Any,
    scenario: Scenario,
) -> ContactForceSample:
    """Sample intended task contacts and unintended scene contacts."""

    scene = base_env.scene
    robot = base_env.agent.robot
    target = scenario.actor("target")
    left = getattr(base_env.agent, "finger1_link", None) or _named_link(
        robot, "panda_leftfinger"
    )
    right = getattr(base_env.agent, "finger2_link", None) or _named_link(
        robot, "panda_rightfinger"
    )

    def pairwise(first: Any | None, second: Any | None) -> np.ndarray:
        if first is None or second is None:
            return np.zeros(3, dtype=np.float64)
        return _vector(scene.get_pairwise_contact_forces(first, second))

    object_net = (
        _vector(target.get_net_contact_forces())
        if target is not None and hasattr(target, "get_net_contact_forces")
        else np.zeros(3, dtype=np.float64)
    )

    unintended: dict[str, np.ndarray] = {}
    collision_actors = {"ground": base_env.ground}
    collision_actors.update(
        {
            name: actor
            for name, actor in scenario.actors.items()
            if name == "obstacle"
        }
    )
    for link in robot.links:
        if link.name == "panda_link0":
            continue
        for actor_name, actor in collision_actors.items():
            force = pairwise(actor, link)
            if np.any(force):
                unintended[f"{actor_name}/{link.name}"] = force

    return ContactForceSample(
        # Match Panda.is_grasping's verified query order.
        left_finger_world_n=pairwise(left, target),
        right_finger_world_n=pairwise(right, target),
        object_net_world_n=object_net,
        unintended_by_pair_world_n=unintended,
    )
