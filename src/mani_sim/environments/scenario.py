from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sapien

from mani_sim.assets import BoxObjectSpec, ObjectFactory
from mani_sim.config import AppConfig
from mani_sim.control.scene_collision_guard import AxisAlignedBox


@dataclass(frozen=True)
class PositionRandomization:
    target_x_bounds_m: tuple[float, float]
    target_y_bounds_m: tuple[float, float]
    goal_x_bounds_m: tuple[float, float]
    goal_y_bounds_m: tuple[float, float]
    minimum_distance_m: float

    def sample(
        self, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        for _ in range(1000):
            target = np.array(
                [
                    rng.uniform(*self.target_x_bounds_m),
                    rng.uniform(*self.target_y_bounds_m),
                ]
            )
            goal = np.array(
                [
                    rng.uniform(*self.goal_x_bounds_m),
                    rng.uniform(*self.goal_y_bounds_m),
                ]
            )
            if np.linalg.norm(target - goal) >= self.minimum_distance_m:
                return target, goal
        raise ValueError(
            "position randomization bounds cannot satisfy minimum distance"
        )


@dataclass
class Scenario:
    """Scene entity registry and reset boundary, independent of task logic."""

    actors: dict[str, Any] = field(default_factory=dict)
    initial_positions: dict[str, np.ndarray] = field(default_factory=dict)
    obstacles: tuple[AxisAlignedBox, ...] = ()
    position_randomization: PositionRandomization | None = None

    def actor(self, name: str) -> Any | None:
        return self.actors.get(name)

    def initial_position(self, name: str) -> np.ndarray | None:
        position = self.initial_positions.get(name)
        return None if position is None else position.copy()

    def reset(self, rng: np.random.Generator | None = None) -> None:
        if self.position_randomization is not None and rng is not None:
            target_xy, goal_xy = self.position_randomization.sample(rng)
            self.initial_positions["target"][:2] = target_xy
            self.initial_positions["goal"][:2] = goal_xy
        for name, position in self.initial_positions.items():
            actor = self.actors.get(name)
            if actor is not None:
                actor.set_pose(sapien.Pose(p=position))

    # Compatibility properties for the first scaffold API.
    @property
    def cube(self) -> Any | None:
        return self.actor("target")

    @property
    def goal_site(self) -> Any | None:
        return self.actor("goal")

    @property
    def obstacle(self) -> Any | None:
        return self.actor("obstacle")

    @property
    def cube_initial_position(self) -> np.ndarray | None:
        return self.initial_position("target")

    @property
    def goal_position(self) -> np.ndarray | None:
        return self.initial_position("goal")


def build_scenario(base_env: Any, config: AppConfig) -> Scenario:
    factory = ObjectFactory(base_env.scene)
    scene = Scenario()
    task = config.cube_task

    if task.enabled:
        cube_position = (
            task.position_xy_m[0],
            task.position_xy_m[1],
            task.size_m / 2,
        )
        scene.actors["target"] = factory.build_box(
            BoxObjectSpec(
                name="pick_cube",
                size_m=(task.size_m,) * 3,
                position_m=cube_position,
                color_rgba=(0.9, 0.12, 0.08, 1.0),
            )
        )
        scene.initial_positions["target"] = np.asarray(
            cube_position, dtype=np.float64
        )

        goal_position = (
            task.goal_position_xy_m[0],
            task.goal_position_xy_m[1],
            0.001,
        )
        scene.actors["goal"] = factory.build_box(
            BoxObjectSpec(
                name="place_goal",
                size_m=(
                    task.goal_tolerance_m * 2,
                    task.goal_tolerance_m * 2,
                    0.002,
                ),
                position_m=goal_position,
                color_rgba=(0.1, 0.85, 0.2, 0.45),
                body_type="kinematic",
                add_collision=False,
            )
        )
        scene.initial_positions["goal"] = np.asarray(
            goal_position, dtype=np.float64
        )
        if task.randomize_positions:
            scene.position_randomization = PositionRandomization(
                target_x_bounds_m=task.target_x_bounds_m,
                target_y_bounds_m=task.target_y_bounds_m,
                goal_x_bounds_m=task.goal_x_bounds_m,
                goal_y_bounds_m=task.goal_y_bounds_m,
                minimum_distance_m=task.minimum_start_goal_distance_m,
            )

    collision = config.collision_protection
    if collision.enabled and collision.obstacle_enabled:
        scene.actors["obstacle"] = factory.build_box(
            BoxObjectSpec(
                name="static_obstacle",
                size_m=collision.obstacle_size_m,
                position_m=collision.obstacle_center_m,
                color_rgba=(0.25, 0.30, 0.38, 1.0),
                body_type="static",
            )
        )
        scene.initial_positions["obstacle"] = np.asarray(
            collision.obstacle_center_m, dtype=np.float64
        )
        scene.obstacles = (
            AxisAlignedBox.create(
                collision.obstacle_center_m,
                collision.obstacle_size_m,
                "static_obstacle",
            ),
        )

    return scene
