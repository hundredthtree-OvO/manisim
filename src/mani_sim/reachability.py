from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class ReachabilityProjection:
    target: NDArray[np.float64]
    projected: bool
    projection_distance_m: float
    method: str = "none"


class ReachabilityMap:
    """Discrete fixed-orientation, fixed-height Panda TCP reachability map."""

    def __init__(
        self,
        points_by_height: dict[float, NDArray[np.float64]],
        maximum_height_delta_m: float,
        grid_step_m: float = 0.0,
        refined_samples_by_height: (
            dict[float, NDArray[np.float64]] | None
        ) = None,
        boundary_grid_step_m: float = 0.0,
    ):
        if not points_by_height:
            raise ValueError("reachability map has no sampled heights")
        self.points_by_height = points_by_height
        self.maximum_height_delta_m = maximum_height_delta_m
        self.grid_step_m = grid_step_m
        self.boundary_grid_step_m = boundary_grid_step_m
        self.refined_samples_by_height = refined_samples_by_height or {}
        self.discarded_isolated_points = 0
        if grid_step_m > 0:
            self._keep_largest_components()
        self._build_lookup_tables()

    def _build_lookup_tables(self) -> None:
        self._coarse_reachable_keys = {
            height: {
                (
                    round(point[0] / self.grid_step_m),
                    round(point[1] / self.grid_step_m),
                )
                for point in points
            }
            for height, points in self.points_by_height.items()
            if self.grid_step_m > 0
        }
        self._refined_cells: dict[float, dict[tuple[int, int], bool]] = {}
        if self.boundary_grid_step_m > 0:
            for height, samples in self.refined_samples_by_height.items():
                self._refined_cells[height] = {
                    (
                        round(sample[0] / self.boundary_grid_step_m),
                        round(sample[1] / self.boundary_grid_step_m),
                    ): bool(sample[2])
                    for sample in samples
                }

    def _keep_largest_components(self) -> None:
        for height, points in tuple(self.points_by_height.items()):
            keys = {
                (round(point[0] / self.grid_step_m), round(point[1] / self.grid_step_m))
                for point in points
            }
            components: list[set[tuple[int, int]]] = []
            unseen = set(keys)
            while unseen:
                seed = unseen.pop()
                component = {seed}
                frontier = [seed]
                while frontier:
                    x, y = frontier.pop()
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            if dx == 0 and dy == 0:
                                continue
                            neighbor = (x + dx, y + dy)
                            if neighbor in unseen:
                                unseen.remove(neighbor)
                                component.add(neighbor)
                                frontier.append(neighbor)
                components.append(component)
            largest = max(components, key=len)
            filtered = np.asarray(
                [
                    point
                    for point in points
                    if (
                        round(point[0] / self.grid_step_m),
                        round(point[1] / self.grid_step_m),
                    )
                    in largest
                ],
                dtype=np.float64,
            )
            self.discarded_isolated_points += len(points) - len(filtered)
            self.points_by_height[height] = filtered

    @classmethod
    def load(
        cls, path: str | Path, maximum_height_delta_m: float = 0.001
    ) -> "ReachabilityMap":
        with Path(path).open(encoding="utf-8") as stream:
            raw = json.load(stream)
        points_by_height = {}
        refined_samples_by_height = {}
        for layer in raw["layers"]:
            points = np.asarray(layer["reachable_points_xy"], dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 2:
                raise ValueError("reachable_points_xy must have shape (N, 2)")
            if len(points):
                height = float(layer["height_m"])
                points_by_height[height] = points
                refined = np.asarray(
                    layer.get("refined_boundary_samples_xy_reachable", []),
                    dtype=np.float64,
                )
                if len(refined):
                    if refined.ndim != 2 or refined.shape[1] != 3:
                        raise ValueError(
                            "refined boundary samples must have shape (N, 3)"
                        )
                    refined_samples_by_height[height] = refined
        return cls(
            points_by_height,
            maximum_height_delta_m,
            grid_step_m=float(raw.get("grid_step_m", 0.0)),
            refined_samples_by_height=refined_samples_by_height,
            boundary_grid_step_m=float(
                raw.get("boundary_grid_step_m", 0.0)
            ),
        )

    @property
    def heights(self) -> tuple[float, ...]:
        return tuple(sorted(self.points_by_height))

    def _layer(self, height_m: float) -> tuple[float, NDArray[np.float64]]:
        height = min(self.heights, key=lambda value: abs(value - height_m))
        if abs(height - height_m) > self.maximum_height_delta_m:
            raise ValueError(
                f"no calibrated layer near z={height_m:.3f}; available={self.heights}"
            )
        return height, self.points_by_height[height]

    def _height_bracket(self, height_m: float) -> tuple[float, float, float]:
        heights = self.heights
        if (
            height_m < heights[0] - self.maximum_height_delta_m
            or height_m > heights[-1] + self.maximum_height_delta_m
        ):
            raise ValueError(
                f"z={height_m:.3f} is outside calibrated range "
                f"[{heights[0]:.3f}, {heights[-1]:.3f}]"
            )
        height_m = float(np.clip(height_m, heights[0], heights[-1]))
        for height in heights:
            if abs(height - height_m) <= self.maximum_height_delta_m:
                return height, height, 0.0
        lower = max(height for height in heights if height < height_m)
        upper = min(height for height in heights if height > height_m)
        alpha = (height_m - lower) / (upper - lower)
        return lower, upper, alpha

    def project(self, target: ArrayLike) -> ReachabilityProjection:
        target_array = np.asarray(target, dtype=np.float64)
        if target_array.shape != (3,):
            raise ValueError("target must have shape (3,)")
        height, points = self._layer(float(target_array[2]))
        fine = self.refined_samples_by_height.get(height)
        if fine is not None:
            fine_reachable = fine[fine[:, 2] > 0.5, :2]
            points = np.vstack([points, fine_reachable])
        offsets = points - target_array[:2]
        index = int(np.argmin(np.einsum("ij,ij->i", offsets, offsets)))
        projected = np.array([points[index, 0], points[index, 1], height])
        distance = float(np.linalg.norm(projected - target_array))
        if self._is_reachable(height, target_array[:2], points):
            return ReachabilityProjection(
                target=target_array.copy(),
                projected=False,
                projection_distance_m=0.0,
                method="none",
            )
        return ReachabilityProjection(
            target=projected,
            projected=distance > 1e-9,
            projection_distance_m=distance,
            method="nearest",
        )

    def _is_reachable(
        self,
        height: float,
        point_xy: NDArray[np.float64],
        points: NDArray[np.float64] | None = None,
    ) -> bool:
        refined = self._refined_cells.get(height)
        if refined and self.boundary_grid_step_m > 0:
            fine_key = (
                round(point_xy[0] / self.boundary_grid_step_m),
                round(point_xy[1] / self.boundary_grid_step_m),
            )
            if fine_key in refined:
                return refined[fine_key]
        if self.grid_step_m > 0:
            coarse_key = (
                round(point_xy[0] / self.grid_step_m),
                round(point_xy[1] / self.grid_step_m),
            )
            return coarse_key in self._coarse_reachable_keys[height]
        assert points is not None
        return bool(np.min(np.sum((points - point_xy) ** 2, axis=1)) <= 1e-18)

    def is_reachable(self, target: ArrayLike) -> bool:
        target_array = np.asarray(target, dtype=np.float64)
        if target_array.shape != (3,):
            raise ValueError("target must have shape (3,)")
        height, points = self._layer(float(target_array[2]))
        return self._is_reachable(height, target_array[:2], points)

    def project_along_ray(
        self, origin: ArrayLike, target: ArrayLike
    ) -> ReachabilityProjection:
        """Project to the first reachable boundary along ``origin -> target``."""

        origin_array = np.asarray(origin, dtype=np.float64)
        target_array = np.asarray(target, dtype=np.float64)
        if origin_array.shape != (3,) or target_array.shape != (3,):
            raise ValueError("origin and target must have shape (3,)")
        height, points = self._layer(float(target_array[2]))
        target_array = target_array.copy()
        target_array[2] = height
        if self._is_reachable(height, target_array[:2], points):
            return ReachabilityProjection(target_array, False, 0.0, "none")

        direction = target_array[:2] - origin_array[:2]
        length = float(np.linalg.norm(direction))
        if length < 1e-12 or self.grid_step_m <= 0:
            nearest = self.project(target_array)
            return ReachabilityProjection(
                nearest.target,
                nearest.projected,
                nearest.projection_distance_m,
                "nearest_fallback",
            )

        effective_step = (
            self.boundary_grid_step_m
            if self.boundary_grid_step_m > 0
            else self.grid_step_m
        )
        sample_step = effective_step / 2.0
        sample_count = max(2, int(np.ceil(length / sample_step)) + 1)
        fractions = np.linspace(0.0, 1.0, sample_count)
        was_reachable = False
        last_reachable_fraction: float | None = None
        first_unreachable_fraction: float | None = None
        for fraction in fractions:
            point = origin_array[:2] + fraction * direction
            reachable = self._is_reachable(height, point, points)
            if reachable:
                was_reachable = True
                last_reachable_fraction = float(fraction)
            elif was_reachable:
                first_unreachable_fraction = float(fraction)
                break

        if (
            last_reachable_fraction is None
            or first_unreachable_fraction is None
        ):
            nearest = self.project(target_array)
            return ReachabilityProjection(
                nearest.target,
                nearest.projected,
                nearest.projection_distance_m,
                "nearest_fallback",
            )

        low = last_reachable_fraction
        high = first_unreachable_fraction
        for _ in range(12):
            middle = (low + high) / 2.0
            point = origin_array[:2] + middle * direction
            if self._is_reachable(height, point, points):
                low = middle
            else:
                high = middle
        boundary_xy = origin_array[:2] + low * direction
        boundary = np.array([boundary_xy[0], boundary_xy[1], height])
        return ReachabilityProjection(
            boundary,
            True,
            float(np.linalg.norm(boundary - target_array)),
            "ray_boundary",
        )

    def project_continuous(
        self, origin: ArrayLike, target: ArrayLike
    ) -> ReachabilityProjection:
        """Project with linear interpolation between neighboring height layers."""

        origin_array = np.asarray(origin, dtype=np.float64)
        target_array = np.asarray(target, dtype=np.float64)
        lower, upper, alpha = self._height_bracket(float(target_array[2]))
        if lower == upper:
            return self.project_along_ray(origin_array, target_array)

        lower_origin = origin_array.copy()
        lower_origin[2] = lower
        lower_target = target_array.copy()
        lower_target[2] = lower
        upper_origin = origin_array.copy()
        upper_origin[2] = upper
        upper_target = target_array.copy()
        upper_target[2] = upper
        lower_projection = self.project_along_ray(lower_origin, lower_target)
        upper_projection = self.project_along_ray(upper_origin, upper_target)
        xy = (
            (1.0 - alpha) * lower_projection.target[:2]
            + alpha * upper_projection.target[:2]
        )
        projected_target = np.array([xy[0], xy[1], target_array[2]])
        distance = float(np.linalg.norm(projected_target - target_array))
        projected = lower_projection.projected or upper_projection.projected
        method = "none" if not projected else "layer_interpolated"
        return ReachabilityProjection(
            projected_target, projected, distance, method
        )

    def limit_projected_target_step(
        self,
        previous_target: ArrayLike,
        candidate_target: ArrayLike,
        maximum_step_m: float,
    ) -> tuple[NDArray[np.float64], bool]:
        previous = np.asarray(previous_target, dtype=np.float64).copy()
        candidate = np.asarray(candidate_target, dtype=np.float64)
        previous[2] = candidate[2]
        delta = candidate - previous
        distance = float(np.linalg.norm(delta))
        if distance <= maximum_step_m:
            return candidate.copy(), False
        limited = previous + delta * (maximum_step_m / distance)
        exact_layer = any(
            abs(limited[2] - height) <= self.maximum_height_delta_m
            for height in self.heights
        )
        if not exact_layer or self.is_reachable(limited):
            return limited, True
        return previous, True
