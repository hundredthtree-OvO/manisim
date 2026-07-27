import json
from pathlib import Path

import numpy as np
import pytest

from mani_sim.reachability import ReachabilityMap


def write_map(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "layers": [
                    {
                        "height_m": 0.45,
                        "reachable_points_xy": [[0.2, 0.0], [0.4, 0.1]],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_projects_to_nearest_sample(tmp_path: Path) -> None:
    path = tmp_path / "map.json"
    write_map(path)
    reachability = ReachabilityMap.load(path)
    result = reachability.project([0.38, 0.08, 0.45])
    np.testing.assert_allclose(result.target, [0.4, 0.1, 0.45])
    assert result.projected


def test_rejects_uncalibrated_height(tmp_path: Path) -> None:
    path = tmp_path / "map.json"
    write_map(path)
    reachability = ReachabilityMap.load(path)
    with pytest.raises(ValueError, match="no calibrated layer"):
        reachability.project([0.2, 0.0, 0.55])


def test_keeps_largest_connected_component() -> None:
    reachability = ReachabilityMap(
        {
            0.45: np.array(
                [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [2.0, 2.0]]
            )
        },
        maximum_height_delta_m=0.001,
        grid_step_m=0.1,
    )
    assert len(reachability.points_by_height[0.45]) == 3
    assert reachability.discarded_isolated_points == 1


def test_ray_projection_preserves_direction_at_boundary() -> None:
    xs, ys = np.meshgrid(
        np.arange(0.0, 1.01, 0.1), np.arange(0.0, 1.01, 0.1)
    )
    points = np.column_stack([xs.ravel(), ys.ravel()])
    reachability = ReachabilityMap(
        {0.45: points},
        maximum_height_delta_m=0.001,
        grid_step_m=0.1,
    )
    result = reachability.project_along_ray(
        [0.2, 0.73, 0.45], [2.0, 0.73, 0.45]
    )
    assert result.method == "ray_boundary"
    assert result.target[0] > 1.0
    assert result.target[0] < 1.1
    assert result.target[1] == pytest.approx(0.73)


def test_fine_boundary_sample_overrides_coarse_cell() -> None:
    reachability = ReachabilityMap(
        {0.45: np.array([[0.0, 0.0], [0.1, 0.0]])},
        maximum_height_delta_m=0.001,
        grid_step_m=0.1,
        refined_samples_by_height={
            0.45: np.array([[0.05, 0.0, 0.0], [0.04, 0.0, 1.0]])
        },
        boundary_grid_step_m=0.01,
    )
    assert not reachability.is_reachable([0.05, 0.0, 0.45])
    assert reachability.is_reachable([0.04, 0.0, 0.45])


def test_projected_target_step_is_limited() -> None:
    xs, ys = np.meshgrid(
        np.arange(0.0, 1.01, 0.1), np.arange(0.0, 1.01, 0.1)
    )
    reachability = ReachabilityMap(
        {0.45: np.column_stack([xs.ravel(), ys.ravel()])},
        maximum_height_delta_m=0.001,
        grid_step_m=0.1,
    )
    target, suppressed = reachability.limit_projected_target_step(
        [0.2, 0.2, 0.45], [0.8, 0.2, 0.45], maximum_step_m=0.1
    )
    assert suppressed
    np.testing.assert_allclose(target, [0.3, 0.2, 0.45])


def test_boundary_interpolates_between_height_layers() -> None:
    lower = np.array([[x, 0.0] for x in np.arange(0.0, 1.01, 0.1)])
    upper = np.array([[x, 0.0] for x in np.arange(0.0, 0.61, 0.1)])
    reachability = ReachabilityMap(
        {0.45: lower, 0.55: upper},
        maximum_height_delta_m=0.001,
        grid_step_m=0.1,
    )
    result = reachability.project_continuous(
        [0.1, 0.0, 0.50], [2.0, 0.0, 0.50]
    )
    assert result.method == "layer_interpolated"
    assert result.target[0] == pytest.approx(0.85, abs=0.02)
    assert result.target[2] == pytest.approx(0.50)


def test_height_roundoff_is_clamped_to_calibrated_endpoint() -> None:
    reachability = ReachabilityMap(
        {0.45: np.array([[0.2, 0.0]]), 0.55: np.array([[0.2, 0.0]])},
        maximum_height_delta_m=0.001,
        grid_step_m=0.1,
    )
    result = reachability.project_continuous(
        [0.2, 0.0, 0.44999999], [0.2, 0.0, 0.44999999]
    )
    assert not result.projected
