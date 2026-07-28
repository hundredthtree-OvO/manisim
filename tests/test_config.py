from pathlib import Path

import pytest

from mani_sim.config import load_config


def test_demo_config_loads() -> None:
    config = load_config(Path("configs/demo0.yaml"))
    assert config.simulation.control_mode == "pd_ee_delta_pos"
    assert config.workspace.work_height_m == 0.45
    assert config.reset.policy == "hold_tcp"
    assert config.reset.pointer_rearm_pixels == 3.0
    assert config.reset.pointer_settle_steps == 2
    assert config.recording.path == "runs"
    assert config.collection.source == "mouse"
    assert config.cube_task.position_xy_m == (0.45, 0.0)
    assert config.cube_task.goal_position_xy_m == (0.30, 0.30)
    assert not config.collision_protection.obstacle_enabled


def test_invalid_workspace_bounds_fail(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("workspace:\n  x_bounds_m: [1, 0]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lower bound"):
        load_config(path)


def test_scripted_collection_config_loads() -> None:
    config = load_config(Path("configs/scripted_pick_place.yaml"))

    assert config.collection.source == "scripted_pick_place"
    assert config.collection.max_episode_steps == 800
