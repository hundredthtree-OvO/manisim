from types import SimpleNamespace

from mani_sim.visualization.status_panel import RuntimeStatus, RuntimeStatusPanel


def test_status_panel_formats_global_and_task_fields() -> None:
    panel = RuntimeStatusPanel()
    panel.update(
        RuntimeStatus.create(
            active_view=2,
            tcp_position=[0.4, 0.1, 0.3],
            contact_force_n=1.25,
            contact_threshold_n=8.0,
            emergency_stop=False,
            recording=True,
            episode_seed=7,
            randomized_scene=True,
            grip_force_n=3.0,
            left_finger_force_n=3.1,
            right_finger_force_n=3.0,
            object_force_n=9.0,
            task_fields=(
                ("phase", "lifted"),
                ("grasped", "yes"),
            ),
        )
    )

    lines = panel.lines()
    assert "ACTIVE VIEW: FRONT XZ" in lines
    assert "UNINTENDED CONTACT: 1.25 / 8.00 N" in lines
    assert "SAFETY: OK" in lines
    assert "EPISODE SEED: 7" in lines
    assert "SCENE: RANDOMIZED" in lines
    assert any("F L/R: 3.10/3.00 N" in line for line in lines)
    assert "PHASE: lifted" in lines


def test_status_panel_blocks_pointer_in_bottom_right() -> None:
    panel = RuntimeStatusPanel(panel_width=360, panel_height=260)
    panel.viewer = SimpleNamespace(
        window=SimpleNamespace(size=(1200, 800))
    )
    assert panel.pointer_over_panel(1000, 700)
    assert not panel.pointer_over_panel(500, 400)
