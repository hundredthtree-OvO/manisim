from types import SimpleNamespace

from mani_sim.visualization.camera_views import AuxiliaryCameraPanel


def test_prefixed_maniskill_camera_names_are_found() -> None:
    panel = AuxiliaryCameraPanel()
    top = SimpleNamespace(name="scene-0_top_observer")
    front = SimpleNamespace(name="scene-0_front_observer")
    wrist = SimpleNamespace(name="scene-0_hand_camera")
    panel.viewer = SimpleNamespace(cameras=[top, front, wrist])
    assert panel._camera("top_observer") is top
    assert panel._camera("front_observer") is front
    assert panel._camera("hand_camera") is wrist


def test_auxiliary_preview_swaps_top_and_front_with_main_view() -> None:
    panel = AuxiliaryCameraPanel()
    assert panel.displayed_cameras()[0][0] == "front_observer"

    panel.set_active_view(2)
    assert panel.displayed_cameras()[0][0] == "top_observer"

    panel.set_active_view(1)
    assert panel.displayed_cameras()[0][0] == "front_observer"
