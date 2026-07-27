from types import SimpleNamespace

from mani_sim.visualization.camera_views import AuxiliaryCameraPanel


def test_prefixed_maniskill_camera_names_are_found() -> None:
    panel = AuxiliaryCameraPanel()
    front = SimpleNamespace(name="scene-0_front_observer")
    wrist = SimpleNamespace(name="scene-0_hand_camera")
    panel.viewer = SimpleNamespace(cameras=[front, wrist])
    assert panel._camera("front_observer") is front
    assert panel._camera("hand_camera") is wrist
