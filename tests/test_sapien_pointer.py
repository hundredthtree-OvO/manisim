from types import SimpleNamespace

import numpy as np

from mani_sim.input.sapien_pointer import SapienPointer


def test_main_front_viewport_samples_xz_on_y_plane() -> None:
    model = np.eye(4)
    model[:3, :3] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    model[:3, 3] = [0.45, -1.0, 0.35]
    window = SimpleNamespace(
        mouse_position=(160.0, 120.0),
        size=(320, 240),
        get_camera_model_matrix=lambda: model,
        get_camera_projection_matrix=lambda: np.diag(
            [2.0, -2.0, -1.0, 1.0]
        ),
    )
    sample = SapienPointer(window, work_height_m=0.45).sample_axis_plane(
        plane_axis=1, plane_value=0.0
    )

    assert sample.valid
    np.testing.assert_allclose(sample.world_target, [0.45, 0.0, 0.35])
