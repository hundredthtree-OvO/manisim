import numpy as np
import pytest
import torch

from mani_sim.visualization.force_monitor import (
    ForceChartRasterizer,
    ForceChartSurface,
)


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="force chart rendering needs the external Vulkan device",
    ),
]


def test_force_chart_texture_is_visible_to_dedicated_camera() -> None:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    env = gym.make(
        "Empty-v1",
        obs_mode="none",
        reward_mode="none",
        render_mode=None,
        control_mode="pd_ee_delta_pos",
        robot_uids="panda_wristcam",
        sim_backend="cpu",
    )
    try:
        env.reset(seed=0)
        rasterizer = ForceChartRasterizer(width=160, height=96)
        surface = ForceChartSurface(env.unwrapped, rasterizer)
        surface.update(
            ((0.0, 0.0, 0.0), (28.0, 12.0, 9.0)),
            threshold_n=8.0,
        )
        env.unwrapped.scene.update_render()
        surface.camera.take_picture()
        color = (
            surface.camera.get_picture("Color")[0][0]
            .detach()
            .cpu()
            .numpy()
        )

        assert color.shape[:2] == (96, 160)
        assert float(np.std(color[..., :3])) > 0.03
        left_half = color[:, :80, :3]
        right_half = color[:, 80:, :3]
        assert float(np.std(left_half)) > 0.03
        assert float(np.std(right_half)) > 0.03
    finally:
        env.close()
