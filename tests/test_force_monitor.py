import numpy as np

from mani_sim.visualization.force_monitor import (
    ForceChartRasterizer,
    ForceChartSurface,
)


def test_force_chart_rasterizer_draws_three_curves_and_threshold() -> None:
    rasterizer = ForceChartRasterizer(width=120, height=80)
    image = rasterizer.render(
        ((0.0, 0.0, 0.0), (28.0, 0.64, 9.0)),
        threshold_n=8.0,
    )

    assert image.shape == (80, 120, 4)
    colors = image.reshape(-1, 4)
    for color in rasterizer.COLORS:
        assert np.any(np.all(colors == color, axis=1))
    orange = np.array([245, 170, 45, 255], dtype=np.uint8)
    assert np.any(np.all(colors == orange, axis=1))


def test_chart_texture_is_transposed_for_yz_plane_uvs() -> None:
    image = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    oriented = ForceChartSurface._orient_for_plane(image)

    assert oriented.shape == (3, 2, 4)
    assert np.array_equal(oriented[2, 1], image[1, 2])
    assert oriented.flags.c_contiguous
