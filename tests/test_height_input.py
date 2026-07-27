import pytest

from mani_sim.input.height import update_height


def test_u_and_j_style_rate_control() -> None:
    assert update_height(
        0.50,
        up_pressed=True,
        down_pressed=False,
        speed_mps=0.12,
        dt_seconds=0.05,
        bounds_m=(0.45, 0.65),
    ) == pytest.approx(0.506)
    assert update_height(
        0.50,
        up_pressed=False,
        down_pressed=True,
        speed_mps=0.12,
        dt_seconds=0.05,
        bounds_m=(0.45, 0.65),
    ) == pytest.approx(0.494)


def test_height_is_clamped_and_opposing_keys_cancel() -> None:
    assert update_height(
        0.65,
        up_pressed=True,
        down_pressed=False,
        speed_mps=1.0,
        dt_seconds=1.0,
        bounds_m=(0.45, 0.65),
    ) == 0.65
    assert update_height(
        0.50,
        up_pressed=True,
        down_pressed=True,
        speed_mps=1.0,
        dt_seconds=1.0,
        bounds_m=(0.45, 0.65),
    ) == 0.50
