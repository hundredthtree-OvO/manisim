import pytest

from mani_sim.experiments.pre_lift import (
    PreLiftIntervention,
    default_pre_lift_interventions,
)
from mani_sim.experiments.run_pre_lift import (
    boundary_anchor_offsets_m,
    default_anchor_offsets_m,
)


def test_default_pre_lift_interventions_cover_one_factor_edits() -> None:
    interventions = default_pre_lift_interventions()

    assert [item.name for item in interventions] == [
        "base",
        "wait_5",
        "slow",
        "fast",
        "x_plus",
        "x_minus",
        "hold_m050",
        "hold_m025",
        "hold_000",
        "hold_p025",
        "hold_p050",
        "hold_p100",
        "hold_p250",
    ]
    assert interventions[1].dwell_steps == 5
    assert interventions[2].lift_scale == 0.5
    assert interventions[3].lift_scale == 1.5
    assert interventions[4].xy_offset_m == (0.003, 0.0)
    assert [
        item.gripper_position for item in interventions[6:]
    ] == [-0.5, -0.25, 0.0, 0.025, 0.05, 0.1, 0.25]


def test_pre_lift_intervention_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="dwell_steps"):
        PreLiftIntervention("bad", dwell_steps=-1)
    with pytest.raises(ValueError, match="lift_scale"):
        PreLiftIntervention("bad", lift_scale=0.0)
    with pytest.raises(ValueError, match="gripper_position"):
        PreLiftIntervention("bad", gripper_position=1.1)


def test_default_anchor_offsets_scan_both_axes_once() -> None:
    offsets = default_anchor_offsets_m()

    assert len(offsets) == 41
    assert offsets[0] == (0.0, 0.0)
    assert len(set(offsets)) == len(offsets)
    assert (0.015, 0.0) in offsets
    assert (-0.015, 0.0) in offsets
    assert (0.0, 0.015) in offsets
    assert (0.0, -0.015) in offsets
    assert (0.030, 0.0) in offsets
    assert (0.0, -0.030) in offsets
    assert (0.029, 0.0) in offsets
    assert (0.0, -0.024) in offsets


def test_boundary_anchor_offsets_include_center_and_four_edges() -> None:
    assert boundary_anchor_offsets_m() == (
        (0.0, 0.0),
        (0.026, 0.0),
        (-0.025, 0.0),
        (0.0, 0.022),
        (0.0, -0.022),
    )
