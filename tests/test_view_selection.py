from mani_sim.input.view_selection import ViewSelection


def test_auxiliary_views_freeze_pointer_and_top_rearms_after_motion() -> None:
    selection = ViewSelection(rearm_distance_px=3.0)
    assert selection.accepts_pointer((100, 100))

    selection.switch(2, (100, 100))
    assert not selection.accepts_pointer((300, 300))

    selection.switch(1, (300, 300))
    assert not selection.accepts_pointer((300, 300))
    assert not selection.accepts_pointer((302, 300))
    assert selection.accepts_pointer((304, 300))


def test_wrist_view_never_accepts_pointer() -> None:
    selection = ViewSelection()
    selection.switch(3, (0, 0))
    assert not selection.accepts_pointer((100, 100))
