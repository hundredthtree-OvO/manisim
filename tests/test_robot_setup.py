import numpy as np

from mani_sim.robot_setup import PANDA_SAFE_QPOS


def test_safe_qpos_has_expected_panda_dofs() -> None:
    assert PANDA_SAFE_QPOS.shape == (9,)
    assert PANDA_SAFE_QPOS[3] < 0
    assert np.isclose(PANDA_SAFE_QPOS[1], 0.17212291)
    np.testing.assert_allclose(PANDA_SAFE_QPOS[-2:], [0.04, 0.04])
