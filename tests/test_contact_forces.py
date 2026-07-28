import numpy as np

from mani_sim.runtime.contact_forces import ContactForceSample


def test_contact_force_sample_derives_ui_scalars() -> None:
    sample = ContactForceSample(
        left_finger_world_n=np.array([3.0, 4.0, 0.0]),
        right_finger_world_n=np.array([0.0, 3.0, 0.0]),
        object_net_world_n=np.array([0.0, 0.0, 9.0]),
        unintended_by_pair_world_n={
            "ground/link": np.array([0.0, 0.0, 8.5])
        },
    )

    assert sample.left_finger_n == 5.0
    assert sample.right_finger_n == 3.0
    assert sample.grip_n == 3.0
    assert sample.object_net_n == 9.0
    assert sample.maximum_unintended_n == 8.5
