import numpy as np

from mani_sim.control.ee_servo import EEServo, build_normalized_panda_action


def test_servo_deadband() -> None:
    servo = EEServo(gain=0.5, max_delta_m=0.01, deadband_m=0.002)
    np.testing.assert_array_equal(
        servo.metric_delta([0.001, 0, 0], [0, 0, 0]), np.zeros(3)
    )


def test_servo_clips_by_vector_norm() -> None:
    servo = EEServo(gain=1.0, max_delta_m=0.01, deadband_m=0)
    delta = servo.metric_delta([3, 4, 0], [0, 0, 0])
    np.testing.assert_allclose(delta, [0.006, 0.008, 0], atol=1e-7)


def test_action_maps_metric_delta_and_gripper() -> None:
    action = build_normalized_panda_action([0.01, -0.02, 0], -1, 0.1)
    np.testing.assert_allclose(action, [0.1, -0.2, 0, -1])


def test_action_is_clipped_to_normalized_space() -> None:
    action = build_normalized_panda_action([1, -1, 0], 2, 0.1)
    np.testing.assert_array_equal(action, [1, -1, 0, 1])
