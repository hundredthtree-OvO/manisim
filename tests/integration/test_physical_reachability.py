import numpy as np
import pytest
import torch

from mani_sim.control.ee_servo import EEServo, build_normalized_panda_action
from mani_sim.reachability import ReachabilityMap
from mani_sim.robot_setup import initialize_panda


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="physical regression needs the external NVIDIA/Vulkan device",
    ),
]


def _tcp(env) -> np.ndarray:
    return env.unwrapped.agent.tcp_pose.p[0].detach().cpu().numpy()


def test_calibrated_targets_are_reached_by_physical_pd_loop() -> None:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    reachability = ReachabilityMap.load(
        "calibrations/panda_fixed_orientation.json"
    )
    targets = [
        np.array([0.30, 0.00, 0.45]),
        np.array([0.55, 0.00, 0.45]),
        np.array([0.45, 0.35, 0.45]),
        np.array([0.45, -0.35, 0.45]),
        np.array([0.70, 0.00, 0.45]),
        np.array([0.40, 0.00, 0.50]),
        np.array([0.35, 0.20, 0.60]),
        np.array([0.40, 0.00, 0.05]),
        np.array([0.30, 0.00, 0.05]),
        np.array([0.55, 0.00, 0.05]),
        np.array([0.40, 0.40, 0.05]),
        np.array([0.40, -0.40, 0.05]),
    ]
    assert all(
        not reachability.project_continuous(
            np.array([0.30, 0.00, target[2]]), target
        ).projected
        for target in targets
    )

    env = gym.make(
        "Empty-v1",
        obs_mode="none",
        reward_mode="none",
        render_mode=None,
        control_mode="pd_ee_delta_pos",
        robot_uids="panda_wristcam",
        sim_backend="cpu",
    )
    servo = EEServo(gain=0.8, max_delta_m=0.01, deadband_m=0.001)
    errors = []
    low_target_peak_ground_forces = []
    try:
        for target in targets:
            env.reset(seed=0)
            initialize_panda(env.unwrapped)
            for _ in range(240):
                delta = servo.metric_delta(target, _tcp(env))
                action = build_normalized_panda_action(delta, 1.0, 0.1)
                env.step(action)
                if target[2] <= 0.05:
                    forces = []
                    for link in env.unwrapped.agent.robot.links:
                        if link.name == "panda_link0":
                            continue
                        force = env.unwrapped.scene.get_pairwise_contact_forces(
                            env.unwrapped.ground, link
                        )[0]
                        forces.append(
                            float(torch.linalg.norm(force).detach().cpu())
                        )
                    low_target_peak_ground_forces.append(max(forces, default=0.0))
            errors.append(float(np.linalg.norm(target - _tcp(env))))
    finally:
        env.close()

    print("physical_target_errors_m=", [round(error, 6) for error in errors])
    print(
        "low_target_peak_ground_force_N=",
        round(max(low_target_peak_ground_forces, default=0.0), 6),
    )
    assert max(errors) < 0.015, errors
    assert max(low_target_peak_ground_forces, default=0.0) < 0.01
