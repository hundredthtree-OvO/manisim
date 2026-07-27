from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs  # noqa: F401
from mani_skill.utils.structs.pose import Pose
from mani_sim.robot_setup import initialize_panda


def _values(lower: float, upper: float, step: float) -> np.ndarray:
    count = int(round((upper - lower) / step))
    return np.linspace(lower, upper, count + 1)


def calibrate(
    output: Path,
    heights: list[float],
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    grid_step_m: float,
    seed: int,
    joint_limit_margin_rad: float,
    maximum_fk_position_error_m: float,
    maximum_fk_orientation_error_rad: float,
    boundary_grid_step_m: float,
    robot_uid: str,
) -> dict:
    env = gym.make(
        "Empty-v1",
        obs_mode="none",
        reward_mode="none",
        render_mode=None,
        control_mode="pd_ee_delta_pos",
        robot_uids=robot_uid,
        sim_backend="cpu",
    )
    try:
        env.reset(seed=seed)
        base_env = env.unwrapped
        initialize_panda(base_env)
        arm = base_env.agent.controller.controllers["arm"]
        q0 = base_env.agent.robot.get_qpos()
        hand_pose = arm.ee_pose_at_base
        tcp_position = base_env.agent.tcp_pose.p
        tcp_offset = tcp_position - hand_pose.p
        qlimits = base_env.agent.robot.get_qlimits()[0, :7].detach().cpu().numpy()
        xs = _values(*x_bounds, grid_step_m)
        ys = _values(*y_bounds, grid_step_m)
        layers = []

        def check_target(
            x: float, y: float, height: float
        ) -> tuple[bool, str, float, float]:
            tcp_target = torch.tensor(
                [[x, y, height]], device=base_env.device, dtype=torch.float32
            )
            hand_target = tcp_target - tcp_offset
            pose = Pose.create_from_pq(hand_target, hand_pose.q)
            solution = arm.kinematics.compute_ik(pose=pose, q0=q0)
            if solution is None:
                return False, "ik", 0.0, 0.0
            solution_array = solution[0].detach().cpu().numpy()
            margin = np.minimum(
                solution_array - qlimits[:, 0],
                qlimits[:, 1] - solution_array,
            )
            if float(np.min(margin)) < joint_limit_margin_rad:
                return False, "joint_margin", 0.0, 0.0

            pmodel_qpos = (
                q0[:, arm.kinematics.pmodel_active_joint_indices][0]
                .detach()
                .cpu()
                .numpy()
                .copy()
            )
            pmodel_qpos[
                arm.kinematics.pmodel_controlled_joint_indices.cpu().numpy()
            ] = solution_array
            arm.kinematics.pmodel.compute_forward_kinematics(pmodel_qpos)
            achieved = arm.kinematics.pmodel.get_link_pose(
                arm.kinematics.end_link_idx
            )
            target_position = hand_target[0].detach().cpu().numpy()
            target_orientation = hand_pose.q[0].detach().cpu().numpy()
            position_error = float(
                np.linalg.norm(np.asarray(achieved.p) - target_position)
            )
            quaternion_dot = min(
                1.0,
                float(abs(np.dot(np.asarray(achieved.q), target_orientation))),
            )
            orientation_error = 2.0 * float(np.arccos(quaternion_dot))
            if (
                position_error > maximum_fk_position_error_m
                or orientation_error > maximum_fk_orientation_error_rad
            ):
                return False, "fk", position_error, orientation_error
            return True, "reachable", position_error, orientation_error

        for height in heights:
            reachable: list[list[float]] = []
            coarse_status: dict[tuple[int, int], bool] = {}
            rejection_counts = {"ik": 0, "joint_margin": 0, "fk": 0}
            maximum_position_error = 0.0
            maximum_orientation_error = 0.0
            for ix, x in enumerate(xs):
                for iy, y in enumerate(ys):
                    valid, reason, position_error, orientation_error = check_target(
                        float(x), float(y), height
                    )
                    coarse_status[(ix, iy)] = valid
                    maximum_position_error = max(
                        maximum_position_error, position_error
                    )
                    maximum_orientation_error = max(
                        maximum_orientation_error, orientation_error
                    )
                    if valid:
                        reachable.append([round(float(x), 6), round(float(y), 6)])
                    else:
                        rejection_counts[reason] += 1

            boundary_centers = []
            for (ix, iy), valid in coarse_status.items():
                if not valid:
                    continue
                neighbors = (
                    coarse_status.get((ix + dx, iy + dy), False)
                    for dx in (-1, 0, 1)
                    for dy in (-1, 0, 1)
                    if dx != 0 or dy != 0
                )
                if not all(neighbors):
                    boundary_centers.append((xs[ix], ys[iy]))

            fine_candidates: set[tuple[float, float]] = set()
            for center_x, center_y in boundary_centers:
                fine_xs = _values(
                    max(x_bounds[0], center_x - grid_step_m),
                    min(x_bounds[1], center_x + grid_step_m),
                    boundary_grid_step_m,
                )
                fine_ys = _values(
                    max(y_bounds[0], center_y - grid_step_m),
                    min(y_bounds[1], center_y + grid_step_m),
                    boundary_grid_step_m,
                )
                fine_candidates.update(
                    (round(float(x), 6), round(float(y), 6))
                    for x in fine_xs
                    for y in fine_ys
                )

            refined_samples = []
            refined_rejection_counts = {"ik": 0, "joint_margin": 0, "fk": 0}
            for x, y in sorted(fine_candidates):
                valid, reason, position_error, orientation_error = check_target(
                    x, y, height
                )
                maximum_position_error = max(
                    maximum_position_error, position_error
                )
                maximum_orientation_error = max(
                    maximum_orientation_error, orientation_error
                )
                if not valid:
                    refined_rejection_counts[reason] += 1
                refined_samples.append([x, y, int(valid)])
            total = len(xs) * len(ys)
            layers.append(
                {
                    "height_m": height,
                    "sample_count": total,
                    "reachable_count": len(reachable),
                    "reachable_ratio": len(reachable) / total,
                    "rejected_ik_count": rejection_counts["ik"],
                    "rejected_joint_margin_count": rejection_counts[
                        "joint_margin"
                    ],
                    "rejected_fk_count": rejection_counts["fk"],
                    "boundary_coarse_cell_count": len(boundary_centers),
                    "refined_sample_count": len(refined_samples),
                    "refined_reachable_count": sum(
                        sample[2] for sample in refined_samples
                    ),
                    "refined_rejected_ik_count": refined_rejection_counts["ik"],
                    "refined_rejected_joint_margin_count": (
                        refined_rejection_counts["joint_margin"]
                    ),
                    "refined_rejected_fk_count": refined_rejection_counts["fk"],
                    "maximum_ik_solution_position_error_m": maximum_position_error,
                    "maximum_ik_solution_orientation_error_rad": (
                        maximum_orientation_error
                    ),
                    "reachable_points_xy": reachable,
                    "refined_boundary_samples_xy_reachable": refined_samples,
                }
            )
            print(
                f"z={height:.3f}: {len(reachable)}/{total} "
                f"({len(reachable) / total:.1%}) coarse reachable, "
                f"{len(refined_samples)} boundary samples"
            )

        result = {
            "schema_version": 2,
            "robot_uid": robot_uid,
            "control_mode": "pd_ee_delta_pos",
            "seed": seed,
            "grid_step_m": grid_step_m,
            "boundary_grid_step_m": boundary_grid_step_m,
            "joint_limit_margin_rad": joint_limit_margin_rad,
            "maximum_fk_position_error_m": maximum_fk_position_error_m,
            "maximum_fk_orientation_error_rad": maximum_fk_orientation_error_rad,
            "x_bounds_m": list(x_bounds),
            "y_bounds_m": list(y_bounds),
            "fixed_tcp_orientation_wxyz": (
                base_env.agent.tcp_pose.q[0].detach().cpu().tolist()
            ),
            "initial_qpos": q0[0].detach().cpu().tolist(),
            "layers": layers,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return result
    finally:
        env.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample Panda fixed-orientation TCP reachability with IK."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("calibrations/panda_fixed_orientation.json"),
    )
    parser.add_argument(
        "--heights",
        type=float,
        nargs="+",
        default=[
            0.05,
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
            0.65,
        ],
    )
    parser.add_argument("--x-bounds", type=float, nargs=2, default=[0.15, 0.75])
    parser.add_argument("--y-bounds", type=float, nargs=2, default=[-0.55, 0.55])
    parser.add_argument("--grid-step-m", type=float, default=0.025)
    parser.add_argument("--boundary-grid-step-m", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--robot-uid", default="panda_wristcam")
    parser.add_argument("--joint-limit-margin-rad", type=float, default=0.02)
    parser.add_argument("--maximum-fk-position-error-m", type=float, default=0.002)
    parser.add_argument(
        "--maximum-fk-orientation-error-rad", type=float, default=0.02
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.grid_step_m <= 0:
        raise SystemExit("--grid-step-m must be positive")
    if not 0 < args.boundary_grid_step_m < args.grid_step_m:
        raise SystemExit("--boundary-grid-step-m must be positive and finer")
    calibrate(
        args.output,
        args.heights,
        tuple(args.x_bounds),
        tuple(args.y_bounds),
        args.grid_step_m,
        args.seed,
        args.joint_limit_margin_rad,
        args.maximum_fk_position_error_m,
        args.maximum_fk_orientation_error_rad,
        args.boundary_grid_step_m,
        args.robot_uid,
    )


if __name__ == "__main__":
    main()
