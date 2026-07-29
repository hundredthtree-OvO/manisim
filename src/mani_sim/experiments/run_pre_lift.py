from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

import mani_skill.envs  # noqa: F401 - register ManiSkill environments
from mani_sim.config import AppConfig, load_config
from mani_sim.control.command import TaskSpaceCommand
from mani_sim.control.ee_servo import EEServo
from mani_sim.control.scene_collision_guard import SceneCollisionGuard
from mani_sim.control.workspace_guard import WorkspaceGuard
from mani_sim.environments.scenario import build_scenario
from mani_sim.experiments.checkpoint import ExperimentCheckpoint
from mani_sim.experiments.pre_lift import (
    PreLiftBranchCollector,
    PreLiftIntervention,
    default_pre_lift_interventions,
)
from mani_sim.reachability import ReachabilityMap
from mani_sim.robot_setup import initialize_panda
from mani_sim.runtime.command_executor import CommandExecutor
from mani_sim.runtime.observation import capture_runtime_observation


def _executor(config: AppConfig) -> CommandExecutor:
    reachability = (
        ReachabilityMap.load(
            config.reachability.path,
            config.reachability.maximum_height_delta_m,
        )
        if config.reachability.enabled
        else None
    )
    return CommandExecutor(
        servo=EEServo(
            config.servo.gain,
            config.servo.max_delta_m,
            config.servo.deadband_m,
        ),
        workspace_guard=WorkspaceGuard(
            x_bounds_m=config.workspace.x_bounds_m,
            y_bounds_m=config.workspace.y_bounds_m,
            z_bounds_m=config.workspace.z_bounds_m,
            work_height_m=config.workspace.work_height_m,
            progress_epsilon_m=config.servo.progress_epsilon_m,
            saturation_steps=config.servo.saturation_steps,
            saturation_distance_m=config.servo.saturation_distance_m,
            release_target_delta_m=config.servo.release_target_delta_m,
        ),
        scene_guard=SceneCollisionGuard(
            ground_clearance_m=(
                config.collision_protection.ground_tcp_clearance_m
            ),
            obstacle_margin_m=config.collision_protection.obstacle_margin_m,
        ),
        reachability=reachability,
        controller_delta_limit_m=0.1,
        previous_safe_target_weight=(
            config.reachability.previous_safe_target_weight
        ),
        maximum_projected_target_step_m=(
            config.reachability.maximum_projected_target_step_m
        ),
        collision_protection_enabled=(
            config.collision_protection.enabled
        ),
    )


def _move(
    env: gym.Env,
    scenario: Any,
    executor: CommandExecutor,
    target: np.ndarray,
    gripper: float,
    steps: int,
) -> None:
    for step in range(steps):
        observation = capture_runtime_observation(
            env.unwrapped, scenario
        )
        execution = executor.prepare(
            TaskSpaceCommand.create(
                target_position=target,
                gripper_position=gripper,
                timestamp=float(step),
                source="pre_lift_setup",
            ),
            observation,
            obstacles=scenario.obstacles,
        )
        env.step(execution.action)


def _prepare_anchor(
    env: gym.Env,
    config: AppConfig,
    scenario: Any,
    executor: CommandExecutor,
    *,
    grasp_offset_xy_m: tuple[float, float] = (0.0, 0.0),
) -> bool:
    cube = scenario.initial_position("target")
    if cube is None:
        raise ValueError("pre-lift experiment requires cube_task")
    grasp_target = cube.copy()
    grasp_target[:2] += np.asarray(grasp_offset_xy_m, dtype=np.float64)
    _move(
        env,
        scenario,
        executor,
        grasp_target + np.array([0.0, 0.0, 0.07]),
        1.0,
        180,
    )
    _move(env, scenario, executor, grasp_target, 1.0, 140)
    _move(env, scenario, executor, grasp_target, -1.0, 100)
    observation = capture_runtime_observation(env.unwrapped, scenario)
    return "target" in observation.grasped_objects


def default_anchor_offsets_m() -> tuple[tuple[float, float], ...]:
    """Scan both table axes without creating a full Cartesian grid."""
    coarse = (0.005, 0.010, 0.015, 0.020, 0.025, 0.030)
    fine_x = (0.026, 0.027, 0.028, 0.029)
    fine_y = (0.021, 0.022, 0.023, 0.024)
    return (
        (0.0, 0.0),
        *((value, 0.0) for value in (*coarse, *fine_x)),
        *((-value, 0.0) for value in (*coarse, *fine_x)),
        *((0.0, value) for value in (*coarse, *fine_y)),
        *((0.0, -value) for value in (*coarse, *fine_y)),
    )


def boundary_anchor_offsets_m() -> tuple[tuple[float, float], ...]:
    return (
        (0.0, 0.0),
        (0.026, 0.0),
        (-0.025, 0.0),
        (0.0, 0.022),
        (0.0, -0.022),
    )


def _anchor_name(offset_xy_m: tuple[float, float]) -> str:
    x_mm, y_mm = (round(value * 1000) for value in offset_xy_m)
    return f"x{x_mm:+d}_y{y_mm:+d}_mm"


def _branch_outcomes(group: Any) -> dict[str, Any]:
    return {
        "maintained_grasp_count": sum(
            branch.maintained_grasp for branch in group.branches
        ),
        "branch_count": len(group.branches),
        "maximum_relative_xy_slip_m": max(
            (
                branch.maximum_relative_xy_slip_m
                for branch in group.branches
            ),
            default=0.0,
        ),
        "outcome_diversity": len(
            {branch.maintained_grasp for branch in group.branches}
        ),
        "branch_group_complete": group.complete,
        "restore_failure_before_intervention": (
            group.restore_failure_before_intervention
        ),
    }


def _spread(values: list[list[float]]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.max(np.linalg.norm(array - array[0], axis=1)))


def _trajectory_spread(
    branches: tuple[Any, ...], field: str
) -> float:
    trajectories = np.asarray(
        [
            [getattr(step, field) for step in branch.steps]
            for branch in branches
        ],
        dtype=np.float64,
    )
    return float(
        np.max(
            np.linalg.norm(
                trajectories - trajectories[0:1], axis=-1
            )
        )
    )


def run_pre_lift_experiment(
    config: AppConfig,
    *,
    output_root: str | Path = "runs/experiments",
    fidelity_repeats: int = 3,
) -> Path:
    if fidelity_repeats < 2:
        raise ValueError("fidelity_repeats must be at least 2")
    env = gym.make(
        config.simulation.env_id,
        obs_mode="none",
        reward_mode="none",
        render_mode=None,
        control_mode=config.simulation.control_mode,
        robot_uids=config.simulation.robot_uid,
        sim_backend=config.simulation.sim_backend,
    )
    try:
        env.reset(seed=config.simulation.seed)
        initialize_panda(env.unwrapped)
        scenario = build_scenario(env.unwrapped, config)
        scenario.reset(np.random.default_rng(config.simulation.seed))
        executor = _executor(config)
        executor.reset(
            capture_runtime_observation(
                env.unwrapped, scenario
            ).tcp_position
        )
        if not _prepare_anchor(env, config, scenario, executor):
            raise RuntimeError(
                "scripted setup failed to reach centered pre_lift anchor"
            )
        checkpoint = ExperimentCheckpoint.capture(
            env.unwrapped,
            components={"executor": executor},
            user_state={
                "anchor": "pre_lift",
                "gripper_target": -1.0,
                "fixed_dynamics": True,
            },
        )
        collector = PreLiftBranchCollector(
            base_env=env.unwrapped,
            scenario=scenario,
            executor=executor,
        )
        fidelity = collector.collect(
            checkpoint,
            tuple(
                PreLiftIntervention(f"repeat_{index}")
                for index in range(fidelity_repeats)
            ),
            checkpoint_id="pre_lift_checkpoint",
        )
        group = collector.collect(
            checkpoint,
            default_pre_lift_interventions(),
            checkpoint_id="pre_lift_group_0",
        )

        experiment_id = datetime.now(timezone.utc).strftime(
            "%Y%m%d-%H%M%S-%f"
        )
        output = Path(output_root) / experiment_id
        output.mkdir(parents=True)
        group.write_json(output / "branch_group.json")
        fidelity.write_json(output / "checkpoint_repeats.json")
        report = {
            "experiment_id": experiment_id,
            "anchor": "pre_lift",
            "fixed_dynamics": True,
            "fidelity_repeats": fidelity_repeats,
            "checkpoint_final_tcp_spread_m": _spread(
                [
                    branch.final_tcp_position
                    for branch in fidelity.branches
                ]
            ),
            "checkpoint_final_object_spread_m": _spread(
                [
                    branch.final_object_position
                    for branch in fidelity.branches
                ]
            ),
            "checkpoint_tcp_trajectory_spread_m": (
                _trajectory_spread(
                    fidelity.branches, "tcp_position"
                )
            ),
            "checkpoint_object_trajectory_spread_m": (
                _trajectory_spread(
                    fidelity.branches, "object_position"
                )
            ),
            "all_repeats_maintained_grasp": all(
                branch.maintained_grasp
                for branch in fidelity.branches
            ),
            "branch_count": len(group.branches),
            "branch_summary": [
                {
                    "name": branch.intervention.name,
                    "maintained_grasp": branch.maintained_grasp,
                    "maximum_relative_xy_slip_m": (
                        branch.maximum_relative_xy_slip_m
                    ),
                    "maximum_grip_force_n": (
                        branch.maximum_grip_force_n
                    ),
                    "maximum_object_force_n": (
                        branch.maximum_object_force_n
                    ),
                    "maximum_unintended_force_n": (
                        branch.maximum_unintended_force_n
                    ),
                    "grip_force_impulse_ns": (
                        branch.grip_force_impulse_ns
                    ),
                    "object_force_impulse_ns": (
                        branch.object_force_impulse_ns
                    ),
                }
                for branch in group.branches
            ],
        }
        (output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return output
    finally:
        env.close()


def run_anchor_sweep_experiment(
    config: AppConfig,
    *,
    output_root: str | Path = "runs/experiments",
    offsets_xy_m: tuple[tuple[float, float], ...] | None = None,
) -> Path:
    """Collect fixed-dynamics branches from progressively degraded anchors."""
    offsets = offsets_xy_m or default_anchor_offsets_m()
    env = gym.make(
        config.simulation.env_id,
        obs_mode="none",
        reward_mode="none",
        render_mode=None,
        control_mode=config.simulation.control_mode,
        robot_uids=config.simulation.robot_uid,
        sim_backend=config.simulation.sim_backend,
    )
    try:
        env.reset(seed=config.simulation.seed)
        initialize_panda(env.unwrapped)
        scenario = build_scenario(env.unwrapped, config)
        scenario.reset(np.random.default_rng(config.simulation.seed))
        executor = _executor(config)
        executor.reset(
            capture_runtime_observation(
                env.unwrapped, scenario
            ).tcp_position
        )
        initial = ExperimentCheckpoint.capture(
            env.unwrapped,
            components={"executor": executor},
            user_state={"anchor": "episode_start"},
        )
        collector = PreLiftBranchCollector(
            base_env=env.unwrapped,
            scenario=scenario,
            executor=executor,
        )
        experiment_id = datetime.now(timezone.utc).strftime(
            "%Y%m%d-%H%M%S-%f"
        )
        output = Path(output_root) / f"{experiment_id}-anchor-sweep"
        output.mkdir(parents=True)
        anchors: list[dict[str, Any]] = []
        for offset in offsets:
            initial.restore(
                env.unwrapped, components={"executor": executor}
            )
            name = _anchor_name(offset)
            grasped = _prepare_anchor(
                env,
                config,
                scenario,
                executor,
                grasp_offset_xy_m=offset,
            )
            entry: dict[str, Any] = {
                "anchor_id": name,
                "grasp_offset_xy_m": list(offset),
                "formed_pre_lift": grasped,
                "restorable_pre_lift": False,
            }
            if grasped:
                checkpoint = ExperimentCheckpoint.capture(
                    env.unwrapped,
                    components={"executor": executor},
                    user_state={
                        "anchor": "pre_lift",
                        "anchor_id": name,
                        "grasp_offset_xy_m": list(offset),
                        "fixed_dynamics": True,
                    },
                )
                try:
                    group = collector.collect(
                        checkpoint,
                        default_pre_lift_interventions(),
                        checkpoint_id=name,
                    )
                except ValueError as error:
                    if "requires target to be grasped" not in str(error):
                        raise
                    entry["restore_failure"] = str(error)
                else:
                    entry["restorable_pre_lift"] = True
                    group.write_json(output / f"{name}.json")
                    entry.update(_branch_outcomes(group))
            anchors.append(entry)
        report = {
            "experiment_id": experiment_id,
            "experiment": "pre_lift_anchor_sweep",
            "simulation_seed": config.simulation.seed,
            "target_initial_position": (
                scenario.initial_position("target").tolist()
                if scenario.initial_position("target") is not None
                else None
            ),
            "fixed_dynamics": True,
            "anchor_count": len(anchors),
            "formed_anchor_count": sum(
                anchor["formed_pre_lift"] for anchor in anchors
            ),
            "restorable_anchor_count": sum(
                anchor["restorable_pre_lift"] for anchor in anchors
            ),
            "anchors": anchors,
        }
        (output / "anchor_sweep_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return output
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect fixed-dynamics branches from a pre-lift anchor."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/demo0.yaml"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/experiments"),
    )
    parser.add_argument("--fidelity-repeats", type=int, default=3)
    parser.add_argument(
        "--anchor-sweep",
        action="store_true",
        help="scan degraded pre-lift anchors instead of the fidelity run",
    )
    parser.add_argument(
        "--boundary-only",
        action="store_true",
        help="with --anchor-sweep, rerun only center and located boundaries",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="override simulation.seed for this experiment",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config = replace(
            config,
            simulation=replace(config.simulation, seed=args.seed),
        )
    output = (
        run_anchor_sweep_experiment(
            config,
            output_root=args.output_root,
            offsets_xy_m=(
                boundary_anchor_offsets_m()
                if args.boundary_only
                else None
            ),
        )
        if args.anchor_sweep
        else run_pre_lift_experiment(
            config,
            output_root=args.output_root,
            fidelity_repeats=args.fidelity_repeats,
        )
    )
    print(f"experiment_dir={output}")
    report_name = (
        "anchor_sweep_report.json"
        if args.anchor_sweep
        else "report.json"
    )
    report = json.loads(
        (output / report_name).read_text(encoding="utf-8")
    )
    if args.anchor_sweep:
        print(
            "formed_anchors="
            f"{report['formed_anchor_count']}/{report['anchor_count']}"
        )
        print(
            "restorable_anchors="
            f"{report['restorable_anchor_count']}/"
            f"{report['anchor_count']}"
        )
    else:
        print(
            "checkpoint_final_spread_m="
            f"tcp:{report['checkpoint_final_tcp_spread_m']:.6f}, "
            f"object:{report['checkpoint_final_object_spread_m']:.6f}"
        )
        print(f"branch_count={report['branch_count']}")
