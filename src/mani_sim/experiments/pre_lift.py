from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from mani_sim.control.command import TaskSpaceCommand
from mani_sim.experiments.checkpoint import (
    ExperimentCheckpoint,
    ExperimentStateful,
)
from mani_sim.runtime.command_executor import CommandExecutor
from mani_sim.runtime.observation import (
    capture_runtime_observation,
    single_env_vector,
)


@dataclass(frozen=True)
class PreLiftIntervention:
    name: str
    dwell_steps: int = 0
    lift_scale: float = 1.0
    xy_offset_m: tuple[float, float] = (0.0, 0.0)
    gripper_position: float = -1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("intervention name must not be empty")
        if self.dwell_steps < 0:
            raise ValueError("dwell_steps must not be negative")
        if self.lift_scale <= 0:
            raise ValueError("lift_scale must be positive")
        if not -1.0 <= self.gripper_position <= 1.0:
            raise ValueError("gripper_position must be in [-1, 1]")


@dataclass(frozen=True)
class BranchStep:
    step: int
    target_world: list[float]
    safe_target_world: list[float]
    action: list[float]
    tcp_position: list[float]
    object_position: list[float]
    object_linear_velocity: list[float]
    qpos: list[float]
    qvel: list[float]
    grasped: bool
    left_finger_force_n: float
    right_finger_force_n: float
    grip_force_n: float
    object_force_n: float
    unintended_force_n: float


@dataclass(frozen=True)
class BranchResult:
    intervention: PreLiftIntervention
    steps: tuple[BranchStep, ...]
    maintained_grasp: bool
    final_tcp_position: list[float]
    final_object_position: list[float]
    maximum_relative_xy_slip_m: float
    maximum_grip_force_n: float
    maximum_object_force_n: float
    maximum_unintended_force_n: float
    grip_force_impulse_ns: float
    object_force_impulse_ns: float


@dataclass(frozen=True)
class PreLiftBranchGroup:
    checkpoint_id: str
    anchor: str
    fixed_dynamics: bool
    control_dt_s: float
    anchor_tcp_position: list[float]
    anchor_object_position: list[float]
    branches: tuple[BranchResult, ...]
    complete: bool = True
    restore_failure_before_intervention: str | None = None

    def write_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return output


def default_pre_lift_interventions() -> tuple[PreLiftIntervention, ...]:
    return (
        PreLiftIntervention("base"),
        PreLiftIntervention("wait_5", dwell_steps=5),
        PreLiftIntervention("slow", lift_scale=0.5),
        PreLiftIntervention("fast", lift_scale=1.5),
        PreLiftIntervention("x_plus", xy_offset_m=(0.003, 0.0)),
        PreLiftIntervention("x_minus", xy_offset_m=(-0.003, 0.0)),
        PreLiftIntervention("hold_m050", gripper_position=-0.5),
        PreLiftIntervention("hold_m025", gripper_position=-0.25),
        PreLiftIntervention("hold_000", gripper_position=0.0),
        PreLiftIntervention("hold_p025", gripper_position=0.025),
        PreLiftIntervention("hold_p050", gripper_position=0.05),
        PreLiftIntervention("hold_p100", gripper_position=0.1),
        PreLiftIntervention("hold_p250", gripper_position=0.25),
    )


class PreLiftBranchCollector:
    """Execute local lift interventions from one restored pre-lift state."""

    def __init__(
        self,
        *,
        base_env: Any,
        scenario: Any,
        executor: CommandExecutor,
        lift_distance_m: float = 0.14,
        ramp_steps: int = 30,
        horizon_steps: int = 70,
    ):
        if lift_distance_m <= 0:
            raise ValueError("lift_distance_m must be positive")
        if ramp_steps < 1 or horizon_steps < 1:
            raise ValueError("branch step counts must be positive")
        self.base_env = base_env
        self.scenario = scenario
        self.executor = executor
        self.lift_distance_m = lift_distance_m
        self.ramp_steps = ramp_steps
        self.horizon_steps = horizon_steps
        self.control_dt_s = 1.0 / float(base_env.control_freq)

    def _components(
        self,
        extra: Mapping[str, ExperimentStateful] | None,
    ) -> dict[str, ExperimentStateful]:
        return {"executor": self.executor, **dict(extra or {})}

    def _object_velocity(self) -> np.ndarray:
        target = self.scenario.actor("target")
        if target is None:
            return np.zeros(3, dtype=np.float64)
        return single_env_vector(target.get_linear_velocity())

    def _validate_anchor(self) -> tuple[np.ndarray, np.ndarray]:
        observation = capture_runtime_observation(
            self.base_env, self.scenario
        )
        if "target" not in observation.object_positions:
            raise ValueError("pre_lift anchor requires target actor")
        if "target" not in observation.grasped_objects:
            raise ValueError(
                "pre_lift anchor requires target to be grasped"
            )
        return (
            observation.tcp_position.copy(),
            observation.object_positions["target"].copy(),
        )

    def collect(
        self,
        checkpoint: ExperimentCheckpoint,
        interventions: Iterable[PreLiftIntervention],
        *,
        checkpoint_id: str,
        components: Mapping[str, ExperimentStateful] | None = None,
    ) -> PreLiftBranchGroup:
        component_map = self._components(components)
        branches: list[BranchResult] = []
        anchor_tcp: np.ndarray | None = None
        anchor_object: np.ndarray | None = None
        restore_failure: str | None = None
        for intervention in interventions:
            checkpoint.restore(
                self.base_env, components=component_map
            )
            try:
                current_tcp, current_object = self._validate_anchor()
            except ValueError:
                if not branches:
                    raise
                restore_failure = intervention.name
                break
            if anchor_tcp is None:
                anchor_tcp = current_tcp
                anchor_object = current_object
            elif not (
                np.allclose(current_tcp, anchor_tcp)
                and np.allclose(current_object, anchor_object)
            ):
                raise RuntimeError(
                    "checkpoint restore changed the pre-lift anchor"
                )
            branches.append(
                self._run_branch(
                    intervention,
                    anchor_tcp=current_tcp,
                    anchor_object=current_object,
                )
            )

        if anchor_tcp is None or anchor_object is None:
            raise ValueError("at least one intervention is required")
        return PreLiftBranchGroup(
            checkpoint_id=checkpoint_id,
            anchor="pre_lift",
            fixed_dynamics=True,
            control_dt_s=self.control_dt_s,
            anchor_tcp_position=anchor_tcp.tolist(),
            anchor_object_position=anchor_object.tolist(),
            branches=tuple(branches),
            complete=restore_failure is None,
            restore_failure_before_intervention=restore_failure,
        )

    def _run_branch(
        self,
        intervention: PreLiftIntervention,
        *,
        anchor_tcp: np.ndarray,
        anchor_object: np.ndarray,
    ) -> BranchResult:
        steps: list[BranchStep] = []
        initial_relative_xy = anchor_object[:2] - anchor_tcp[:2]
        maximum_slip = 0.0
        maintained_grasp = True
        grip_values: list[float] = []
        object_values: list[float] = []
        unintended_values: list[float] = []

        for step in range(self.horizon_steps):
            observation = capture_runtime_observation(
                self.base_env, self.scenario
            )
            active_step = max(
                0, step - intervention.dwell_steps + 1
            )
            progress = min(
                1.0,
                active_step
                * intervention.lift_scale
                / self.ramp_steps,
            )
            target = anchor_tcp.copy()
            target[:2] += (
                progress
                * np.asarray(
                    intervention.xy_offset_m, dtype=np.float64
                )
            )
            target[2] += progress * self.lift_distance_m
            execution = self.executor.prepare(
                TaskSpaceCommand.create(
                    target_position=target,
                    gripper_position=intervention.gripper_position,
                    timestamp=step * self.control_dt_s,
                    source="pre_lift_branch",
                    metadata={
                        "intervention": intervention.name,
                        "branch_step": step,
                    },
                ),
                observation,
                obstacles=self.scenario.obstacles,
            )
            self.base_env.step(execution.action)
            next_observation = capture_runtime_observation(
                self.base_env, self.scenario
            )
            force = next_observation.contact_forces
            obj = next_observation.object_positions["target"]
            relative_xy = obj[:2] - next_observation.tcp_position[:2]
            maximum_slip = max(
                maximum_slip,
                float(np.linalg.norm(relative_xy - initial_relative_xy)),
            )
            grasped = "target" in next_observation.grasped_objects
            maintained_grasp = maintained_grasp and grasped
            grip_values.append(force.grip_n)
            object_values.append(force.object_net_n)
            unintended_values.append(force.maximum_unintended_n)
            steps.append(
                BranchStep(
                    step=step,
                    target_world=target.tolist(),
                    safe_target_world=execution.safe_target.tolist(),
                    action=execution.action.tolist(),
                    tcp_position=next_observation.tcp_position.tolist(),
                    object_position=obj.tolist(),
                    object_linear_velocity=(
                        self._object_velocity().tolist()
                    ),
                    qpos=next_observation.qpos.tolist(),
                    qvel=next_observation.qvel.tolist(),
                    grasped=grasped,
                    left_finger_force_n=force.left_finger_n,
                    right_finger_force_n=force.right_finger_n,
                    grip_force_n=force.grip_n,
                    object_force_n=force.object_net_n,
                    unintended_force_n=force.maximum_unintended_n,
                )
            )

        final = steps[-1]
        return BranchResult(
            intervention=intervention,
            steps=tuple(steps),
            maintained_grasp=maintained_grasp,
            final_tcp_position=final.tcp_position,
            final_object_position=final.object_position,
            maximum_relative_xy_slip_m=maximum_slip,
            maximum_grip_force_n=max(grip_values, default=0.0),
            maximum_object_force_n=max(object_values, default=0.0),
            maximum_unintended_force_n=max(
                unintended_values, default=0.0
            ),
            grip_force_impulse_ns=(
                float(sum(grip_values)) * self.control_dt_s
            ),
            object_force_impulse_ns=(
                float(sum(object_values)) * self.control_dt_s
            ),
        )
