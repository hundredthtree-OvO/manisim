"""Compatibility exports for the pre-scenario module path."""

from mani_sim.environments.scenario import Scenario, build_scenario

TaskScene = Scenario
build_task_scene = build_scenario

__all__ = ["TaskScene", "build_task_scene"]
