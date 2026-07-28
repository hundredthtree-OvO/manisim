import numpy as np

from mani_sim.environments.scenario import PositionRandomization, Scenario


class _Actor:
    def __init__(self) -> None:
        self.pose = None

    def set_pose(self, pose) -> None:
        self.pose = pose


def test_scenario_registry_resets_registered_entities() -> None:
    actor = _Actor()
    scenario = Scenario(
        actors={"target": actor},
        initial_positions={"target": np.array([0.4, 0.0, 0.02])},
    )

    assert scenario.cube is actor
    assert np.allclose(scenario.cube_initial_position, [0.4, 0.0, 0.02])
    scenario.reset()
    assert np.allclose(scenario.cube.pose.p, [0.4, 0.0, 0.02])


def test_scenario_position_randomization_is_seeded_and_separated() -> None:
    target = _Actor()
    goal = _Actor()
    randomization = PositionRandomization(
        target_x_bounds_m=(0.38, 0.52),
        target_y_bounds_m=(-0.12, 0.12),
        goal_x_bounds_m=(0.25, 0.40),
        goal_y_bounds_m=(0.18, 0.34),
        minimum_distance_m=0.18,
    )
    scenario = Scenario(
        actors={"target": target, "goal": goal},
        initial_positions={
            "target": np.array([0.45, 0.0, 0.02]),
            "goal": np.array([0.30, 0.30, 0.001]),
        },
        position_randomization=randomization,
    )

    scenario.reset(np.random.default_rng(7))
    first_target = scenario.cube_initial_position
    first_goal = scenario.goal_position
    scenario.reset(np.random.default_rng(7))

    assert np.allclose(scenario.cube_initial_position, first_target)
    assert np.allclose(scenario.goal_position, first_goal)
    assert np.linalg.norm(first_target[:2] - first_goal[:2]) >= 0.18
