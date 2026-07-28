import numpy as np

from mani_sim.environments.scenario import Scenario


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
