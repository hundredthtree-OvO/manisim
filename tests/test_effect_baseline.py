from __future__ import annotations

import numpy as np

from mani_sim.experiments.effect_baseline import (
    _fit_ridge,
    _resplit_by_experiment,
)


def test_ridge_uses_validation_to_fit_state_dependent_effect() -> None:
    train_x = np.arange(1.0, 9.0)[:, None]
    train_y = np.column_stack([2.0 * train_x[:, 0], -train_x[:, 0]])
    validation_x = np.asarray([[9.0], [10.0]])
    validation_y = np.column_stack(
        [2.0 * validation_x[:, 0], -validation_x[:, 0]]
    )

    model = _fit_ridge(
        train_x, train_y, validation_x, validation_y
    )
    prediction = model.predict(np.asarray([[11.0]]))

    assert np.allclose(prediction, [[22.0, -11.0]], atol=0.05)


def test_resplit_keeps_experiments_together() -> None:
    records = [
        {"experiment_id": f"exp-{experiment}", "branch": branch}
        for experiment in range(10)
        for branch in range(3)
    ]

    _resplit_by_experiment(records, seed=4)

    for experiment in range(10):
        assert len(
            {
                record["split"]
                for record in records
                if record["experiment_id"] == f"exp-{experiment}"
            }
        ) == 1
