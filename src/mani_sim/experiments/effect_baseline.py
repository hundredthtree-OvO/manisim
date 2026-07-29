from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


TARGET_NAMES = (
    "slip_delta_m",
    "grip_force_delta_n",
    "object_force_delta_n",
    "grip_impulse_delta_ns",
    "object_impulse_delta_ns",
    "object_dx_m",
    "object_dy_m",
    "object_dz_m",
)
RIDGE_ALPHAS = (1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0)
KNN_NEIGHBORS = (1, 2, 3, 5)


def _load_records(dataset_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (dataset_dir / "effects.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]


def _target(record: dict[str, Any]) -> np.ndarray:
    effect = record["effect_delta_from_base"]
    object_delta = effect["final_object_delta_m"]
    return np.asarray(
        [
            effect["maximum_relative_xy_slip_m"],
            effect["maximum_grip_force_n"],
            effect["maximum_object_force_n"],
            effect["grip_force_impulse_ns"],
            effect["object_force_impulse_ns"],
            *object_delta,
        ],
        dtype=np.float64,
    )


def _state(record: dict[str, Any]) -> np.ndarray:
    anchor = record["anchor"]
    tcp = np.asarray(anchor["tcp_position"], dtype=np.float64)
    obj = np.asarray(anchor["object_position"], dtype=np.float64)
    return np.asarray(
        [
            obj[0],
            obj[1],
            *anchor["grasp_offset_xy_m"],
            *(obj - tcp),
        ],
        dtype=np.float64,
    )


def _action(
    record: dict[str, Any], intervention_names: tuple[str, ...]
) -> np.ndarray:
    intervention = record["intervention"]
    one_hot = [
        float(intervention["name"] == name)
        for name in intervention_names
    ]
    return np.asarray(
        [
            intervention["dwell_steps"] / 10.0,
            intervention["lift_scale"],
            *intervention["xy_offset_m"],
            intervention["gripper_position"],
            *one_hot,
        ],
        dtype=np.float64,
    )


def _features(
    records: list[dict[str, Any]],
    intervention_names: tuple[str, ...],
    *,
    state_conditioned: bool,
) -> np.ndarray:
    rows = []
    for record in records:
        action = _action(record, intervention_names)
        if not state_conditioned:
            rows.append(action)
            continue
        state = _state(record)
        rows.append(
            np.concatenate(
                [state, action, np.outer(state, action).ravel()]
            )
        )
    return np.asarray(rows, dtype=np.float64)


@dataclass(frozen=True)
class _Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "_Standardizer":
        scale = np.std(values, axis=0)
        return cls(
            mean=np.mean(values, axis=0),
            scale=np.where(scale < 1e-12, 1.0, scale),
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale + self.mean


@dataclass(frozen=True)
class RidgeModel:
    x_scaler: _Standardizer
    y_scaler: _Standardizer
    coefficients: np.ndarray
    alpha: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        standardized = self.x_scaler.transform(features)
        design = np.column_stack(
            [np.ones(len(standardized)), standardized]
        )
        return self.y_scaler.inverse(design @ self.coefficients)


def _fit_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
) -> RidgeModel:
    x_scaler = _Standardizer.fit(train_x)
    y_scaler = _Standardizer.fit(train_y)
    x_train = np.column_stack(
        [np.ones(len(train_x)), x_scaler.transform(train_x)]
    )
    y_train = y_scaler.transform(train_y)
    identity = np.eye(x_train.shape[1])
    identity[0, 0] = 0.0
    best: RidgeModel | None = None
    best_score = float("inf")
    for alpha in RIDGE_ALPHAS:
        coefficients = np.linalg.solve(
            x_train.T @ x_train + alpha * identity,
            x_train.T @ y_train,
        )
        candidate = RidgeModel(
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            coefficients=coefficients,
            alpha=alpha,
        )
        prediction = candidate.predict(validation_x)
        score = float(
            np.sqrt(
                np.mean(
                    (
                        (prediction - validation_y)
                        / y_scaler.scale
                    )
                    ** 2
                )
            )
        )
        if score < best_score:
            best_score = score
            best = candidate
    assert best is not None
    return best


def _intervention_mean_predictions(
    train: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
) -> np.ndarray:
    global_mean = np.mean([_target(record) for record in train], axis=0)
    means: dict[str, np.ndarray] = {}
    for name in {record["intervention"]["name"] for record in train}:
        means[name] = np.mean(
            [
                _target(record)
                for record in train
                if record["intervention"]["name"] == name
            ],
            axis=0,
        )
    return np.asarray(
        [
            means.get(record["intervention"]["name"], global_mean)
            for record in evaluation
        ]
    )


def _knn_predictions(
    train: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
    neighbors: int,
) -> np.ndarray:
    train_state = np.asarray([_state(record) for record in train])
    scaler = _Standardizer.fit(train_state)
    train_state = scaler.transform(train_state)
    evaluation_state = scaler.transform(
        np.asarray([_state(record) for record in evaluation])
    )
    train_targets = np.asarray([_target(record) for record in train])
    predictions = []
    for record, state in zip(evaluation, evaluation_state, strict=True):
        candidates = np.asarray(
            [
                index
                for index, candidate in enumerate(train)
                if candidate["intervention"]["name"]
                == record["intervention"]["name"]
            ]
        )
        if len(candidates) == 0:
            candidates = np.arange(len(train))
        distances = np.linalg.norm(
            train_state[candidates] - state, axis=1
        )
        selected = candidates[
            np.argsort(distances)[: min(neighbors, len(candidates))]
        ]
        predictions.append(np.mean(train_targets[selected], axis=0))
    return np.asarray(predictions)


def _metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    train_scale: np.ndarray,
) -> dict[str, Any]:
    error = prediction - truth
    mae = np.mean(np.abs(error), axis=0)
    rmse = np.sqrt(np.mean(error**2, axis=0))
    denominator = np.sum((truth - np.mean(truth, axis=0)) ** 2, axis=0)
    r2 = np.where(
        denominator > 1e-18,
        1.0 - np.sum(error**2, axis=0) / denominator,
        np.nan,
    )
    return {
        "normalized_rmse": float(
            np.sqrt(np.mean((error / train_scale) ** 2))
        ),
        "per_target": {
            name: {
                "mae": float(mae[index]),
                "rmse": float(rmse[index]),
                "r2": (
                    None if np.isnan(r2[index]) else float(r2[index])
                ),
            }
            for index, name in enumerate(TARGET_NAMES)
        },
    }


def _resplit_by_experiment(
    records: list[dict[str, Any]], seed: int
) -> None:
    experiments = sorted({record["experiment_id"] for record in records})
    if len(experiments) < 5:
        raise ValueError("resplit needs at least five experiments")
    random.Random(seed).shuffle(experiments)
    train_end = int(len(experiments) * 0.6)
    validation_end = train_end + int(len(experiments) * 0.2)
    assignments = {
        experiment: (
            "train"
            if index < train_end
            else (
                "validation"
                if index < validation_end
                else "test"
            )
        )
        for index, experiment in enumerate(experiments)
    }
    for record in records:
        record["split"] = assignments[record["experiment_id"]]


def train_effect_baseline(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    split_seed: int | None = None,
) -> Path:
    dataset = Path(dataset_dir)
    records = _load_records(dataset)
    if split_seed is not None:
        _resplit_by_experiment(records, split_seed)
    held = [
        record
        for record in records
        if record["labels"]["maintained_grasp"]
    ]
    by_split = {
        split: [record for record in held if record["split"] == split]
        for split in ("train", "validation", "test")
    }
    if any(not split_records for split_records in by_split.values()):
        raise ValueError("train, validation and test need held samples")
    interventions = tuple(
        sorted({record["intervention"]["name"] for record in held})
    )
    train_y = np.asarray([_target(r) for r in by_split["train"]])
    validation_y = np.asarray(
        [_target(r) for r in by_split["validation"]]
    )
    test_y = np.asarray([_target(r) for r in by_split["test"]])
    target_scaler = _Standardizer.fit(train_y)

    action_train = _features(
        by_split["train"], interventions, state_conditioned=False
    )
    action_validation = _features(
        by_split["validation"], interventions, state_conditioned=False
    )
    action_test = _features(
        by_split["test"], interventions, state_conditioned=False
    )
    action_model = _fit_ridge(
        action_train, train_y, action_validation, validation_y
    )
    state_train = _features(
        by_split["train"], interventions, state_conditioned=True
    )
    state_validation = _features(
        by_split["validation"], interventions, state_conditioned=True
    )
    state_test = _features(
        by_split["test"], interventions, state_conditioned=True
    )
    state_model = _fit_ridge(
        state_train, train_y, state_validation, validation_y
    )
    knn_scores = {}
    for neighbors in KNN_NEIGHBORS:
        prediction = _knn_predictions(
            by_split["train"], by_split["validation"], neighbors
        )
        knn_scores[neighbors] = float(
            np.sqrt(
                np.mean(
                    (
                        (prediction - validation_y)
                        / target_scaler.scale
                    )
                    ** 2
                )
            )
        )
    selected_neighbors = min(knn_scores, key=knn_scores.get)

    global_prediction = np.repeat(
        np.mean(train_y, axis=0, keepdims=True),
        len(test_y),
        axis=0,
    )
    intervention_prediction = _intervention_mean_predictions(
        by_split["train"], by_split["test"]
    )
    action_prediction = action_model.predict(action_test)
    state_prediction = state_model.predict(state_test)
    knn_prediction = _knn_predictions(
        by_split["train"], by_split["test"], selected_neighbors
    )
    models = {
        "global_mean": _metrics(
            test_y, global_prediction, target_scaler.scale
        ),
        "intervention_mean": _metrics(
            test_y, intervention_prediction, target_scaler.scale
        ),
        "action_ridge": _metrics(
            test_y, action_prediction, target_scaler.scale
        ),
        "state_action_ridge": _metrics(
            test_y, state_prediction, target_scaler.scale
        ),
        "state_knn": _metrics(
            test_y, knn_prediction, target_scaler.scale
        ),
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": str(dataset),
        "resplit_seed": split_seed,
        "continuous_subset": "maintained_grasp_only",
        "targets": TARGET_NAMES,
        "held_sample_counts": {
            split: len(split_records)
            for split, split_records in by_split.items()
        },
        "lost_sample_counts": {
            split: sum(
                record["split"] == split
                and not record["labels"]["maintained_grasp"]
                for record in records
            )
            for split in ("train", "validation", "test")
        },
        "selected_alphas": {
            "action_ridge": action_model.alpha,
            "state_action_ridge": state_model.alpha,
        },
        "selected_knn_neighbors": selected_neighbors,
        "models": models,
        "state_conditioning_gain_vs_intervention_mean": (
            models["intervention_mean"]["normalized_rmse"]
            - models["state_action_ridge"]["normalized_rmse"]
        ),
        "knn_gain_vs_intervention_mean": (
            models["intervention_mean"]["normalized_rmse"]
            - models["state_knn"]["normalized_rmse"]
        ),
    }
    report_path = destination / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    predictions = []
    for record, truth, prediction in zip(
        by_split["test"], test_y, state_prediction, strict=True
    ):
        predictions.append(
            {
                "sample_id": record["sample_id"],
                "matched_group_id": record["matched_group_id"],
                "intervention": record["intervention"]["name"],
                "target": dict(zip(TARGET_NAMES, truth.tolist())),
                "prediction": dict(
                    zip(TARGET_NAMES, prediction.tolist())
                ),
            }
        )
    (destination / "test_predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in predictions),
        encoding="utf-8",
    )
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the first matched-branch effect regression baseline."
    )
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int)
    args = parser.parse_args()
    report = train_effect_baseline(
        args.dataset_dir,
        args.output_dir,
        split_seed=args.split_seed,
    )
    print(f"report={report}")
