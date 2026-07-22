"""Versioned, observable-history-only recursive world-model entrant contract."""
from __future__ import annotations

import math
from typing import Any, Literal, Protocol, TypedDict

import numpy as np

from .errors import ReleaseContractError
from .schema import schema_path, validate_instance


PROTOCOL_VERSION = "kdd235a.runtime.v1"
POLICY_SUM_TOLERANCE = 1e-8
FLOAT_COMPARISON_TOLERANCE = 1e-12
COVERAGE_LEVELS = (0.50, 0.80, 0.90, 0.95)
RETENTION_GRID = tuple(float(value) for value in np.linspace(0.1, 1.0, 10))


class TaskMetadata(TypedDict):
    profile: str
    environment_seed: int
    feature_count: int
    action_count: int
    supported_actions: list[int]


class ObservableHistory(TypedDict):
    observations: list[list[float]]
    masks: list[list[int]]
    recency: list[list[float]]
    previous_actions: list[int]


class RecursiveRequest(ObservableHistory):
    action_sequences: list[list[int]]
    horizon: int


class WorldModelEntrant(Protocol):
    def initialize(self, metadata: TaskMetadata, seed: int) -> dict[str, Any]: ...
    def fit_or_load(self, payload: dict[str, Any], seed: int) -> dict[str, Any]: ...
    def predict_one_step(self, request: RecursiveRequest, seed: int) -> dict[str, Any]: ...
    def predict_rollout(self, request: RecursiveRequest, seed: int) -> dict[str, Any]: ...
    def predict_policy(self, history: ObservableHistory, seed: int) -> dict[str, Any]: ...


def validate_history(payload: dict[str, Any], feature_count: int, action_count: int,
                     supported_actions: list[int]) -> int:
    observations = _matrix(payload.get("observations"), "observations", feature_count)
    rows = len(observations)
    masks = _matrix(payload.get("masks"), "masks", feature_count, rows)
    recency = _matrix(payload.get("recency"), "recency", feature_count, rows)
    previous = payload.get("previous_actions")
    if not isinstance(previous, list) or len(previous) != rows:
        raise ReleaseContractError("history_previous_action_dimension_failure")
    support = set(supported_actions)
    for row_index, row in enumerate(masks):
        if any(value not in (0, 1) or isinstance(value, bool) for value in row):
            raise ReleaseContractError(f"history_mask_binary_failure:{row_index}")
    for row_index, row in enumerate(recency):
        if any(value < 0 for value in row):
            raise ReleaseContractError(f"history_recency_negative:{row_index}")
    for value in previous:
        if not _exact_int(value) or value < 0 or value >= action_count or value not in support:
            raise ReleaseContractError("history_previous_action_invalid")
    return rows


def validate_recursive_request(payload: dict[str, Any], feature_count: int, action_count: int,
                               supported_actions: list[int]) -> tuple[int, int]:
    validate_instance(payload, schema_path("world_model_request"))
    rows = validate_history(payload, feature_count, action_count, supported_actions)
    horizon = payload.get("horizon")
    sequences = payload.get("action_sequences")
    if not _exact_int(horizon) or horizon < 1 or horizon > 64:
        raise ReleaseContractError("recursive_horizon_invalid")
    if not isinstance(sequences, list) or len(sequences) != rows:
        raise ReleaseContractError("recursive_action_sequence_batch_failure")
    support = set(supported_actions)
    for sequence in sequences:
        if not isinstance(sequence, list) or len(sequence) != horizon:
            raise ReleaseContractError("recursive_action_horizon_dimension_failure")
        if any(not _exact_int(value) or value < 0 or value >= action_count or value not in support
               for value in sequence):
            raise ReleaseContractError("recursive_action_invalid_or_unsupported")
    return rows, int(horizon)


def validate_prediction(result: dict[str, Any], declaration: dict[str, Any], rows: int,
                        horizon: int, feature_count: int) -> dict[str, np.ndarray | None]:
    validate_instance(result, schema_path("world_model_prediction"))
    if result["prediction_type"] != declaration["prediction_type"]:
        raise ReleaseContractError("prediction_type_declaration_mismatch")
    if result["horizon"] != horizon:
        raise ReleaseContractError("prediction_horizon_mismatch")
    mean = _prediction_array(result["mean"], rows, horizon, feature_count, "mean")
    prediction_type = str(result["prediction_type"])
    scale: np.ndarray | None = None
    if prediction_type == "point":
        if any(key in result for key in ("scale", "members", "within_variance", "between_variance", "total_variance")):
            raise ReleaseContractError("point_prediction_must_not_fabricate_uncertainty")
    elif prediction_type == "independent_gaussian":
        if "scale" not in result or "members" in result:
            raise ReleaseContractError("gaussian_scale_required")
        scale = _prediction_array(result["scale"], rows, horizon, feature_count, "scale")
        if np.any(scale <= 0):
            raise ReleaseContractError("gaussian_scale_must_be_positive")
    else:
        members = result.get("members")
        if not isinstance(members, list) or len(members) < 2:
            raise ReleaseContractError("ensemble_members_required")
        member_means = np.asarray([
            _prediction_array(member["mean"], rows, horizon, feature_count, "member_mean")
            for member in members
        ])
        member_scales = np.asarray([
            _prediction_array(member["scale"], rows, horizon, feature_count, "member_scale")
            for member in members
        ])
        if np.any(member_scales <= 0):
            raise ReleaseContractError("ensemble_scale_must_be_positive")
        expected_mean = member_means.mean(axis=0)
        within = np.mean(member_scales ** 2, axis=0)
        between = np.mean((member_means - expected_mean) ** 2, axis=0)
        total = within + between
        for key, expected in (("within_variance", within), ("between_variance", between), ("total_variance", total)):
            actual = _prediction_array(result.get(key), rows, horizon, feature_count, key)
            if not np.allclose(actual, expected, rtol=FLOAT_COMPARISON_TOLERANCE, atol=FLOAT_COMPARISON_TOLERANCE):
                raise ReleaseContractError(f"ensemble_{key}_identity_failure")
        if not np.allclose(mean, expected_mean, rtol=FLOAT_COMPARISON_TOLERANCE, atol=FLOAT_COMPARISON_TOLERANCE):
            raise ReleaseContractError("ensemble_mean_identity_failure")
        scale = np.sqrt(total)
    _validate_component_sources(result, declaration, rows, horizon)
    return {"mean": mean, "scale": scale}


def validate_policy_output(result: dict[str, Any], rows: int, action_count: int,
                           supported_actions: list[int]) -> np.ndarray:
    validate_instance(result, schema_path("world_model_policy"))
    planner = result["planner"]
    if planner != frozen_h4_contract():
        raise ReleaseContractError("frozen_h4_planner_identity_mismatch")
    probabilities = result["probabilities"]
    if len(probabilities) != rows:
        raise ReleaseContractError("policy_probability_row_count_mismatch")
    support = set(supported_actions)
    output = np.asarray(probabilities, dtype=float)
    if output.shape != (rows, action_count) or not np.isfinite(output).all():
        raise ReleaseContractError("policy_probability_dimension_or_finite_failure")
    if np.any(output < 0) or np.any(output > 1):
        raise ReleaseContractError("policy_probability_range_failure")
    if not np.allclose(output.sum(axis=1), 1.0, rtol=0, atol=POLICY_SUM_TOLERANCE):
        raise ReleaseContractError("policy_probability_sum_failure")
    unsupported = [index for index in range(action_count) if index not in support]
    if unsupported and np.any(output[:, unsupported] > POLICY_SUM_TOLERANCE):
        raise ReleaseContractError("policy_probability_support_failure")
    return output


def frozen_h4_contract() -> dict[str, Any]:
    return {
        "name": "H4_support_only_sequence_categorical_CEM", "horizon": 4,
        "candidates": 64, "iterations": 3, "elites": 8, "smoothing": 0.2,
        "support_only": True, "execute_first_action": True, "planner_seed": 1903408,
    }


def support_only_h4_probabilities(score_sequences: Any, rows: int, action_count: int,
                                  supported_actions: list[int], environment_seed: int) -> np.ndarray:
    """Frozen KDD224/KDD232 categorical-CEM adapter; ties retain stable candidate order."""
    contract = frozen_h4_contract()
    output = np.zeros((rows, action_count), dtype=float)
    for row in range(rows):
        rng = np.random.default_rng(contract["planner_seed"] + 1009 * environment_seed + row)
        categorical = np.full((4, len(supported_actions)), 1.0 / len(supported_actions))
        best_sequence: np.ndarray | None = None
        best_score = -math.inf
        for _ in range(3):
            candidates = np.column_stack([
                rng.choice(supported_actions, size=64, p=categorical[step]) for step in range(4)
            ])
            scores = np.asarray(score_sequences(row, candidates), dtype=float)
            if scores.shape != (64,) or not np.isfinite(scores).all():
                raise ReleaseContractError("planner_sequence_score_failure")
            order = np.argsort(-scores, kind="stable")
            if scores[order[0]] > best_score + FLOAT_COMPARISON_TOLERANCE:
                best_score, best_sequence = float(scores[order[0]]), candidates[order[0]].copy()
            elite = candidates[order[:8]]
            empirical = np.asarray([[(elite[:, step] == action).mean() for action in supported_actions]
                                    for step in range(4)])
            categorical = 0.2 * categorical + 0.8 * empirical
            categorical /= categorical.sum(axis=1, keepdims=True)
        if best_sequence is None:
            raise ReleaseContractError("planner_no_candidate")
        output[row, int(best_sequence[0])] = 1.0
    return output


def structural_probability_metrics(prediction_type: str) -> dict[str, Any]:
    if prediction_type != "point":
        return {}
    return {key: None for key in (
        "crps", "coverage_50", "coverage_80", "coverage_90", "coverage_95",
        "interval_width_50", "interval_width_80", "interval_width_90", "interval_width_95",
        "mace", "risk_coverage_area",
    )} | {"probabilistic_status": "structural_na_point_only"}


def _validate_component_sources(result: dict[str, Any], declaration: dict[str, Any],
                                rows: int, horizon: int) -> None:
    expected = rows * horizon
    sources = declaration["component_sources"]
    for component, key in (("reward", "reward"), ("termination", "termination_probability")):
        present = key in result
        if sources[component] == "entrant" and not present:
            raise ReleaseContractError(f"entrant_{component}_component_missing")
        if sources[component] == "benchmark" and present:
            raise ReleaseContractError(f"benchmark_{component}_component_must_not_be_supplied")
        if present:
            values = result[key]
            if not isinstance(values, list) or len(values) != expected:
                raise ReleaseContractError(f"{component}_horizon_dimension_failure")


def _matrix(value: Any, name: str, columns: int, rows: int | None = None) -> list[list[float]]:
    if not isinstance(value, list) or (rows is not None and len(value) != rows) or not value:
        raise ReleaseContractError(f"history_{name}_row_dimension_failure")
    output: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != columns:
            raise ReleaseContractError(f"history_{name}_column_dimension_failure")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) for item in row):
            raise ReleaseContractError(f"history_{name}_nonfinite_or_type_failure")
        output.append([float(item) for item in row])
    return output


def _prediction_array(value: Any, rows: int, horizon: int, features: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (rows * horizon, features) or not np.isfinite(array).all():
        raise ReleaseContractError(f"prediction_{name}_shape_or_finite_failure")
    return array.reshape(rows, horizon, features)


def _exact_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
