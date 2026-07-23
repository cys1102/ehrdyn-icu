"""Aggregate-only canonical-v2 EHR prediction component scorer."""
from __future__ import annotations

import math
from statistics import NormalDist
from pathlib import Path
from typing import Any

import numpy as np

from .canonical import load_strict_json
from .errors import ReleaseContractError
from .schema import schema_path, validate_instance


EHR_COMPONENT_BENCHMARK_VERSION = "ehrdyn-icu-canonical-v2.0.0"
EHR_COMPONENT_EVALUATOR_VERSION = "ehr-component-scorer-v2.0.0"
SCHEMA_VERSION = "ehr-component-submission-v2.0.0"
RESULT_SCHEMA_VERSION = "ehr-component-result-v2.0.0"
FEATURE_COUNT = 33
FLOAT_TOLERANCE = 1e-12
COVERAGE_LEVELS = (0.50, 0.80, 0.90, 0.95)
RISK_RETENTION = tuple(float(value) for value in np.linspace(0.1, 1.0, 10))
TASK_CONTRACT = {
    "sepsis": {"action_count": 25, "max_horizon": 11},
    "respiratory_support": {"action_count": 25, "max_horizon": 10},
    "shock": {"action_count": 25, "max_horizon": 11},
    "aki": {"action_count": 4, "max_horizon": 11},
    "heart_failure": {"action_count": 2, "max_horizon": 10},
}
CLAIM_BOUNDARY = (
    "Retrospective development-benchmark prediction and evaluability diagnostics only. "
    "Planning, direct return, known policy value, treatment benefit, causal effect, "
    "clinical utility, and deployment are not supported."
)


def score_submission(path: Path) -> dict[str, Any]:
    payload = load_strict_json(path)
    validate_instance(payload, schema_path("ehr_component_submission"))
    if not isinstance(payload, dict):
        raise ReleaseContractError("ehr_component_submission_object_required")
    checked = _validate_semantics(payload)
    result = _aggregate(checked)
    validate_instance(result, schema_path("ehr_component_result"))
    return result


def _validate_semantics(payload: dict[str, Any]) -> dict[str, Any]:
    task_id = str(payload["task_id"])
    task = TASK_CONTRACT[task_id]
    prediction_type = str(payload["prediction_type"])
    records = payload["records"]
    for index, record in enumerate(records):
        horizon = int(record["horizon_step"])
        if horizon > task["max_horizon"]:
            raise ReleaseContractError(f"horizon_exceeds_task_contract:{index}")
        action = record["action"]
        if isinstance(action, bool) or action < 0 or action >= task["action_count"]:
            raise ReleaseContractError(f"action_out_of_range:{index}")
        target = np.asarray(record["target"], dtype=float)
        mask = np.asarray(record["target_mask"], dtype=int)
        mean = np.asarray(record["mean"], dtype=float)
        if target.shape != (FEATURE_COUNT,) or mask.shape != (FEATURE_COUNT,) or mean.shape != (FEATURE_COUNT,):
            raise ReleaseContractError(f"feature_dimension_mismatch:{index}")
        if np.any((mask != 0) & (mask != 1)):
            raise ReleaseContractError(f"target_mask_not_binary:{index}")
        if prediction_type == "point":
            if "scale" in record or "members" in record:
                raise ReleaseContractError(f"point_uncertainty_fabrication:{index}")
        elif prediction_type == "independent_gaussian":
            if "scale" not in record or "members" in record:
                raise ReleaseContractError(f"gaussian_scale_contract:{index}")
            scale = np.asarray(record["scale"], dtype=float)
            if scale.shape != (FEATURE_COUNT,) or np.any(scale <= 0):
                raise ReleaseContractError(f"gaussian_scale_invalid:{index}")
        else:
            if "members" not in record or "scale" in record:
                raise ReleaseContractError(f"ensemble_members_contract:{index}")
            member_means = np.asarray([member["mean"] for member in record["members"]], dtype=float)
            member_scales = np.asarray([member["scale"] for member in record["members"]], dtype=float)
            if member_means.ndim != 2 or member_means.shape[1] != FEATURE_COUNT:
                raise ReleaseContractError(f"ensemble_member_mean_dimension:{index}")
            if member_scales.shape != member_means.shape or np.any(member_scales <= 0):
                raise ReleaseContractError(f"ensemble_member_scale_invalid:{index}")
            expected = member_means.mean(axis=0)
            if not np.allclose(mean, expected, rtol=FLOAT_TOLERANCE, atol=FLOAT_TOLERANCE):
                raise ReleaseContractError(f"ensemble_mean_identity:{index}")
    return payload


def _aggregate(payload: dict[str, Any]) -> dict[str, Any]:
    prediction_type = str(payload["prediction_type"])
    records = payload["records"]
    errors_by_horizon: dict[int, list[float]] = {}
    all_errors: list[float] = []
    all_scales: list[float] = []
    term_targets: list[float] = []
    term_probabilities: list[float] = []
    weights: list[float] = []
    supported: list[bool] = []
    behavior_nll: list[float] = []
    target_mass = 0.0
    unsupported_mass = 0.0
    for record in records:
        horizon = int(record["horizon_step"])
        target = np.asarray(record["target"], dtype=float)
        mask = np.asarray(record["target_mask"], dtype=bool)
        mean = np.asarray(record["mean"], dtype=float)
        errors = (mean[mask] - target[mask]).astype(float)
        errors_by_horizon.setdefault(horizon, []).extend(errors.tolist())
        all_errors.extend(errors.tolist())
        if prediction_type != "point":
            scale = _predictive_scale(record, prediction_type)[mask]
            all_scales.extend(scale.astype(float).tolist())
        termination_target = float(record["termination_target"])
        termination_probability = float(record["termination_probability"])
        term_targets.append(termination_target)
        term_probabilities.append(termination_probability)
        behavior = float(record["behavior_probability"])
        target_probability = float(record["target_probability"])
        weights.append(target_probability / behavior)
        is_supported = bool(record["action_supported"])
        supported.append(is_supported)
        behavior_nll.append(-math.log(behavior))
        target_mass += target_probability
        if not is_supported:
            unsupported_mass += target_probability
    error_array = np.asarray(all_errors, dtype=float)
    probability = _probabilistic_metrics(error_array, np.asarray(all_scales), prediction_type)
    termination = _termination_metrics(np.asarray(term_targets), np.asarray(term_probabilities))
    context = payload["aggregate_context"]
    suppressed = (
        not bool(payload["synthetic"])
        and (int(context["distinct_subjects"]) < 100 or int(context["episodes"]) < 100)
    )
    weight_array = np.asarray(weights, dtype=float)
    support = {
        "status": "suppressed_minimum_cell" if suppressed else "diagnostic_only",
        "records": len(records),
        "supported_fraction": None if suppressed else float(np.mean(supported)),
        "overlap_fraction": None if suppressed else float(np.mean(np.asarray(supported) & (weight_array > 0))),
        "unsupported_target_mass": None if suppressed else (
            float(unsupported_mass / target_mass) if target_mass > 0 else 0.0
        ),
        "behavior_denominator_nll": None if suppressed else float(np.mean(behavior_nll)),
        "importance_weight_ess": None if suppressed else _ess(weight_array),
        "importance_weight_ess_fraction": None if suppressed else _ess(weight_array) / len(weight_array),
        "claim": "retrospective_diagnostic_only",
    }
    horizon_rows = [
        {
            "horizon_step": horizon,
            "observed_cells": len(values),
            "rmse": _rmse(np.asarray(values, dtype=float)),
        }
        for horizon, values in sorted(errors_by_horizon.items())
    ]
    one_step = next((row["rmse"] for row in horizon_rows if row["horizon_step"] == 1), None)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "benchmark_version": EHR_COMPONENT_BENCHMARK_VERSION,
        "evaluator_version": EHR_COMPONENT_EVALUATOR_VERSION,
        "task_id": payload["task_id"],
        "prediction_type": prediction_type,
        "synthetic": payload["synthetic"],
        "records_scored": len(records),
        "observed_cells": len(all_errors),
        "one_step_rmse": one_step,
        "recursive_rmse_by_horizon": horizon_rows,
        "probabilistic_metrics": probability,
        "termination_metrics": termination,
        "support_diagnostics": support,
        "suppression": {
            "applied": suppressed,
            "minimum_distinct_subjects": 100,
            "minimum_episodes": 100,
        },
        "capabilities": {
            "one_step_forecasting": "Yes",
            "recursive_forecasting": "Yes",
            "uncertainty": "Structural N/A" if prediction_type == "point" else "Yes",
            "termination": "Yes",
            "support_diagnostics": "Yes",
            "planning": "No",
            "direct_return": "Structural N/A",
            "known_policy_value": "Structural N/A",
            "treatment_benefit": "No",
            "clinical_utility": "No",
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _predictive_scale(record: dict[str, Any], prediction_type: str) -> np.ndarray:
    if prediction_type == "independent_gaussian":
        return np.asarray(record["scale"], dtype=float)
    means = np.asarray([member["mean"] for member in record["members"]], dtype=float)
    scales = np.asarray([member["scale"] for member in record["members"]], dtype=float)
    within = np.mean(scales ** 2, axis=0)
    between = np.mean((means - means.mean(axis=0)) ** 2, axis=0)
    return np.sqrt(within + between)


def _probabilistic_metrics(errors: np.ndarray, scales: np.ndarray, prediction_type: str) -> dict[str, Any]:
    names = (
        "crps", "coverage_50", "coverage_80", "coverage_90", "coverage_95",
        "interval_width_50", "interval_width_80", "interval_width_90", "interval_width_95",
        "mace", "risk_coverage_area",
    )
    if prediction_type == "point":
        return {"status": "structural_na_point_only"} | {name: None for name in names}
    z = -errors / scales
    normal = NormalDist()
    phi = np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    cdf = np.asarray([normal.cdf(float(value)) for value in z])
    crps = scales * (z * (2 * cdf - 1) + 2 * phi - 1 / math.sqrt(math.pi))
    output: dict[str, Any] = {"status": "available", "crps": float(np.mean(crps))}
    calibration_errors: list[float] = []
    for level in COVERAGE_LEVELS:
        quantile = normal.inv_cdf((1 + level) / 2)
        covered = float(np.mean(np.abs(errors) <= quantile * scales))
        suffix = int(level * 100)
        output[f"coverage_{suffix}"] = covered
        output[f"interval_width_{suffix}"] = float(np.mean(2 * quantile * scales))
        calibration_errors.append(abs(covered - level))
    output["mace"] = float(np.mean(calibration_errors))
    order = np.argsort(scales, kind="stable")
    risks: list[float] = []
    retained: list[float] = []
    for fraction in RISK_RETENTION:
        count = max(1, int(math.ceil(fraction * len(order))))
        risks.append(_rmse(errors[order[:count]]))
        retained.append(count / len(order))
    output["risk_coverage_area"] = float(np.trapezoid(risks, retained) / (retained[-1] - retained[0]))
    return output


def _termination_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    clipped = np.clip(probability, 1e-12, 1 - 1e-12)
    return {
        "brier": float(np.mean((probability - target) ** 2)),
        "log_loss": float(np.mean(-(target * np.log(clipped) + (1 - target) * np.log(1 - clipped)))),
        "accuracy_at_0_5": float(np.mean((probability >= 0.5) == target)),
        "positive_count": int(target.sum()),
    }


def _rmse(values: np.ndarray) -> float:
    return float(math.sqrt(float(np.mean(values ** 2))))


def _ess(weights: np.ndarray) -> float:
    denominator = float(np.sum(weights ** 2))
    return 0.0 if denominator == 0 else float(np.sum(weights) ** 2 / denominator)
