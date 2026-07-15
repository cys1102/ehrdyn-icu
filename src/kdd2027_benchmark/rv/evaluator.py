"""Aggregate-only evaluator for KDD-RV successor prediction rows."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import numpy as np

from ..errors import ReleaseContractError
from . import (
    CLAIM_BOUNDARY,
    EVALUATION_CONTRACT_VERSION,
    EVALUATION_RECEIPT_VERSION,
    FEATURE_NAMES,
    MODES,
    NORMALIZATION_RULE,
    REALIZED_ACTION_CLASSES,
    SUCCESSOR_BENCHMARK_VERSION,
    TASK_VERSIONS,
    TASKS,
)
FIXED_COLUMNS: Final = {
    "role",
    "task_id",
    "mode",
    "method_id",
    "transition_index",
    "horizon",
    "feature_name",
    "target_value",
    "target_observed",
    "prediction_mean",
    "prediction_std",
    "action_class",
}


@dataclass(frozen=True, slots=True)
class PredictionRow:
    cluster: str
    sequence: str
    role: str
    task: str
    mode: str
    method: str
    transition: int
    horizon: int
    feature: str
    target: float
    observed: bool
    prediction: float
    scale: float | None
    action: str

    @property
    def cell(self) -> tuple[str, str, int, str]:
        return self.cluster, self.sequence, self.transition, self.feature


@dataclass(frozen=True, slots=True)
class FeatureScale:
    mean: float
    population_std: float


def load_normalization(path: Path, *, synthetic: bool | None = None) -> dict[str, FeatureScale]:
    try:
        raw = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseContractError(f"Cannot read successor normalization receipt: {error}") from error
    if not isinstance(raw, dict):
        raise ReleaseContractError("Successor normalization receipt must be an object")
    data = cast(dict[str, object], raw)
    if data.get("benchmark_version") != SUCCESSOR_BENCHMARK_VERSION:
        raise ReleaseContractError("Successor normalization benchmark version mismatch")
    if data.get("fit_role") != "train" or data.get("rule") != NORMALIZATION_RULE:
        raise ReleaseContractError("Normalization must use the frozen train-only rule")
    if synthetic is True and data.get("synthetic") is not True:
        raise ReleaseContractError("Synthetic evaluation requires a synthetic normalization receipt")
    if synthetic is False:
        required_lineage = {
            "task_id",
            "task_version",
            "source_commit",
            "split_contract_sha256",
            "construction_receipt_sha256",
        }
        missing_lineage = required_lineage - set(data)
        if missing_lineage:
            raise ReleaseContractError(
                "Credentialed normalization receipt is missing lineage: " + ",".join(sorted(missing_lineage))
            )
        task = str(data["task_id"])
        if task not in TASK_VERSIONS or data["task_version"] != TASK_VERSIONS[task]:
            raise ReleaseContractError("Credentialed normalization task identity mismatch")
        for field in ("source_commit", "split_contract_sha256", "construction_receipt_sha256"):
            value = str(data[field])
            if len(value) != 64 and not (field == "source_commit" and len(value) == 40):
                raise ReleaseContractError(f"Credentialed normalization lineage has invalid {field}")
            if any(character not in "0123456789abcdef" for character in value.lower()):
                raise ReleaseContractError(f"Credentialed normalization lineage has non-hexadecimal {field}")
    features = data.get("features")
    if not isinstance(features, dict) or set(features) != set(FEATURE_NAMES):
        raise ReleaseContractError("Normalization receipt must contain the exact 33-feature set")
    result: dict[str, FeatureScale] = {}
    for name, value in cast(dict[str, object], features).items():
        if not isinstance(value, dict):
            raise ReleaseContractError(f"Invalid normalization row: {name}")
        row = cast(dict[str, object], value)
        mean = _finite_object(row.get("mean"), f"normalization mean for {name}")
        scale = _finite_object(row.get("population_std"), f"normalization scale for {name}")
        if scale < 1e-6:
            raise ReleaseContractError(f"Normalization scale must be at least 1e-6: {name}")
        result[name] = FeatureScale(mean, scale)
    return result


def evaluate_predictions(
    predictions: Path,
    normalization: Path,
    evaluation_contract: Path,
    *,
    cluster_key_column: str = "local_subject_key",
    sequence_key_column: str = "local_sequence_key",
    synthetic: bool = False,
    bootstrap_replicates: int = 1000,
    bootstrap_seed: int = 8813408,
) -> dict[str, object]:
    if bootstrap_replicates < 20:
        raise ReleaseContractError("At least 20 subject-cluster bootstrap replicates are required")
    scales = load_normalization(normalization, synthetic=synthetic)
    rows = _read_rows(predictions, cluster_key_column, sequence_key_column, synthetic)
    contract = _load_evaluation_contract(evaluation_contract, predictions, normalization, rows, synthetic)
    if not synthetic:
        normalization_data = cast(dict[str, object], json.loads(normalization.read_text(encoding="utf-8")))
        row_tasks = {row.task for row in rows}
        if row_tasks != {str(normalization_data["task_id"])}:
            raise ReleaseContractError("Credentialed evaluation accepts one normalization task per invocation")
    grouped: dict[tuple[str, str], list[PredictionRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.task, row.mode)].append(row)
    metrics: list[dict[str, object]] = []
    by_horizon: list[dict[str, object]] = []
    by_group: list[dict[str, object]] = []
    probability: list[dict[str, object]] = []
    bootstrap_point: list[dict[str, object]] = []
    bootstrap_persistence: list[dict[str, object]] = []
    leaders: list[dict[str, object]] = []
    for task_mode in sorted(grouped):
        task, mode = task_mode
        methods, method_rows = _paired_method_rows(grouped[task_mode])
        for method in methods:
            local = method_rows[method]
            metrics.append(_metric_row(task, mode, method, local, scales))
            for horizon in sorted({row.horizon for row in local}):
                selected = [row for row in local if row.horizon == horizon]
                by_horizon.append(_metric_row(task, mode, method, selected, scales, horizon=horizon))
            for group, feature_set in _feature_groups(task).items():
                selected = [row for row in local if row.feature in feature_set]
                by_group.append(_metric_row(task, mode, method, selected, scales, feature_group=group))
            if any(row.scale is not None for row in local):
                if any(row.scale is None for row in local if row.observed):
                    raise ReleaseContractError(f"Probabilistic scales are incomplete for {task}/{mode}/{method}")
                probability.append(_probability_row(task, mode, method, local, scales, "all"))
                for horizon in sorted({row.horizon for row in local}):
                    selected = [row for row in local if row.horizon == horizon]
                    probability.append(_probability_row(task, mode, method, selected, scales, horizon))
        point_rows, persistence_rows, leader_rows = _bootstrap(
            task,
            mode,
            methods,
            method_rows,
            scales,
            bootstrap_replicates,
            bootstrap_seed,
        )
        bootstrap_point.extend(point_rows)
        bootstrap_persistence.extend(persistence_rows)
        leaders.extend(leader_rows)
    payload: dict[str, object] = {
        "benchmark_version": SUCCESSOR_BENCHMARK_VERSION,
        "synthetic_fixture": synthetic,
        "prediction_file_sha256": _sha256(predictions),
        "normalization_receipt_sha256": _sha256(normalization),
        "scoring_unit": "normalized_observed_feature_cell_micro_average",
        "bootstrap_unit": "subject_cluster",
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "metrics": metrics,
        "metrics_by_horizon": by_horizon,
        "metrics_by_feature_group": by_group,
        "probabilistic_metrics": probability,
        "paired_subject_cluster_delta_vs_point_minimum": bootstrap_point,
        "paired_subject_cluster_delta_vs_persistence": bootstrap_persistence,
        "practical_leader_sets": leaders,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    receipt = {
        "receipt_version": EVALUATION_RECEIPT_VERSION,
        "benchmark_version": SUCCESSOR_BENCHMARK_VERSION,
        "evaluator_version": SUCCESSOR_BENCHMARK_VERSION,
        "evaluation_contract_sha256": _sha256(evaluation_contract),
        "prediction_file_sha256": _sha256(predictions),
        "normalization_receipt_sha256": _sha256(normalization),
        "aggregate_payload_sha256": _json_sha256(payload),
        "task_modes": contract["task_modes"],
        "complete_frozen_cell_contract": True,
        "synthetic_fixture": synthetic,
    }
    return {**payload, "evaluation_receipt": receipt}


def create_evaluation_contract(
    predictions: Path,
    normalization: Path,
    output: Path,
    *,
    cluster_key_column: str,
    sequence_key_column: str,
    synthetic: bool,
) -> dict[str, object]:
    _ = load_normalization(normalization, synthetic=synthetic)
    rows = _read_rows(predictions, cluster_key_column, sequence_key_column, synthetic)
    summaries = _task_mode_summaries(rows)
    contract: dict[str, object] = {
        "contract_version": EVALUATION_CONTRACT_VERSION,
        "benchmark_version": SUCCESSOR_BENCHMARK_VERSION,
        "synthetic_fixture": synthetic,
        "role": "synthetic" if synthetic else "sealed_test",
        "normalization_receipt_sha256": _sha256(normalization),
        "task_modes": summaries,
        "operator_attestation": "frozen_target_cell_contract_created_before_prediction_scoring",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return contract


def _read_rows(
    path: Path,
    cluster_key_column: str,
    sequence_key_column: str,
    synthetic: bool,
) -> list[PredictionRow]:
    if cluster_key_column == sequence_key_column or not cluster_key_column or not sequence_key_column:
        raise ReleaseContractError("Cluster and sequence key columns must be distinct and non-empty")
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = (FIXED_COLUMNS | {cluster_key_column, sequence_key_column}) - set(reader.fieldnames or ())
            if missing:
                raise ReleaseContractError(f"Successor prediction file is missing columns: {','.join(sorted(missing))}")
            raw_rows = list(reader)
    except OSError as error:
        raise ReleaseContractError(f"Cannot read successor predictions: {error}") from error
    if not raw_rows:
        raise ReleaseContractError("Successor prediction file must contain rows")
    output: list[PredictionRow] = []
    for raw in raw_rows:
        cluster = raw[cluster_key_column].strip()
        sequence = raw[sequence_key_column].strip()
        if not cluster or not sequence:
            raise ReleaseContractError("Local cluster and sequence keys must be non-empty")
        if synthetic and (not cluster.startswith("syn-") or not sequence.startswith("syn-")):
            raise ReleaseContractError("Synthetic fixtures require synthetic-only keys")
        role = raw["role"].strip()
        if role != ("synthetic" if synthetic else "sealed_test"):
            raise ReleaseContractError("Local evaluation accepts only sealed_test rows; fixtures use synthetic rows")
        task, mode, method = raw["task_id"].strip(), raw["mode"].strip(), raw["method_id"].strip()
        if task not in TASKS or mode not in MODES or not method:
            raise ReleaseContractError("Unknown successor task, mode, or empty method")
        feature = raw["feature_name"].strip()
        if feature not in FEATURE_NAMES:
            raise ReleaseContractError(f"Unknown successor feature: {feature}")
        transition = _integer(raw, "transition_index", minimum=0)
        horizon = _integer(raw, "horizon", minimum=1)
        observed = raw["target_observed"].strip()
        if observed not in {"0", "1"}:
            raise ReleaseContractError("target_observed must be 0 or 1")
        standard_deviation = raw["prediction_std"].strip()
        output.append(
            PredictionRow(
                cluster=cluster,
                sequence=sequence,
                role=role,
                task=task,
                mode=mode,
                method=method,
                transition=transition,
                horizon=horizon,
                feature=feature,
                target=_finite(raw, "target_value"),
                observed=observed == "1",
                prediction=_finite(raw, "prediction_mean"),
                scale=None if not standard_deviation else _positive_float(standard_deviation, "prediction_std"),
                action=raw["action_class"].strip(),
            )
        )
        if not output[-1].action:
            raise ReleaseContractError("action_class must be non-empty")
        if output[-1].action not in REALIZED_ACTION_CLASSES[task]:
            raise ReleaseContractError(f"Action class is outside the realized successor contract: {task}")
    return output


def _load_evaluation_contract(
    path: Path,
    predictions: Path,
    normalization: Path,
    rows: list[PredictionRow],
    synthetic: bool,
) -> dict[str, object]:
    try:
        raw = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseContractError(f"Cannot read successor evaluation contract: {error}") from error
    if not isinstance(raw, dict):
        raise ReleaseContractError("Successor evaluation contract must be an object")
    contract = cast(dict[str, object], raw)
    expected_scalars = {
        "contract_version": EVALUATION_CONTRACT_VERSION,
        "benchmark_version": SUCCESSOR_BENCHMARK_VERSION,
        "synthetic_fixture": synthetic,
        "role": "synthetic" if synthetic else "sealed_test",
        "normalization_receipt_sha256": _sha256(normalization),
        "operator_attestation": "frozen_target_cell_contract_created_before_prediction_scoring",
    }
    for field, expected in expected_scalars.items():
        if contract.get(field) != expected:
            raise ReleaseContractError(f"Successor evaluation contract mismatch: {field}")
    expected_summaries = _task_mode_summaries(rows)
    if contract.get("task_modes") != expected_summaries:
        raise ReleaseContractError("Prediction rows do not match the complete frozen evaluation cell contract")
    if _sha256(predictions) == _sha256(path):
        raise ReleaseContractError("Prediction and evaluation-contract files must be distinct")
    return contract


def _task_mode_summaries(rows: list[PredictionRow]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[PredictionRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.task, row.mode)].append(row)
    summaries: list[dict[str, object]] = []
    for task, mode in sorted(grouped):
        methods, method_rows = _paired_method_rows(grouped[(task, mode)])
        reference = method_rows[methods[0]]
        features = sorted({row.feature for row in reference})
        if features != sorted(FEATURE_NAMES):
            raise ReleaseContractError(f"Every task/mode must contain the exact 33-feature contract: {task}/{mode}")
        _validate_horizon_contract(task, mode, reference)
        summaries.append(
            {
                "task_id": task,
                "task_version": TASK_VERSIONS[task],
                "mode": mode,
                "cell_set_sha256": _cell_set_sha256(reference),
                "cells_per_method": len(reference),
                "observed_cells_per_method": sum(row.observed for row in reference),
                "subject_clusters": len({row.cluster for row in reference}),
                "sequences": len({row.sequence for row in reference}),
                "transitions": len({(row.sequence, row.transition) for row in reference}),
                "feature_names": features,
                "horizons": sorted({row.horizon for row in reference}),
                "action_classes": sorted({row.action for row in reference}),
            }
        )
    return summaries


def _validate_horizon_contract(task: str, mode: str, rows: list[PredictionRow]) -> None:
    by_transition: dict[tuple[str, int], set[int]] = defaultdict(set)
    for row in rows:
        by_transition[(row.sequence, row.transition)].add(row.horizon)
    if any(len(values) != 1 for values in by_transition.values()):
        raise ReleaseContractError(f"Features disagree on horizon within a transition: {task}/{mode}")
    by_sequence: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for (sequence, transition), values in by_transition.items():
        by_sequence[sequence].append((transition, next(iter(values))))
    for sequence_rows in by_sequence.values():
        previous_transition: int | None = None
        previous_horizon: int | None = None
        for transition, horizon in sorted(sequence_rows):
            if mode == "one_step":
                expected = 1
            elif previous_transition is not None and transition == previous_transition + 1:
                expected = cast(int, previous_horizon) + 1
            else:
                expected = 1
            if horizon != expected:
                raise ReleaseContractError(f"Invalid transition-to-horizon lineage for {task}/{mode}")
            previous_transition, previous_horizon = transition, horizon


def _cell_set_sha256(rows: list[PredictionRow]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item.cell):
        record = [
            row.cluster,
            row.sequence,
            str(row.transition),
            str(row.horizon),
            row.feature,
            "1" if row.observed else "0",
            format(row.target, ".17g"),
            row.action,
        ]
        digest.update(json.dumps(record, separators=(",", ":"), ensure_ascii=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _paired_method_rows(
    rows: list[PredictionRow],
) -> tuple[list[str], dict[str, list[PredictionRow]]]:
    method_order: list[str] = []
    indexed: dict[str, dict[tuple[str, str, int, str], PredictionRow]] = {}
    for row in rows:
        if row.method not in indexed:
            indexed[row.method] = {}
            method_order.append(row.method)
        if row.cell in indexed[row.method]:
            raise ReleaseContractError(f"Duplicate successor prediction cell for method {row.method}")
        indexed[row.method][row.cell] = row
    reference_cells = set(indexed[method_order[0]])
    for method in method_order:
        if set(indexed[method]) != reference_cells:
            raise ReleaseContractError("Methods must be evaluated on identical successor cells")
    for cell in reference_cells:
        reference = indexed[method_order[0]][cell]
        for method in method_order[1:]:
            candidate = indexed[method][cell]
            if (
                candidate.target,
                candidate.observed,
                candidate.horizon,
                candidate.action,
            ) != (
                reference.target,
                reference.observed,
                reference.horizon,
                reference.action,
            ):
                raise ReleaseContractError("Paired successor cells disagree on target, mask, horizon, or action")
    ordered = {
        method: sorted(indexed[method].values(), key=lambda row: row.cell)
        for method in method_order
    }
    return method_order, ordered


def _metric_row(
    task: str,
    mode: str,
    method: str,
    rows: list[PredictionRow],
    scales: dict[str, FeatureScale],
    *,
    horizon: int | None = None,
    feature_group: str | None = None,
) -> dict[str, object]:
    observed = [row for row in rows if row.observed]
    if not observed:
        raise ReleaseContractError(f"No observed cells for {task}/{mode}/{method}")
    errors = [_normalized_error(row, scales) for row in observed]
    result: dict[str, object] = {
        "task_id": task,
        "mode": mode,
        "method_id": method,
        "normalized_rmse": math.sqrt(sum(value * value for value in errors) / len(errors)),
        "mae": sum(abs(value) for value in errors) / len(errors),
        "observed_cells": len(observed),
        "transitions": len({(row.cluster, row.sequence, row.transition) for row in rows}),
        "subject_clusters": len({row.cluster for row in rows}),
        "action_classes_observed": len({row.action for row in rows}),
    }
    if horizon is not None:
        result["horizon"] = horizon
    if feature_group is not None:
        result["feature_group"] = feature_group
    return result


def _probability_row(
    task: str,
    mode: str,
    method: str,
    rows: list[PredictionRow],
    scales: dict[str, FeatureScale],
    horizon: str | int,
) -> dict[str, object]:
    observed = [row for row in rows if row.observed]
    if not observed or any(row.scale is None for row in observed):
        raise ReleaseContractError(f"No complete probabilistic cells for {task}/{mode}/{method}")
    errors = [_normalized_target(row, scales) - _normalized_prediction(row, scales) for row in observed]
    sigmas = [cast(float, row.scale) / scales[row.feature].population_std for row in observed]
    z_values = [error / sigma for error, sigma in zip(errors, sigmas, strict=True)]
    nll = [
        0.5 * z * z + math.log(sigma) + 0.5 * math.log(2.0 * math.pi)
        for z, sigma in zip(z_values, sigmas, strict=True)
    ]
    crps = []
    for z, sigma in zip(z_values, sigmas, strict=True):
        phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        crps.append(sigma * (z * (2.0 * cdf - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi)))
    critical = {"cov50": 0.67448975, "cov80": 1.28155157, "cov90": 1.64485363, "cov95": 1.95996398}
    coverage = {
        name: sum(abs(error) <= value * sigma for error, sigma in zip(errors, sigmas, strict=True)) / len(errors)
        for name, value in critical.items()
    }
    q90 = critical["cov90"]
    widths = [2.0 * q90 * sigma for sigma in sigmas]
    interval_scores = []
    for error, sigma, width in zip(errors, sigmas, widths, strict=True):
        lower_error = -q90 * sigma - error
        upper_error = error - q90 * sigma
        interval_scores.append(width + 20.0 * max(lower_error, 0.0) + 20.0 * max(upper_error, 0.0))
    result: dict[str, object] = {
        "task_id": task,
        "mode": mode,
        "horizon": horizon,
        "method_id": method,
        "native_nll": sum(nll) / len(nll),
        "gaussian_crps": sum(crps) / len(crps),
        **coverage,
        "width90": sum(widths) / len(widths),
        "interval_score90": sum(interval_scores) / len(interval_scores),
        "uncertainty_absolute_error_spearman": _spearman(sigmas, [abs(value) for value in errors]),
        "scoring_unit": "observed_feature_cell",
        "observed_cells": len(observed),
    }
    return result


def _bootstrap(
    task: str,
    mode: str,
    methods: list[str],
    method_rows: dict[str, list[PredictionRow]],
    scales: dict[str, FeatureScale],
    replicates: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    clusters = sorted({row.cluster for row in method_rows[methods[0]]})
    if len(clusters) < 2:
        raise ReleaseContractError(f"Subject-cluster bootstrap requires at least two clusters: {task}/{mode}")
    counts = {cluster: 0 for cluster in clusters}
    sse = {method: {cluster: 0.0 for cluster in clusters} for method in methods}
    for method in methods:
        for row in method_rows[method]:
            if row.observed:
                if method == methods[0]:
                    counts[row.cluster] += 1
                error = _normalized_error(row, scales)
                sse[method][row.cluster] += error * error
    if any(value == 0 for value in counts.values()):
        raise ReleaseContractError(f"Every subject cluster must contribute observed cells: {task}/{mode}")
    total_count = sum(counts.values())
    point = {method: math.sqrt(sum(sse[method].values()) / total_count) for method in methods}
    point_minimum = min(methods, key=lambda method: (point[method], methods.index(method)))
    rng = np.random.default_rng(seed)
    sampled_sets = [
        [clusters[int(index)] for index in rng.integers(0, len(clusters), size=len(clusters))]
        for _ in range(replicates)
    ]
    values: dict[str, list[float]] = {method: [] for method in methods}
    persistence_values: dict[str, list[float]] = {method: [] for method in methods}
    for sampled in sampled_sets:
        denominator = sum(counts[cluster] for cluster in sampled)
        reference_rmse = math.sqrt(sum(sse[point_minimum][cluster] for cluster in sampled) / denominator)
        persistence_rmse = (
            math.sqrt(sum(sse["persistence_locf"][cluster] for cluster in sampled) / denominator)
            if "persistence_locf" in methods
            else None
        )
        for method in methods:
            method_rmse = math.sqrt(sum(sse[method][cluster] for cluster in sampled) / denominator)
            values[method].append(method_rmse - reference_rmse)
            if persistence_rmse is not None:
                persistence_values[method].append(method_rmse - persistence_rmse)
    raw_p = {
        method: min(
            1.0,
            2.0 * min(
                (1 + sum(value <= 0.0 for value in values[method])) / (replicates + 1),
                (1 + sum(value >= 0.0 for value in values[method])) / (replicates + 1),
            ),
        )
        for method in methods
    }
    adjusted = _holm_adjust(raw_p)
    point_rows: list[dict[str, object]] = []
    for method in methods:
        point_rows.append(
            {
                "task_id": task,
                "mode": mode,
                "method_id": method,
                "reference": point_minimum,
                "point_delta_rmse": point[method] - point[point_minimum],
                "ci_lower_95": _quantile(values[method], 0.025),
                "ci_upper_95": _quantile(values[method], 0.975),
                "bootstrap_replicates": replicates,
                "bootstrap_unit": "subject_cluster",
                "bootstrap_seed": seed,
                "raw_two_sided_p": raw_p[method],
                "holm_adjusted_p": adjusted[method],
            }
        )
    persistence_rows: list[dict[str, object]] = []
    if "persistence_locf" in methods:
        for method in methods:
            persistence_rows.append(
                {
                    "task_id": task,
                    "mode": mode,
                    "method_id": method,
                    "reference": "persistence_locf",
                    "point_delta_rmse": point[method] - point["persistence_locf"],
                    "ci_lower_95": _quantile(persistence_values[method], 0.025),
                    "ci_upper_95": _quantile(persistence_values[method], 0.975),
                    "bootstrap_replicates": replicates,
                    "bootstrap_unit": "subject_cluster",
                }
            )
    leader_rows: list[dict[str, object]] = []
    for epsilon in (0.0, 0.01, 0.02):
        members = [
            method
            for method in methods
            if abs(point[method] - point[point_minimum]) <= epsilon + 1e-12
            and _quantile(values[method], 0.025) <= 0.0 <= _quantile(values[method], 0.975)
        ]
        separated = len(members) == 1 and all(adjusted[method] < 0.05 for method in methods if method not in members)
        leader_rows.append(
            {
                "task_id": task,
                "mode": mode,
                "epsilon": epsilon,
                "point_minimum": point_minimum,
                "leader_set": ";".join(members),
                "leader_count": len(members),
                "unique_winner_claim_allowed": False,
                "exploratory_holm_separation": separated,
                "inference_status": "descriptive_fixed_test_reference_no_winner_claim",
                "holm_alpha": 0.05,
            }
        )
    return point_rows, persistence_rows, leader_rows


def _feature_groups(task: str) -> dict[str, set[str]]:
    feature_names: set[str] = set(FEATURE_NAMES)
    vital: set[str] = {"heart_rate", "sbp", "mbp", "dbp", "respiratory_rate", "temperature_c", "spo2", "shock_index", "fio2"}
    score: set[str] = {"gcs_proxy", "sirs_proxy"}
    treatment: set[str] = {"urine_output", "mechanical_ventilation"}
    sparse: set[str] = {
        "lactate", "pao2", "paco2", "ph", "base_excess", "pao2_fio2", "ptt", "pt", "inr", "ast", "alt",
        "total_bilirubin", "ionized_calcium",
    }
    action_by_task: dict[str, set[str]] = {
        "sepsis": {"mbp", "shock_index", "lactate", "urine_output"},
        "respiratory": {"spo2", "respiratory_rate", "fio2", "pao2", "paco2", "pao2_fio2", "mechanical_ventilation"},
        "aki": {"creatinine", "bun", "urine_output", "mbp"},
        "af_flutter": {"heart_rate", "sbp", "mbp", "magnesium", "ionized_calcium", "creatinine"},
        "heart_failure": {"sbp", "mbp", "spo2", "creatinine", "bun", "urine_output"},
    }
    action = action_by_task[task]
    return {
        "physiology_without_score_or_treatment_context": feature_names - score - treatment,
        "sparse_lab": sparse,
        "vital": vital,
        "laboratory": feature_names - vital - score - treatment,
        "score": score,
        "treatment_context": treatment,
        "action_relevant": action,
    }


def _normalized_target(row: PredictionRow, scales: dict[str, FeatureScale]) -> float:
    spec = scales[row.feature]
    return (row.target - spec.mean) / spec.population_std


def _normalized_prediction(row: PredictionRow, scales: dict[str, FeatureScale]) -> float:
    spec = scales[row.feature]
    return (row.prediction - spec.mean) / spec.population_std


def _normalized_error(row: PredictionRow, scales: dict[str, FeatureScale]) -> float:
    return _normalized_prediction(row, scales) - _normalized_target(row, scales)


def _finite(row: dict[str, str], field: str) -> float:
    return _finite_object(row.get(field), field)


def _finite_object(value: object, field: str) -> float:
    try:
        number = float(cast(str | int | float, value))
    except (TypeError, ValueError) as error:
        raise ReleaseContractError(f"Non-numeric {field}") from error
    if not math.isfinite(number):
        raise ReleaseContractError(f"Non-finite {field}")
    return number


def _positive_float(value: str, field: str) -> float:
    number = _finite_object(value, field)
    if number <= 0.0:
        raise ReleaseContractError(f"{field} must be positive")
    return number


def _integer(row: dict[str, str], field: str, *, minimum: int) -> int:
    try:
        value = int(row[field])
    except ValueError as error:
        raise ReleaseContractError(f"Non-integer {field}") from error
    if value < minimum:
        raise ReleaseContractError(f"{field} must be at least {minimum}")
    return value


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    output = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + end - 1) / 2.0 + 1.0
        for index in order[start:end]:
            output[index] = average
        start = end
    return output


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or len(set(left)) == 1 or len(set(right)) == 1:
        return None
    x, y = _ranks(left), _ranks(right)
    mean_x, mean_y = sum(x) / len(x), sum(y) / len(y)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y, strict=True))
    denominator = math.sqrt(sum((a - mean_x) ** 2 for a in x) * sum((b - mean_y) ** 2 for b in y))
    return numerator / denominator if denominator else None


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=lambda key: (pvalues[key], key))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
