"""Validate evaluator-produced successor aggregate receipts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import cast

from ..errors import ReleaseContractError
from . import (
    EVALUATION_RECEIPT_VERSION,
    FEATURE_NAMES,
    MIN_PUBLIC_OBSERVED_CELLS,
    MIN_PUBLIC_SUBJECT_CLUSTERS,
    MODES,
    SUCCESSOR_BENCHMARK_VERSION,
)


def validate_submission(path: Path, configs: list[dict[str, object]]) -> dict[str, object]:
    try:
        raw = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseContractError(f"Cannot read successor evaluator output: {error}") from error
    if not isinstance(raw, dict):
        raise ReleaseContractError("Successor submission must be an evaluator-produced JSON object")
    data = cast(dict[str, object], raw)
    if data.get("benchmark_version") != SUCCESSOR_BENCHMARK_VERSION:
        raise ReleaseContractError("Successor submission benchmark version mismatch")
    receipt_raw = data.get("evaluation_receipt")
    if not isinstance(receipt_raw, dict):
        raise ReleaseContractError("Self-reported aggregate rows are rejected; evaluator receipt is required")
    if data.get("claim_boundary") is None:
        raise ReleaseContractError("Successor evaluator output must retain the claim boundary")
    receipt = cast(dict[str, object], receipt_raw)
    required_receipt = {
        "receipt_version",
        "benchmark_version",
        "evaluator_version",
        "evaluation_contract_sha256",
        "prediction_file_sha256",
        "normalization_receipt_sha256",
        "aggregate_payload_sha256",
        "task_modes",
        "complete_frozen_cell_contract",
        "synthetic_fixture",
    }
    missing = required_receipt - set(receipt)
    if missing:
        raise ReleaseContractError("Successor evaluator receipt is incomplete: " + ",".join(sorted(missing)))
    if receipt["receipt_version"] != EVALUATION_RECEIPT_VERSION:
        raise ReleaseContractError("Successor evaluator receipt version mismatch")
    if receipt["benchmark_version"] != SUCCESSOR_BENCHMARK_VERSION:
        raise ReleaseContractError("Successor evaluator receipt benchmark mismatch")
    if receipt["evaluator_version"] != SUCCESSOR_BENCHMARK_VERSION:
        raise ReleaseContractError("Successor evaluator version mismatch")
    if receipt["complete_frozen_cell_contract"] is not True:
        raise ReleaseContractError("Successor evaluator receipt must attest complete frozen cell coverage")
    for field in (
        "evaluation_contract_sha256",
        "prediction_file_sha256",
        "normalization_receipt_sha256",
        "aggregate_payload_sha256",
    ):
        _sha256_text(receipt[field], field)
    if data.get("prediction_file_sha256") != receipt["prediction_file_sha256"]:
        raise ReleaseContractError("Prediction hash disagrees with evaluator receipt")
    if data.get("normalization_receipt_sha256") != receipt["normalization_receipt_sha256"]:
        raise ReleaseContractError("Normalization hash disagrees with evaluator receipt")
    payload = {key: value for key, value in data.items() if key != "evaluation_receipt"}
    if _json_sha256(payload) != receipt["aggregate_payload_sha256"]:
        raise ReleaseContractError("Successor evaluator aggregate payload hash mismatch")

    configs_by_task = {str(config["task_id"]): config for config in configs}
    task_modes = receipt["task_modes"]
    if not isinstance(task_modes, list) or not task_modes:
        raise ReleaseContractError("Successor evaluator receipt has no task-mode coverage")
    coverage_pairs: set[tuple[str, str]] = set()
    for item in task_modes:
        if not isinstance(item, dict):
            raise ReleaseContractError("Invalid task-mode coverage row")
        row = cast(dict[str, object], item)
        task, mode = str(row.get("task_id", "")), str(row.get("mode", ""))
        if task not in configs_by_task or mode not in MODES:
            raise ReleaseContractError("Unknown task or mode in evaluator receipt")
        if row.get("task_version") != configs_by_task[task]["task_version"]:
            raise ReleaseContractError("Task version mismatch in evaluator receipt")
        pair = (task, mode)
        if pair in coverage_pairs:
            raise ReleaseContractError("Duplicate task-mode coverage in evaluator receipt")
        coverage_pairs.add(pair)
        if row.get("feature_names") != sorted(FEATURE_NAMES):
            raise ReleaseContractError("Evaluator receipt feature contract mismatch")
        _sha256_text(row.get("cell_set_sha256"), "cell_set_sha256")

    metrics = data.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ReleaseContractError("Successor evaluator output has no primary metric rows")
    synthetic = receipt["synthetic_fixture"] is True
    if data.get("synthetic_fixture") is not synthetic:
        raise ReleaseContractError("Synthetic status disagrees with evaluator receipt")
    seen_metrics: set[tuple[str, str, str]] = set()
    for raw_metric in metrics:
        if not isinstance(raw_metric, dict):
            raise ReleaseContractError("Invalid successor primary metric row")
        metric = cast(dict[str, object], raw_metric)
        task, mode, method = (
            str(metric.get("task_id", "")),
            str(metric.get("mode", "")),
            str(metric.get("method_id", "")),
        )
        if (task, mode) not in coverage_pairs or not method:
            raise ReleaseContractError("Primary metric row is outside evaluator receipt coverage")
        key = (task, mode, method)
        if key in seen_metrics:
            raise ReleaseContractError("Duplicate successor primary metric row")
        seen_metrics.add(key)
        for field in ("normalized_rmse", "mae"):
            if _finite(metric.get(field), field) < 0.0:
                raise ReleaseContractError(f"Successor metric must be non-negative: {field}")
        observed_cells = _positive_integer(metric.get("observed_cells"), "observed_cells")
        subject_clusters = _positive_integer(metric.get("subject_clusters"), "subject_clusters")
        if not synthetic:
            if observed_cells < MIN_PUBLIC_OBSERVED_CELLS:
                raise ReleaseContractError("Aggregate disclosure floor not met: observed_cells")
            if subject_clusters < MIN_PUBLIC_SUBJECT_CLUSTERS:
                raise ReleaseContractError("Aggregate disclosure floor not met: subject_clusters")
    return {
        "valid_metric_rows": len(metrics),
        "task_modes": len(coverage_pairs),
        "benchmark_version": SUCCESSOR_BENCHMARK_VERSION,
        "evaluator_receipt_verified": True,
        "synthetic_fixture": synthetic,
    }


def _sha256_text(value: object, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text.lower()):
        raise ReleaseContractError(f"Invalid SHA-256 field: {name}")
    return text


def _finite(value: object, name: str) -> float:
    try:
        number = float(cast(str | int | float, value))
    except (TypeError, ValueError) as error:
        raise ReleaseContractError(f"Non-numeric successor metric: {name}") from error
    if not math.isfinite(number):
        raise ReleaseContractError(f"Non-finite successor metric: {name}")
    return number


def _positive_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or value < 1:
        raise ReleaseContractError(f"Successor count metric must be a positive integer: {name}")
    return value


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
