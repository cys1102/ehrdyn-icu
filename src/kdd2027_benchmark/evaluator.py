from __future__ import annotations

import csv
import math
from pathlib import Path

from . import BENCHMARK_VERSION, CLAIM_BOUNDARY
from .errors import ReleaseContractError

MEASUREMENT_COLUMNS = {
    "step_index",
    "feature_name",
    "feature_group",
    "current_value",
    "previous_value",
    "target_value",
    "prediction_mean",
    "prediction_std",
    "action_index",
    "reward_component",
}
REQUIRED_COLUMNS = MEASUREMENT_COLUMNS | {"synthetic_episode_key"}


def evaluate_fixture(path: Path, task_id: str) -> dict[str, str | int | float | bool]:
    return evaluate_predictions(path, task_id, "synthetic_episode_key", synthetic=True)


def evaluate_predictions(
    path: Path,
    task_id: str,
    episode_key_column: str,
    *,
    synthetic: bool = False,
) -> dict[str, str | int | float | bool]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = (MEASUREMENT_COLUMNS | {episode_key_column}) - set(reader.fieldnames or ())
        if missing:
            raise ReleaseContractError(f"Prediction file is missing columns: {','.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ReleaseContractError("Prediction file must contain at least one row")
    if synthetic and any(not row[episode_key_column].startswith("syn-") for row in rows):
        raise ReleaseContractError("Fixture must contain synthetic-only episode keys")
    errors = [_float(row, "prediction_mean") - _float(row, "target_value") for row in rows]
    dynamic = [error for error, row in zip(errors, rows, strict=True) if abs(_float(row, "target_value") - _float(row, "current_value")) > 0.05]
    sparse = [error for error, row in zip(errors, rows, strict=True) if row["feature_group"] == "sparse_lab"]
    if not dynamic or not sparse:
        raise ReleaseContractError("Fixture must contain dynamic and sparse-lab evaluation cells")
    standard = [(_float(row, "target_value") - _float(row, "prediction_mean")) / _positive_std(row) for row in rows]
    widths = [2.0 * 1.6448536269514722 * _positive_std(row) for row in rows]
    widths80 = [2.0 * 1.2815515655446004 * _positive_std(row) for row in rows]
    absolute = [abs(error) for error in errors]
    variances = [_positive_std(row) ** 2 for row in rows]
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "task_id": task_id,
        "synthetic_fixture": synthetic,
        "row_count": len(rows),
        "episode_count": len({row[episode_key_column] for row in rows}),
        "feature_count": len({row["feature_name"] for row in rows}),
        "overall_rmse": _rmse(errors),
        "overall_mae": _mae(errors),
        "dynamic_rmse": _rmse(dynamic),
        "sparse_lab_rmse": _rmse(sparse),
        "gaussian_nll": _nll(rows),
        "gaussian_crps": _crps(rows),
        "cov80": sum(abs(value) <= 1.2815515655446004 for value in standard) / len(standard),
        "cov90": sum(abs(value) <= 1.6448536269514722 for value in standard) / len(standard),
        "width80": sum(widths80) / len(widths80),
        "width90": sum(widths) / len(widths),
        "interval_score90": _interval_score(rows, 1.6448536269514722, 0.1),
        "uncertainty_absolute_error_spearman": _spearman(variances, absolute),
        "action_count_observed": len({row["action_index"] for row in rows}),
        "reward_nonzero_density": sum(abs(_float(row, "reward_component")) > 0 for row in rows) / len(rows),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _float(row: dict[str, str], name: str) -> float:
    try:
        value = float(row[name])
    except ValueError as error:
        raise ReleaseContractError(f"Fixture has a non-numeric {name}") from error
    if not math.isfinite(value):
        raise ReleaseContractError(f"Fixture has a non-finite {name}")
    return value


def _positive_std(row: dict[str, str]) -> float:
    value = _float(row, "prediction_std")
    if value <= 0.0:
        raise ReleaseContractError("Fixture prediction_std must be positive")
    return value


def _rmse(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def _mae(values: list[float]) -> float:
    return sum(abs(value) for value in values) / len(values)


def _nll(rows: list[dict[str, str]]) -> float:
    total = 0.0
    for row in rows:
        variance = _positive_std(row) ** 2
        error = _float(row, "target_value") - _float(row, "prediction_mean")
        total += 0.5 * (math.log(2.0 * math.pi * variance) + error * error / variance)
    return total / len(rows)


def _crps(rows: list[dict[str, str]]) -> float:
    total = 0.0
    for row in rows:
        sigma = _positive_std(row)
        z = (_float(row, "target_value") - _float(row, "prediction_mean")) / sigma
        phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        total += sigma * (z * (2.0 * cdf - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))
    return total / len(rows)


def _interval_score(rows: list[dict[str, str]], quantile: float, alpha: float) -> float:
    values = []
    for row in rows:
        mean = _float(row, "prediction_mean")
        width = quantile * _positive_std(row)
        lower, upper, target = mean - width, mean + width, _float(row, "target_value")
        values.append((upper - lower) + 2.0 / alpha * max(lower - target, 0.0) + 2.0 / alpha * max(target - upper, 0.0))
    return sum(values) / len(values)


def _spearman(left: list[float], right: list[float]) -> float:
    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=values.__getitem__)
        output = [0.0] * len(values)
        for rank, index in enumerate(order):
            output[index] = float(rank)
        return output
    x, y = ranks(left), ranks(right)
    mean_x, mean_y = sum(x) / len(x), sum(y) / len(y)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y, strict=True))
    denominator = math.sqrt(sum((a - mean_x) ** 2 for a in x) * sum((b - mean_y) ** 2 for b in y))
    return numerator / denominator if denominator else 0.0
