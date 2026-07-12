from __future__ import annotations

import csv
from pathlib import Path

from .errors import ReleaseContractError
from .evaluator import REQUIRED_COLUMNS, evaluate_fixture

BASELINES = {"persistence": "current_value", "previous_window": "previous_value"}


def run_baseline(fixture: Path, output_fixture: Path, task_id: str, baseline: str) -> dict[str, str | int | float | bool]:
    source_column = BASELINES.get(baseline)
    if source_column is None:
        raise ReleaseContractError(f"Unsupported baseline: {baseline}")
    with fixture.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        if REQUIRED_COLUMNS - set(fields):
            raise ReleaseContractError("Baseline input does not match the synthetic fixture schema")
        rows = list(reader)
    output_fixture.parent.mkdir(parents=True, exist_ok=True)
    with output_fixture.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            copied = dict(row)
            copied["prediction_mean"] = copied[source_column]
            writer.writerow(copied)
    metrics = evaluate_fixture(output_fixture, task_id)
    metrics["baseline"] = baseline
    return metrics
