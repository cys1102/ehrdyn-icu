from __future__ import annotations

from pathlib import Path
from typing import cast

from . import CLAIM_BOUNDARY
from .errors import ReleaseContractError
from .schema import schema_path, validate_json_file


def write_aggregate_report(input_dir: Path, output: Path) -> int:
    records: list[dict[str, object]] = []
    for path in sorted(input_dir.glob("*.json")):
        value = cast(object, validate_json_file(path, schema_path("aggregate_metrics")))
        if isinstance(value, dict) and "overall_rmse" in value:
            records.append(cast(dict[str, object], value))
    if not records:
        raise ReleaseContractError("No aggregate metric JSON files found")
    lines = [
        "# KDD 2027 Synthetic Aggregate Report",
        "",
        "| task_id | baseline | RMSE | MAE | Cov90 | Width90 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in records:
        line = f"| {row['task_id']} | {row.get('baseline', 'provided_predictions')} | "
        line += f"{_number(row['overall_rmse']):.6f} | {_number(row['overall_mae']):.6f} | "
        line += f"{_number(row['cov90']):.6f} | {_number(row['width90']):.6f} |"
        lines.append(line)
    lines.extend(("", f"Claim boundary: {CLAIM_BOUNDARY}", ""))
    output.parent.mkdir(parents=True, exist_ok=True)
    _ = output.write_text("\n".join(lines), encoding="utf-8")
    return len(records)


def _number(value: object) -> float:
    if isinstance(value, int | float | str):
        return float(value)
    raise ReleaseContractError("Aggregate report metric must be numeric")
