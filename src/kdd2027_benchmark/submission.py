from __future__ import annotations

import json
import math
from pathlib import Path
from typing import cast

from . import BENCHMARK_VERSION
from .errors import ReleaseContractError


TRACK_METRICS = {
    "point_transition": {"one_step_rmse", "one_step_mae", "open_loop_rmse", "open_loop_mae"},
    "uncertainty": {"nll", "cov90", "width90"},
    "policy_diagnostic": {"behavior_nll", "ece", "ess", "ess_fraction", "wis", "wpdis"},
}
FIDELITY = {"exact", "adapted", "proxy", "local_control"}


def validate_submission(path: Path, task_ids: set[str]) -> dict[str, object]:
    try:
        value = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseContractError(f"Cannot read submission: {error}") from error
    if not isinstance(value, dict):
        raise ReleaseContractError("Submission must be a JSON object")
    data = cast(dict[str, object], value)
    if data.get("benchmark_version") != BENCHMARK_VERSION:
        raise ReleaseContractError("Submission benchmark version mismatch")
    rows = data.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ReleaseContractError("Submission rows must be a non-empty list")
    for raw in rows:
        if not isinstance(raw, dict):
            raise ReleaseContractError("Submission row must be an object")
        row = cast(dict[str, object], raw)
        task, track, fidelity = str(row.get("task_id", "")), str(row.get("track", "")), str(row.get("implementation_fidelity", ""))
        if task not in task_ids:
            raise ReleaseContractError(f"Unknown submission task: {task}")
        if track not in TRACK_METRICS:
            raise ReleaseContractError(f"Unsupported submission track: {track}")
        if fidelity not in FIDELITY:
            raise ReleaseContractError(f"Invalid implementation fidelity: {fidelity}")
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            raise ReleaseContractError("Submission metrics must be an object")
        missing = TRACK_METRICS[track] - set(metrics)
        if missing:
            raise ReleaseContractError(f"Missing submission metrics: {','.join(sorted(missing))}")
        for key in TRACK_METRICS[track]:
            try:
                number = float(metrics[key])
            except (TypeError, ValueError) as error:
                raise ReleaseContractError(f"Non-numeric submission metric: {key}") from error
            if not math.isfinite(number):
                raise ReleaseContractError(f"Non-finite submission metric: {key}")
    if data.get("claim_boundary_acknowledged") is not True:
        raise ReleaseContractError("Submission must acknowledge the claim boundary")
    return {"valid_rows": len(rows), "benchmark_version": BENCHMARK_VERSION}
