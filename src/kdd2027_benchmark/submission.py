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
}
FIDELITY = {"exact", "adapted", "proxy", "local_control"}
ROW_FIELDS = {
    "task_id", "track", "implementation_fidelity", "model_id", "metrics", "action_view",
    "timing_view", "horizon", "seed_count", "evaluator_version", "training_budget",
    "source_commit",
}


def validate_submission(path: Path, configs: list[dict[str, object]]) -> dict[str, object]:
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
    by_task = {str(config["task_id"]): config for config in configs}
    for raw in rows:
        if not isinstance(raw, dict):
            raise ReleaseContractError("Submission row must be an object")
        row = cast(dict[str, object], raw)
        missing_fields = ROW_FIELDS - set(row)
        if missing_fields:
            raise ReleaseContractError(f"Submission row is missing governance fields: {','.join(sorted(missing_fields))}")
        task, track, fidelity = str(row.get("task_id", "")), str(row.get("track", "")), str(row.get("implementation_fidelity", ""))
        if task not in by_task:
            raise ReleaseContractError(f"Unknown submission task: {task}")
        if track not in TRACK_METRICS:
            raise ReleaseContractError(f"Unsupported submission track: {track}")
        config = by_task[task]
        allowed = config.get("allowed_tracks")
        if not isinstance(allowed, list) or track not in allowed:
            raise ReleaseContractError(f"Track {track} is not allowed for task {task}")
        if fidelity not in FIDELITY:
            raise ReleaseContractError(f"Invalid implementation fidelity: {fidelity}")
        action = config.get("action")
        timing = config.get("timing")
        if not isinstance(action, dict) or row["action_view"] != action.get("primary_view"):
            raise ReleaseContractError(f"Submission action view does not match task {task}")
        if not isinstance(timing, dict) or row["timing_view"] != timing.get("primary"):
            raise ReleaseContractError(f"Submission timing view does not match task {task}")
        if row["horizon"] not in {"one_step", "conditional_recursive_17"}:
            raise ReleaseContractError("Submission horizon must be one_step or conditional_recursive_17")
        if not isinstance(row["seed_count"], int) or int(row["seed_count"]) < 1:
            raise ReleaseContractError("Submission seed_count must be a positive integer")
        if row["evaluator_version"] != BENCHMARK_VERSION:
            raise ReleaseContractError("Submission evaluator version mismatch")
        if not isinstance(row["training_budget"], str) or not row["training_budget"].strip():
            raise ReleaseContractError("Submission training budget is required")
        source_commit = str(row["source_commit"])
        if len(source_commit) < 7 or any(character not in "0123456789abcdef" for character in source_commit.lower()):
            raise ReleaseContractError("Submission source_commit must be a hexadecimal commit")
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
