from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .errors import ReleaseContractError


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_transition_submission(path: Path, config_dir: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("rows"), list) or not data["rows"]:
        raise ReleaseContractError("Transition submission must contain nonempty rows")
    configs = {}
    for config_path in sorted(config_dir.glob("*.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        configs[str(config["task_id"])] = (config, file_sha256(config_path))
    required = {
        "task_id", "task_config_sha256", "benchmark_version", "model_id", "source_commit",
        "horizon", "metric_name", "metric_value", "observed_target_count", "claim_boundary_acknowledged",
    }
    for row in data["rows"]:
        if not isinstance(row, dict) or required - set(row):
            raise ReleaseContractError("Transition submission row is missing frozen identity fields")
        task = str(row["task_id"])
        if task not in configs:
            raise ReleaseContractError(f"Unknown transition task: {task}")
        config, digest = configs[task]
        if row["task_config_sha256"] != digest or row["benchmark_version"] != config["benchmark_version"]:
            raise ReleaseContractError(f"Task/version hash mismatch: {task}")
        if row["horizon"] not in {"one_step", "recursive_40h", "task_specific_44h_sensitivity"}:
            raise ReleaseContractError("Unsupported transition horizon")
        if not isinstance(row["observed_target_count"], int) or row["observed_target_count"] <= 0:
            raise ReleaseContractError("observed_target_count must be positive")
        try:
            metric = float(row["metric_value"])
        except (TypeError, ValueError) as error:
            raise ReleaseContractError("Transition metric must be numeric") from error
        if not math.isfinite(metric) or row["claim_boundary_acknowledged"] is not True:
            raise ReleaseContractError("Transition metric or claim boundary is invalid")
    return {"valid_rows": len(data["rows"]), "task_hashes_verified": True, "aggregate_only": True}
