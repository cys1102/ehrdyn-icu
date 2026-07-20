from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from .errors import ReleaseContractError
from .identity import feature_order_sha256, file_sha256
from .schema import schema_path, validate_json_file


def validate_transition_submission(path: Path, config_dir: Path) -> dict[str, object]:
    data = validate_json_file(path, schema_path("transition_submission"))
    root = config_dir.parent.parent
    configs: dict[str, tuple[dict[str, object], Path]] = {}
    for config_path in sorted(config_dir.glob("*.json")):
        config = cast(dict[str, object], json.loads(config_path.read_text(encoding="utf-8")))
        configs[str(config["task_id"])] = (config, config_path)
    rows = cast(list[dict[str, object]], data["rows"])
    identities: set[tuple[str, str, str, str]] = set()
    counts: dict[tuple[str, str, str], int] = {}
    for row in rows:
        task = str(row["task_id"])
        if task not in configs:
            raise ReleaseContractError(f"Unknown transition task: {task}")
        config, config_path = configs[task]
        if row["task_config_sha256"] != file_sha256(config_path):
            raise ReleaseContractError(f"Transition task hash mismatch: {task}")
        if row["feature_order_sha256"] != feature_order_sha256(config, root):
            raise ReleaseContractError(f"Transition feature order mismatch: {task}")
        identity = (task, str(row["model_id"]), str(row["horizon"]), str(row["metric_name"]))
        if identity in identities:
            raise ReleaseContractError(f"Duplicate transition metric identity: {identity}")
        identities.add(identity)
        count_identity = identity[:3]
        count = int(row["observed_target_count"])
        if count_identity in counts and counts[count_identity] != count:
            raise ReleaseContractError(f"Observed target count mismatch: {count_identity}")
        counts[count_identity] = count
    return {
        "valid_rows": len(rows),
        "schema_version": data["schema_version"],
        "schema_bound": True,
        "task_hashes_verified": True,
        "feature_order_verified": True,
        "aggregate_only": True,
    }
