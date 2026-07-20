from __future__ import annotations

from pathlib import Path
from typing import cast

from .config import validate_config_directory
from .errors import ReleaseContractError
from .identity import feature_order_sha256, file_sha256
from .schema import schema_path, validate_json_file


def validate_submission(path: Path, config_dir: Path) -> dict[str, object]:
    data = validate_json_file(path, schema_path("leaderboard_submission"))
    configs = validate_config_directory(config_dir)
    root = config_dir.parent.parent
    by_task = {
        str(config["task_id"]): (config, config_dir / f"{config['task_id']}.json")
        for config in configs
    }
    rows = cast(list[dict[str, object]], data["rows"])
    identities: set[tuple[str, str, str, str, str]] = set()
    for row in rows:
        task = str(row["task_id"])
        if task not in by_task:
            raise ReleaseContractError(f"Unknown submission task: {task}")
        config, config_path = by_task[task]
        if row["task_config_sha256"] != file_sha256(config_path):
            raise ReleaseContractError(f"Submission task hash mismatch: {task}")
        if row["feature_order_sha256"] != feature_order_sha256(config, root):
            raise ReleaseContractError(f"Submission feature order mismatch: {task}")
        track = str(row["track"])
        allowed = config.get("allowed_tracks")
        if not isinstance(allowed, list) or track not in allowed:
            raise ReleaseContractError(f"Track {track} is not allowed for task {task}")
        action = config.get("action")
        timing = config.get("timing")
        if not isinstance(action, dict) or row["action_view"] != action.get("primary_view"):
            raise ReleaseContractError(f"Submission action view does not match task {task}")
        if not isinstance(timing, dict) or row["timing_view"] != timing.get("primary"):
            raise ReleaseContractError(f"Submission timing view does not match task {task}")
        identity = (task, track, str(row["model_id"]), str(row["horizon"]), str(row["source_commit"]))
        if identity in identities:
            raise ReleaseContractError(f"Duplicate submission row identity: {identity}")
        identities.add(identity)
    return {
        "valid_rows": len(rows),
        "schema_version": data["schema_version"],
        "schema_bound": True,
        "task_hashes_verified": True,
        "feature_order_verified": True,
    }
