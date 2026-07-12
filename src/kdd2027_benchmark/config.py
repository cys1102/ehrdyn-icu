from __future__ import annotations

import json
from pathlib import Path
from typing import Final, cast

from . import BENCHMARK_VERSION
from .errors import ReleaseContractError
from .split import SPLIT_CONTRACT_HASH

REQUIRED_FIELDS: Final = (
    "benchmark_version",
    "task_id",
    "tier",
    "cohort_anchor",
    "episode_contract",
    "feature_dictionary",
    "action",
    "rewards",
    "timing",
    "split_contract_hash",
    "claim_boundary",
    "task_role",
    "allowed_tracks",
)


def load_task_config(path: Path) -> dict[str, object]:
    try:
        value = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseContractError(f"Cannot read task config {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise ReleaseContractError(f"Task config must be an object: {path.name}")
    mapping = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise ReleaseContractError(f"Task config keys must be strings: {path.name}")
    return cast(dict[str, object], mapping)


def validate_task_config(path: Path) -> dict[str, object]:
    config = load_task_config(path)
    missing = [field for field in REQUIRED_FIELDS if field not in config]
    if missing:
        raise ReleaseContractError(f"Missing required task fields in {path.name}: {','.join(missing)}")
    if config["benchmark_version"] != BENCHMARK_VERSION:
        raise ReleaseContractError(f"Benchmark version mismatch in {path.name}")
    if config["split_contract_hash"] != SPLIT_CONTRACT_HASH:
        raise ReleaseContractError(f"Split contract mismatch in {path.name}")
    episode = _mapping(config["episode_contract"], "episode_contract")
    if (episode.get("bin_hours"), episode.get("steps"), episode.get("pre_hours"), episode.get("post_hours")) != (4, 18, 24, 48):
        raise ReleaseContractError(f"Episode contract mismatch in {path.name}")
    action = _mapping(config["action"], "action")
    action_count = action.get("action_count")
    exclusion = config.get("task_role") == "no_rich_action_exclusion"
    if not isinstance(action_count, int) or (action_count < 2 and not (exclusion and action_count == 0)):
        raise ReleaseContractError(f"Invalid action count in {path.name}")
    timing = _mapping(config["timing"], "timing")
    if timing.get("primary") != "current_4h_window_exposure_predicts_next_4h_state" and not exclusion:
        raise ReleaseContractError(f"Timing contract mismatch in {path.name}")
    tracks = config.get("allowed_tracks")
    if not isinstance(tracks, list) or not tracks or "clinical_policy" in tracks:
        raise ReleaseContractError(f"Invalid allowed tracks in {path.name}")
    return config


def validate_config_directory(path: Path) -> list[dict[str, object]]:
    configs = [validate_task_config(item) for item in sorted(path.glob("*.json"))]
    if len(configs) != 7:
        raise ReleaseContractError(f"Expected seven frozen runnable task configs, found {len(configs)}")
    task_ids = {str(config["task_id"]) for config in configs}
    if len(task_ids) != len(configs):
        raise ReleaseContractError("Task identifiers must be unique")
    return configs


def _mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ReleaseContractError(f"{field} must be an object")
    return cast(dict[str, object], value)
