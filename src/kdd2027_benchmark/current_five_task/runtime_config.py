from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RuntimeConfigError(RuntimeError):
    """Raised when the source-closed five-task runtime config is invalid."""


DEFAULT_RUNTIME_CONFIG = Path(__file__).with_name("runtime_config.json")


def load_runtime_config(path: Path | None = None) -> dict[str, Any]:
    source = path or DEFAULT_RUNTIME_CONFIG
    config = json.loads(source.read_text(encoding="utf-8"))
    temporal = config.get("temporal", {})
    required = {
        "bin_hours": 4,
        "episode_pre_anchor_hours": 24,
        "episode_post_anchor_hours": 48,
        "episode_window_hours": 72,
        "episode_bins": 18,
        "raw_rv01r_post_base_anchor_extraction_hours": 96,
        "sepsis_max_base_to_final_anchor_shift_hours": 48,
        "longest_recursive_target_hours": 44,
    }
    mismatches = {key: (temporal.get(key), expected) for key, expected in required.items() if temporal.get(key) != expected}
    if mismatches:
        raise RuntimeConfigError(f"frozen temporal contract mismatch: {mismatches}")
    if temporal["episode_window_hours"] != temporal["episode_pre_anchor_hours"] + temporal["episode_post_anchor_hours"]:
        raise RuntimeConfigError("episode window is not pre-anchor plus post-anchor")
    if temporal["episode_bins"] * temporal["bin_hours"] != temporal["episode_window_hours"]:
        raise RuntimeConfigError("episode bins do not span the episode window")
    if temporal["raw_rv01r_post_base_anchor_extraction_hours"] != temporal["sepsis_max_base_to_final_anchor_shift_hours"] + temporal["episode_post_anchor_hours"]:
        raise RuntimeConfigError("raw extraction buffer does not cover anchor shift plus final post-anchor window")
    if temporal["longest_recursive_target_hours"] != temporal["episode_post_anchor_hours"] - temporal["bin_hours"]:
        raise RuntimeConfigError("recursive target horizon is inconsistent with state/action/target alignment")
    expected_tasks = {"sepsis", "respiratory_support", "shock", "aki", "heart_failure"}
    if set(config.get("lineage_router", {})) != expected_tasks:
        raise RuntimeConfigError("lineage router must contain exactly the retained five tasks")
    if config.get("mimiciv_version") != "3.1":
        raise RuntimeConfigError("only MIMIC-IV 3.1 is supported")
    chunk_rows = config.get("runtime", {}).get("high_volume_chunk_rows")
    if not isinstance(chunk_rows, int) or chunk_rows <= 0:
        raise RuntimeConfigError("high-volume chunk rows must be a positive integer")
    return config


RUNTIME_CONFIG = load_runtime_config()
