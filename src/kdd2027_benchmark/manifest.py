from __future__ import annotations

import csv
import json
from pathlib import Path

from .errors import ReleaseContractError


TASK_COLUMNS = {
    "task_id", "paper_role", "config_path", "clinical_packet_path", "primary_action_view",
    "evidence_task_selector", "expected_episodes", "expected_windows",
}
CONTRACT_COLUMNS = {
    "contract_id", "task_id", "config_path", "action_view", "action_level",
    "timing_convention", "frozen_reference", "leakage_negative_control",
    "leaderboard_selector",
}


def validate_paper_manifests(task_manifest: Path, contract_manifest: Path, evidence: Path) -> dict[str, int | bool]:
    task_rows = _read(task_manifest, TASK_COLUMNS)
    contract_rows = _read(contract_manifest, CONTRACT_COLUMNS)
    evidence_rows = _read(evidence, {"contract_id", "frozen_task_id", "baseline_id"})
    if len(task_rows) != 7:
        raise ReleaseContractError(f"Expected seven paper task rows, found {len(task_rows)}")
    primary = [row for row in task_rows if row["paper_role"] == "primary"]
    if len(primary) != 5:
        raise ReleaseContractError(f"Expected five primary paper tasks, found {len(primary)}")
    task_ids = {row["task_id"] for row in task_rows}
    if len(task_ids) != len(task_rows):
        raise ReleaseContractError("Paper task identifiers must be unique")
    if len(contract_rows) != 41 or len({row["contract_id"] for row in contract_rows}) != 41:
        raise ReleaseContractError("Paper contract manifest must contain 41 unique contracts")
    if any(row["task_id"] not in task_ids for row in contract_rows):
        raise ReleaseContractError("Contract manifest references an unknown paper task")
    evidence_contracts = {row["contract_id"] for row in evidence_rows}
    manifest_contracts = {row["contract_id"] for row in contract_rows}
    if evidence_contracts != manifest_contracts:
        raise ReleaseContractError("Contract manifest and public leaderboard selectors differ")
    if len(evidence_rows) != 533:
        raise ReleaseContractError(f"Expected 533 public leaderboard rows, found {len(evidence_rows)}")
    evidence_tasks = {row["frozen_task_id"] for row in evidence_rows}
    if evidence_tasks != {row["task_id"] for row in primary}:
        raise ReleaseContractError("Public leaderboard rows must map exactly to the five primary tasks")
    for row in task_rows:
        config_path = _resolve(task_manifest, row["config_path"])
        packet_path = _resolve(task_manifest, row["clinical_packet_path"])
        if not config_path.exists():
            raise ReleaseContractError(f"Missing task config: {row['config_path']}")
        if not packet_path.exists():
            raise ReleaseContractError(f"Missing clinical packet: {row['clinical_packet_path']}")
        config = _json(config_path)
        action = config.get("action")
        if config.get("task_id") != row["task_id"] or row["evidence_task_selector"] != row["task_id"]:
            raise ReleaseContractError(f"Task manifest selector mismatch: {row['task_id']}")
        if not isinstance(action, dict) or action.get("primary_view") != row["primary_action_view"]:
            raise ReleaseContractError(f"Task manifest action mismatch: {row['task_id']}")
        try:
            episodes, windows = int(row["expected_episodes"]), int(row["expected_windows"])
        except ValueError as error:
            raise ReleaseContractError(f"Invalid expected scale for {row['task_id']}") from error
        if episodes <= 0 or windows != episodes * 18:
            raise ReleaseContractError(f"Task manifest scale mismatch: {row['task_id']}")
    return {
        "pass": True,
        "paper_tasks": len(task_rows),
        "primary_tasks": len(primary),
        "contracts": len(contract_rows),
        "leaderboard_rows": len(evidence_rows),
    }


def _read(path: Path, required: set[str]) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ReleaseContractError(f"{path.name} is missing columns: {','.join(sorted(missing))}")
            return list(reader)
    except OSError as error:
        raise ReleaseContractError(f"Cannot read manifest {path}: {error}") from error


def _resolve(manifest: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest.parents[1] / path


def _json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseContractError(f"Cannot read task config {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReleaseContractError(f"Task config is not an object: {path}")
    return value
