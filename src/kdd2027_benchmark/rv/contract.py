"""Validation for the isolated five-task successor contract."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Final, cast

from ..errors import ReleaseContractError
from . import CLAIM_BOUNDARY, CORE_FAMILIES, FEATURE_NAMES, MODES, NORMALIZATION_RULE, SUCCESSOR_BENCHMARK_VERSION, TASKS

TASK_ROW_HASHES: Final = {
    "sepsis": "22c2cbd59010a17da0759a321fb02af0a768e3ac25516daeac80952c8aca7c64",
    "respiratory": "21085b43c2bc7f7dca09fd39a02761e31eb3c09a7d00e04ff1aada17ffc56779",
    "aki": "0170a8a799a5dc33eee8c11d7a338f3e6268347729b29225de9c072160873794",
    "af_flutter": "2259a1da764d29b974d81812d65fabe92c1eb2e997e65b4e1f0b29def7bbe9d7",
    "heart_failure": "cfe9ce69c3c65c9670817613aafb81d32c4ac34515d4a637019a99c9ce3e359a",
}
TASK_VERSIONS: Final = {
    "sepsis": "KDD-RV-SEPSIS-SI-OD-v1.0.0",
    "respiratory": "KDD-RV-RESP-SUPPORT-v1.0.0",
    "aki": "KDD-RV-AKI-KDIGO-CR-RRT-v1.0.0",
    "af_flutter": "KDD-RV-AF-PRIOR-DX-RR-v1.0.0",
    "heart_failure": "KDD-RV-HF-PRIOR-DX-DECONG-v1.0.0",
}
REQUIRED_FIELDS: Final = {
    "benchmark_version",
    "task_id",
    "task_version",
    "task_contract_row_sha256",
    "status",
    "anchor_contract",
    "action_contract",
    "state_interval",
    "action_interval",
    "target_interval",
    "episode_contract",
    "role_contract",
    "clinical_review_status",
    "claim_boundary",
}


def load_task_config(path: Path) -> dict[str, object]:
    try:
        value = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseContractError(f"Cannot read successor task config {path.name}: {error}") from error
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ReleaseContractError(f"Successor task config must be an object: {path.name}")
    return cast(dict[str, object], value)


def validate_task_config(path: Path) -> dict[str, object]:
    config = load_task_config(path)
    missing = REQUIRED_FIELDS - set(config)
    if missing:
        raise ReleaseContractError(f"Missing successor task fields in {path.name}: {','.join(sorted(missing))}")
    task = str(config["task_id"])
    if config["benchmark_version"] != SUCCESSOR_BENCHMARK_VERSION:
        raise ReleaseContractError(f"Successor benchmark version mismatch in {path.name}")
    if task not in TASKS or config["task_version"] != TASK_VERSIONS[task]:
        raise ReleaseContractError(f"Successor task identity mismatch in {path.name}")
    if config["task_contract_row_sha256"] != TASK_ROW_HASHES[task]:
        raise ReleaseContractError(f"Frozen task-row hash mismatch in {path.name}")
    if config["status"] != "defined_pending_clinical_review":
        raise ReleaseContractError(f"Successor task status drift in {path.name}")
    if config["clinical_review_status"] != "pending_not_simulated":
        raise ReleaseContractError(f"Clinical review must remain pending in {path.name}")
    if config["claim_boundary"] != CLAIM_BOUNDARY:
        raise ReleaseContractError(f"Claim boundary mismatch in {path.name}")
    episode = _mapping(config["episode_contract"], "episode_contract")
    expected_episode = {
        "bin_hours": 4,
        "pre_anchor_history_hours": 24,
        "post_anchor_grid_hours": 48,
        "history_bins": 6,
        "fully_stay_contained": True,
    }
    if episode != expected_episode:
        raise ReleaseContractError(f"Episode contract mismatch in {path.name}")
    role = _mapping(config["role_contract"], "role_contract")
    if role.get("salt") != "KDD-RV-SUBJECT-ROLE-v1|" or role.get("ranges") != {
        "train": [0, 7000],
        "validation": [7000, 8500],
        "sealed_test": [8500, 10000],
    }:
        raise ReleaseContractError(f"Role contract mismatch in {path.name}")
    return config


def validate_config_directory(path: Path) -> list[dict[str, object]]:
    configs = [validate_task_config(item) for item in sorted(path.glob("*.json"))]
    if len(configs) != len(TASKS):
        raise ReleaseContractError(f"Expected five successor task configs, found {len(configs)}")
    if {str(config["task_id"]) for config in configs} != set(TASKS):
        raise ReleaseContractError("Successor task directory does not contain the exact frozen task set")
    return configs


def validate_contract_manifest(path: Path) -> dict[str, object]:
    data = load_task_config(path)
    expected = {
        "benchmark_version": SUCCESSOR_BENCHMARK_VERSION,
        "contract_status": "local_release_candidate_not_public",
        "clinical_review_status": "pending_not_simulated",
        "claim_boundary": CLAIM_BOUNDARY,
        "history_bins": 6,
        "feature_names": list(FEATURE_NAMES),
        "model_families": list(CORE_FAMILIES),
        "modes": list(MODES),
        "seeds": [3408, 3411, 3414],
        "normalization_rule": NORMALIZATION_RULE,
        "task_versions": TASK_VERSIONS,
        "task_row_hashes": TASK_ROW_HASHES,
    }
    for field, value in expected.items():
        if data.get(field) != value:
            raise ReleaseContractError(f"Successor contract manifest mismatch: {field}")
    role = _mapping(data.get("role_contract"), "role_contract")
    if role.get("salt") != "KDD-RV-SUBJECT-ROLE-v1|" or role.get("ranges") != {
        "train": [0, 7000],
        "validation": [7000, 8500],
        "sealed_test": [8500, 10000],
    }:
        raise ReleaseContractError("Successor contract manifest role drift")
    bootstrap = _mapping(data.get("bootstrap"), "bootstrap")
    if bootstrap != {
        "replicates": 1000,
        "seed": 8813408,
        "unit": "subject_cluster",
        "practical_epsilons": [0.0, 0.01, 0.02],
        "holm_alpha": 0.05,
    }:
        raise ReleaseContractError("Successor contract manifest bootstrap drift")
    recursive = data.get("recursive_contract")
    if recursive != (
        "start from first logged pre-action history; replace subsequent state values with prior predictions; "
        "retain logged observation masks, recencies, and actions; never use later observed values; "
        "reset only after a sequence or relative-index gap"
    ):
        raise ReleaseContractError("Successor contract manifest recursion drift")
    backend = _mapping(data.get("backend_source"), "backend_source")
    source_manifest = Path(__file__).resolve().parent / "contracts/source_manifest.csv"
    source_digest = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    if backend.get("commit") != "64d76a34b7e14db364f6025889e82289d8bdd8c2" or backend.get("manifest_sha256") != source_digest:
        raise ReleaseContractError("Successor backend source receipt drift")
    receipts = _mapping(data.get("researchforge_receipts"), "researchforge_receipts")
    if receipts != {
        "rv00_task_table_sha256": "317a442aa8d50c955895ff8079765f0c45df38d820e4c5a6cbaabe8917c7da0a",
        "rv01_config_sha256": "3d47505fed8ff11467846f15fc8b114a58ec39bfa6626b4b84613a9358ad0803",
        "rv02_config_sha256": "e662afc96ff6003f4e9205af681f44868cdff58ec275ed51e701831f2f8a372c",
    }:
        raise ReleaseContractError("Successor ResearchForge receipt drift")
    validate_task_table(path.parent / "task_contract_candidates.csv")
    return data


def validate_task_table(path: Path) -> list[dict[str, str]]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != "317a442aa8d50c955895ff8079765f0c45df38d820e4c5a6cbaabe8917c7da0a":
        raise ReleaseContractError("Frozen successor task table hash mismatch")
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise ReleaseContractError(f"Cannot read frozen successor task table: {error}") from error
    if {row.get("task_id") for row in rows} != set(TASKS):
        raise ReleaseContractError("Frozen successor task table identity mismatch")
    for row in rows:
        task = row["task_id"]
        canonical = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        if hashlib.sha256(canonical).hexdigest() != TASK_ROW_HASHES[task]:
            raise ReleaseContractError(f"Frozen successor task row hash mismatch: {task}")
        if row.get("version") != TASK_VERSIONS[task]:
            raise ReleaseContractError(f"Frozen successor task version mismatch: {task}")
    return rows


def _mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ReleaseContractError(f"{field} must be an object")
    return cast(dict[str, object], value)
