from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from ..errors import ReleaseContractError


EXPECTED_ROWS = {
    "decision/evidence/cross_cohort_evaluation_layers.csv": 6,
    "decision/evidence/known_value_cross_task_summary.csv": 4,
    "decision/evidence/known_value_reference_full_matrix.csv": 1632,
    "decision/evidence/known_value_task_extension_full_matrix.csv": 816,
    "decision/evidence/world_model_recursive_horizon_metrics.csv": 924,
    "decision/evidence/ope_reference_all_tuple_metrics.csv": 6912,
    "decision/evidence/ope_task_matched_all_tuple_metrics.csv": 9216,
}

REQUIRED_REFERENCE_SOURCES = {
    "kdd069_model_types.py",
    "kdd069_sequence_models.py",
    "kdd069_rssm_models.py",
    "kdd_e01_evaluator.py",
    "run_kdd_e02_known_value_full.py",
    "run_kdd_x02_cross_cohort_policy_benchmark.py",
    "run_kdd_x08_task_matched_evaluator.py",
    "run_kdd_x09_promoted_cohort_policy_benchmark.py",
}


def validate_decision_release(root: Path) -> dict[str, object]:
    root = root.resolve()
    manifest_path = root / "decision/evidence/manifest.csv"
    task_contract_path = root / "decision/contracts/task_contracts.json"
    gate_path = root / "decision/contracts/ope_gates.json"
    if not manifest_path.is_file():
        raise ReleaseContractError("Decision evidence manifest is missing")
    manifest = _csv_rows(manifest_path)
    indexed = {row["path"]: row for row in manifest}
    for relative, expected_rows in EXPECTED_ROWS.items():
        if relative not in indexed:
            raise ReleaseContractError(f"Required decision artifact is not manifested: {relative}")
        if int(indexed[relative]["data_rows"]) != expected_rows:
            raise ReleaseContractError(f"Decision artifact row-count mismatch: {relative}")

    for row in manifest:
        path = root / row["path"]
        if not path.is_file():
            raise ReleaseContractError(f"Manifest file is missing: {row['path']}")
        if _sha256(path) != row["sha256"]:
            raise ReleaseContractError(f"Decision artifact checksum mismatch: {row['path']}")
        if path.stat().st_size != int(row["bytes"]):
            raise ReleaseContractError(f"Decision artifact byte-count mismatch: {row['path']}")
        if row["aggregate_or_document_only"] != "True":
            raise ReleaseContractError(f"Non-aggregate artifact entered decision manifest: {row['path']}")

    source_manifest = _csv_rows(root / "decision/reference_code/source_manifest.csv")
    packaged_source_names = {Path(row["packaged_path"]).name for row in source_manifest}
    missing_sources = REQUIRED_REFERENCE_SOURCES - packaged_source_names
    if missing_sources:
        raise ReleaseContractError(
            "Decision reference-code manifest is missing required snapshots: "
            + ", ".join(sorted(missing_sources))
        )
    for row in source_manifest:
        path = root / row["packaged_path"]
        if not path.is_file() or _sha256(path) != row["sha256"]:
            raise ReleaseContractError(f"Decision reference-code hash mismatch: {row['packaged_path']}")
        if row["provenance"] != "byte_identical_source_snapshot":
            raise ReleaseContractError("Decision source provenance label drifted")

    task_contract = json.loads(task_contract_path.read_text(encoding="utf-8"))
    if len(task_contract["ehr_world_model_tasks"]) != 6:
        raise ReleaseContractError("Decision contract must contain six EHR world-model tasks")
    if len(task_contract["known_value_policy_tasks"]) != 4:
        raise ReleaseContractError("Decision contract must contain four known-value policy tasks")
    if task_contract["retrospective_ehr_policy_value"] != "not_executed":
        raise ReleaseContractError("Retrospective policy-value boundary drifted")

    gates = json.loads(gate_path.read_text(encoding="utf-8"))
    if gates["family_name_transport_allowed"] is not False:
        raise ReleaseContractError("Estimator-family transport must remain disabled")

    layers = _csv_rows(root / "decision/evidence/cross_cohort_evaluation_layers.csv")
    if any(not row["retrospective_policy_value"].startswith("not executed") for row in layers):
        raise ReleaseContractError("A retrospective policy value appeared in the release")

    reference = _csv_rows(root / "decision/evidence/ope_reference_all_tuple_metrics.csv")
    task_matched = _csv_rows(root / "decision/evidence/ope_task_matched_all_tuple_metrics.csv")
    reference_approved = sum(_truth(row["approved_exact_contract"]) for row in reference)
    reference_tier2_approved = sum(
        _truth(row["approved_exact_contract"])
        for row in reference
        if row["tier"].startswith("tier2")
    )
    aki_tier2_approved = sum(
        _truth(row["approved_exact_tuple"])
        for row in task_matched
        if row["task"] == "aki_rrt" and row["tier"].startswith("tier2")
    )
    hf_approved = sum(
        _truth(row["approved_exact_tuple"])
        for row in task_matched
        if row["task"] == "heart_failure"
    )
    if (reference_approved, reference_tier2_approved, aki_tier2_approved, hf_approved) != (32, 0, 236, 0):
        raise ReleaseContractError("OPE authorization counts do not match the paper contract")

    return {
        "benchmark_id": task_contract["benchmark_id"],
        "manifest_files_verified": len(manifest),
        "reference_source_files_verified": len(source_manifest),
        "ehr_world_model_tasks": 6,
        "known_value_policy_tasks": 4,
        "known_value_policy_rows": 2448,
        "ope_tuple_rows": 16128,
        "reference_exact_approved": reference_approved,
        "aki_task_matched_tier2_approved": aki_tier2_approved,
        "retrospective_ehr_policy_value": "not_executed",
        "pass": True,
    }


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _truth(value: str) -> bool:
    return value.strip().lower() == "true"
