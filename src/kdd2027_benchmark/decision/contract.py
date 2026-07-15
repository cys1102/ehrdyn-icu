from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from ..errors import ReleaseContractError


EXPECTED_ROWS = {
    "decision/evidence/related_work_landscape.csv": 8,
    "decision/evidence/cohort_scale.csv": 6,
    "decision/evidence/primary_cohort_scale_eligibility.csv": 6,
    "decision/evidence/complete_model_performance_by_cohort.csv": 36,
    "decision/evidence/complete_model_performance_task_balanced.csv": 6,
    "decision/evidence/current_scale_qualified_model_performance_by_cohort.csv": 18,
    "decision/evidence/current_scale_qualified_model_performance_task_balanced.csv": 6,
    "decision/evidence/cross_cohort_evaluation_layers.csv": 6,
    "decision/evidence/adaptive_environment_contract.csv": 4,
    "decision/evidence/adaptive_policy_performance_all_methods.csv": 136,
    "decision/evidence/adaptive_policy_true_returns_all_rows.csv": 700,
    "decision/evidence/adaptive_policy_regret_all_rows.csv": 700,
    "decision/evidence/adaptive_exploitation_gap_all_rows.csv": 480,
    "decision/evidence/heterogeneous_policy_true_returns_all_rows.csv": 4080,
    "decision/evidence/heterogeneous_world_model_planner_all_rows.csv": 2880,
    "decision/evidence/heterogeneous_model_exploitation_all_rows.csv": 2880,
    "decision/evidence/heterogeneous_all_method_seed_means.csv": 544,
    "decision/evidence/heterogeneous_cell_discrimination.csv": 16,
    "decision/evidence/current_scale_qualified_policy_true_returns_all_rows.csv": 3060,
    "decision/evidence/current_scale_qualified_world_model_planner_all_rows.csv": 2160,
    "decision/evidence/current_scale_qualified_model_exploitation_all_rows.csv": 2160,
    "decision/evidence/current_scale_qualified_cell_discrimination.csv": 12,
    "decision/evidence/known_value_cross_task_summary.csv": 4,
    "decision/evidence/known_value_reference_full_matrix.csv": 1632,
    "decision/evidence/known_value_task_extension_full_matrix.csv": 816,
    "decision/evidence/world_model_recursive_horizon_metrics.csv": 924,
    "decision/evidence/ope_reference_all_tuple_metrics.csv": 6912,
    "decision/evidence/ope_task_matched_all_tuple_metrics.csv": 9216,
    "decision/evidence/repeated_dataset_ope_coverage.csv": 10368,
    "decision/evidence/repeated_dataset_ope_rank_and_sign.csv": 1728,
    "decision/evidence/repeated_dataset_ope_authorization.csv": 1728,
    "decision/evidence/repeated_dataset_ope_gate_summary.csv": 2,
    "decision/evidence/current_scale_qualified_repeated_dataset_ope_coverage.csv": 7776,
    "decision/evidence/current_scale_qualified_repeated_dataset_ope_rank_and_sign.csv": 1296,
    "decision/evidence/current_scale_qualified_repeated_dataset_ope_authorization.csv": 1296,
    "decision/evidence/current_scale_qualified_repeated_dataset_ope_gate_summary.csv": 2,
    "decision/evidence/policy_set_interval_inclusion_diagnostic_only.csv": 6912,
    "decision/evidence/ehr_to_known_value_contract_matrix.csv": 80,
    "decision/evidence/cross_surface_model_family_rows.csv": 16,
    "decision/evidence/ehr_known_value_bridge_coefficients.csv": 8,
}

REQUIRED_REFERENCE_SOURCES = {
    "kdd069_model_types.py",
    "kdd069_sequence_models.py",
    "kdd069_rssm_models.py",
    "kdd_e01_evaluator.py",
    "run_kdd_e02_known_value_full.py",
    "run_kdd_s02_canonical_sepsis_materialization.py",
    "run_kdd098r_world_models.py",
    "run_kdd101_model_free_diagnostics.py",
    "run_kdd_adapt01_adaptive_known_value.py",
    "run_kdd107_heterogeneous_known_value.py",
    "run_kdd_ope_rd01_repeated_dataset.py",
    "run_kdd_bridge01_ehr_known_value.py",
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
        provenance = row["provenance"]
        if provenance not in {
            "byte_identical_source_snapshot",
            "portable_path_sanitized_source_snapshot",
        }:
            raise ReleaseContractError("Decision source provenance label drifted")
        source_digest = row.get("source_sha256", "")
        if len(source_digest) != 64:
            raise ReleaseContractError("Decision source digest is missing")
        if provenance == "byte_identical_source_snapshot" and source_digest != row["sha256"]:
            raise ReleaseContractError("Byte-identical source digest does not match packaged digest")
        if provenance == "portable_path_sanitized_source_snapshot" and source_digest == row["sha256"]:
            raise ReleaseContractError("Sanitized source snapshot did not change")

    task_contract = json.loads(task_contract_path.read_text(encoding="utf-8"))
    if len(task_contract["ehr_world_model_tasks"]) != 6:
        raise ReleaseContractError("Decision contract must contain six EHR P/R/T-component tasks")
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

    complete_models = _csv_rows(root / "decision/evidence/complete_model_performance_by_cohort.csv")
    if len({row["cohort"] for row in complete_models}) != 6:
        raise ReleaseContractError("Complete transition inventory must cover six cohorts")
    if any(sum(candidate["cohort"] == row["cohort"] for candidate in complete_models) != 6 for row in complete_models):
        raise ReleaseContractError("Each cohort must expose all six transition methods")

    primary_scale = _csv_rows(root / "decision/evidence/primary_cohort_scale_eligibility.csv")
    expected_primary_cohorts = {
        "sepsis",
        "respiratory",
        "shock",
        "af flutter",
        "aki",
        "heart failure",
    }
    if {row["cohort"] for row in primary_scale} != expected_primary_cohorts:
        raise ReleaseContractError("Final cohort-scale inventory drifted")
    if any(
        int(row["primary_subjects"]) < 10_000
        or int(row["primary_episodes"]) < 10_000
        or row["scale_gate"] != "pass_ge10000_subjects_and_episodes"
        for row in primary_scale
    ):
        raise ReleaseContractError("A final cohort target violates the frozen 10k scale gate")
    current_primary_cohorts = {
        row["cohort"]
        for row in primary_scale
        if _truth(row["current_primary_result_available"])
    }
    if current_primary_cohorts != {"respiratory", "shock", "aki"}:
        raise ReleaseContractError("Current large-lineage result availability drifted")

    current_models = _csv_rows(
        root / "decision/evidence/current_scale_qualified_model_performance_by_cohort.csv"
    )
    if {row["cohort"] for row in current_models} != current_primary_cohorts:
        raise ReleaseContractError("Current transition matrix includes a pending lineage")
    if any(
        sum(candidate["cohort"] == row["cohort"] for candidate in current_models) != 6
        for row in current_models
    ):
        raise ReleaseContractError("Each current cohort must expose all six transition methods")
    if any(row["evidence_status"] != "current_scale_qualified_result" for row in current_models):
        raise ReleaseContractError("Current transition evidence status drifted")

    adaptive_summary = _csv_rows(root / "decision/evidence/adaptive_policy_performance_all_methods.csv")
    if len({row["task"] for row in adaptive_summary}) != 4:
        raise ReleaseContractError("Adaptive policy summary must cover four tasks")
    if any(sum(candidate["task"] == row["task"] for candidate in adaptive_summary) != 34 for row in adaptive_summary):
        raise ReleaseContractError("Each adaptive task must expose all 34 non-oracle labels")
    adaptive_regret = _csv_rows(root / "decision/evidence/adaptive_policy_regret_all_rows.csv")
    if any(_truth(row["negative_regret"]) for row in adaptive_regret):
        raise ReleaseContractError("Adaptive exact-oracle evaluation contains negative regret")

    heterogeneous = _csv_rows(root / "decision/evidence/heterogeneous_policy_true_returns_all_rows.csv")
    heterogeneous_planners = _csv_rows(root / "decision/evidence/heterogeneous_world_model_planner_all_rows.csv")
    heterogeneous_cells = _csv_rows(root / "decision/evidence/heterogeneous_cell_discrimination.csv")
    if len({(row["task"], row["mechanism"]) for row in heterogeneous}) != 24:
        raise ReleaseContractError("Heterogeneous policy inventory must cover 24 task-mechanism environments")
    if any(float(row["exact_regret"]) < -1e-12 for row in heterogeneous):
        raise ReleaseContractError("Heterogeneous exact-oracle evaluation contains material negative regret")
    if any(not _truth(row["exact_mc_agreement_pass"]) or not _truth(row["support_pass"]) for row in heterogeneous):
        raise ReleaseContractError("Heterogeneous policy evaluator or support gate failed")
    if len(heterogeneous_planners) != 2880:
        raise ReleaseContractError("Heterogeneous world-model planner inventory is incomplete")
    discrimination_passes = sum(_truth(row["learned_beats_both_fixed_extremes"]) for row in heterogeneous_cells)
    if discrimination_passes != 11:
        raise ReleaseContractError("Heterogeneous fixed-control discrimination count drifted")

    current_policy = _csv_rows(
        root / "decision/evidence/current_scale_qualified_policy_true_returns_all_rows.csv"
    )
    current_planners = _csv_rows(
        root / "decision/evidence/current_scale_qualified_world_model_planner_all_rows.csv"
    )
    current_cells = _csv_rows(
        root / "decision/evidence/current_scale_qualified_cell_discrimination.csv"
    )
    current_policy_tasks = {"respiratory", "shock", "aki_rrt"}
    if {row["task"] for row in current_policy} != current_policy_tasks:
        raise ReleaseContractError("Current policy ledger includes a pending lineage")
    if {row["task"] for row in current_planners} != current_policy_tasks:
        raise ReleaseContractError("Current planner ledger includes a pending lineage")
    if any(float(row["exact_regret"]) < -1e-12 for row in current_policy):
        raise ReleaseContractError("Current exact-oracle evaluation contains material negative regret")
    if any(not _truth(row["exact_mc_agreement_pass"]) or not _truth(row["support_pass"]) for row in current_policy):
        raise ReleaseContractError("Current policy evaluator or support gate failed")
    current_discrimination_passes = sum(
        _truth(row["learned_beats_both_fixed_extremes"]) for row in current_cells
    )
    if current_discrimination_passes != 8:
        raise ReleaseContractError("Current fixed-control discrimination count drifted")

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

    repeated = _csv_rows(root / "decision/evidence/repeated_dataset_ope_authorization.csv")
    repeated_adaptive_approved = sum(
        _truth(row["approved_known_value_tuple"])
        for row in repeated
        if row["response_regime"] == "adaptive_composite"
    )
    repeated_null_approved = sum(
        _truth(row["approved_known_value_tuple"])
        for row in repeated
        if row["response_regime"] == "null_response"
    )
    if (repeated_adaptive_approved, repeated_null_approved) != (0, 40):
        raise ReleaseContractError("Repeated-dataset OPE authorization counts drifted")
    if any(_truth(row["retrospective_ehr_ope_authorized"]) for row in repeated):
        raise ReleaseContractError("Repeated-dataset calibration authorized retrospective EHR OPE")

    current_repeated = _csv_rows(
        root
        / "decision/evidence/current_scale_qualified_repeated_dataset_ope_authorization.csv"
    )
    if {row["task"] for row in current_repeated} != current_policy_tasks:
        raise ReleaseContractError("Current repeated-dataset OPE includes a pending lineage")
    current_repeated_adaptive_approved = sum(
        _truth(row["approved_known_value_tuple"])
        for row in current_repeated
        if row["response_regime"] == "adaptive_composite"
    )
    current_repeated_null_approved = sum(
        _truth(row["approved_known_value_tuple"])
        for row in current_repeated
        if row["response_regime"] == "null_response"
    )
    if (current_repeated_adaptive_approved, current_repeated_null_approved) != (0, 40):
        raise ReleaseContractError("Current repeated-dataset OPE authorization counts drifted")
    if any(_truth(row["retrospective_ehr_ope_authorized"]) for row in current_repeated):
        raise ReleaseContractError("Current repeated-dataset calibration authorized retrospective EHR OPE")

    return {
        "benchmark_id": task_contract["benchmark_id"],
        "manifest_files_verified": len(manifest),
        "reference_source_files_verified": len(source_manifest),
        "ehr_world_model_tasks": 6,
        "known_value_policy_tasks": 4,
        "complete_transition_rows": len(complete_models),
        "current_transition_rows": len(current_models),
        "scale_qualified_cohorts": len(primary_scale),
        "current_numeric_cohorts": len(current_primary_cohorts),
        "adaptive_policy_summary_rows": len(adaptive_summary),
        "adaptive_policy_seed_rows": len(adaptive_regret),
        "heterogeneous_policy_seed_rows": len(heterogeneous),
        "heterogeneous_planner_rows": len(heterogeneous_planners),
        "heterogeneous_discriminative_cells": discrimination_passes,
        "current_policy_seed_rows": len(current_policy),
        "current_planner_rows": len(current_planners),
        "current_discriminative_cells": current_discrimination_passes,
        "historical_known_value_policy_rows": 2448,
        "historical_ope_tuple_rows": 16128,
        "repeated_dataset_ope_tuples": len(repeated),
        "current_repeated_dataset_ope_tuples": len(current_repeated),
        "repeated_dataset_adaptive_approved": repeated_adaptive_approved,
        "repeated_dataset_null_approved": repeated_null_approved,
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
