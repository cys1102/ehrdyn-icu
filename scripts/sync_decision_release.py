#!/usr/bin/env python3
"""Sync the allowlisted aggregate manuscript package into the decision release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

ALLOWLIST = {
    "manuscript.tex": "decision/manuscript/manuscript.tex",
    "manuscript.pdf": "decision/manuscript/manuscript.pdf",
    "references.bib": "decision/manuscript/references.bib",
    "benchmark-card.md": "decision/manuscript/benchmark-card.md",
    "contracts.md": "decision/manuscript/contracts.md",
    "estimator-planner-cards.md": "decision/manuscript/estimator-planner-cards.md",
    "reproducibility.md": "decision/manuscript/reproducibility.md",
    "number-audit.csv": "decision/evidence/number-audit.csv",
    "provenance-manifest.csv": "decision/evidence/source-provenance.csv",
    "figures/known_value_policy_level.pdf": "decision/figures/known_value_policy_level.pdf",
    "figures/known_value_policy_level.png": "decision/figures/known_value_policy_level.png",
    "figures/cohort_evaluability_roles.pdf": "decision/figures/cohort_evaluability_roles.pdf",
    "figures/cohort_evaluability_roles.png": "decision/figures/cohort_evaluability_roles.png",
    "tables/cohort_scale.csv": "decision/evidence/cohort_scale.csv",
    "tables/cross_cohort_evaluation_layers.csv": "decision/evidence/cross_cohort_evaluation_layers.csv",
    "tables/factual_action_information_primary.csv": "decision/evidence/factual_action_information_primary.csv",
    "tables/world_model_fidelity_summary.csv": "decision/evidence/world_model_fidelity_summary.csv",
    "tables/world_model_uncertainty_summary.csv": "decision/evidence/world_model_uncertainty_summary.csv",
    "tables/world_model_recursive_horizon_metrics.csv": "decision/evidence/world_model_recursive_horizon_metrics.csv",
    "tables/world_model_planner_bridge.csv": "decision/evidence/world_model_planner_bridge.csv",
    "tables/known_value_cross_task_summary.csv": "decision/evidence/known_value_cross_task_summary.csv",
    "tables/known_value_policy_level.csv": "decision/evidence/known_value_policy_level.csv",
    "tables/known_value_reference_full_matrix.csv": "decision/evidence/known_value_reference_full_matrix.csv",
    "tables/known_value_task_extension_full_matrix.csv": "decision/evidence/known_value_task_extension_full_matrix.csv",
    "tables/task_extension_world_model_planner_matrix.csv": "decision/evidence/task_extension_world_model_planner_matrix.csv",
    "tables/task_extension_rank_stability.csv": "decision/evidence/task_extension_rank_stability.csv",
    "tables/ope_gate_thresholds.csv": "decision/evidence/ope_gate_thresholds.csv",
    "tables/ope_gate_breakdown.csv": "decision/evidence/ope_gate_breakdown.csv",
    "tables/ope_approved_exact_tuples.csv": "decision/evidence/ope_approved_exact_tuples.csv",
    "tables/ope_reference_all_tuple_metrics.csv": "decision/evidence/ope_reference_all_tuple_metrics.csv",
    "tables/ope_task_matched_all_tuple_metrics.csv": "decision/evidence/ope_task_matched_all_tuple_metrics.csv",
    "tables/task_matched_ope_authorization.csv": "decision/evidence/task_matched_ope_authorization.csv",
    "tables/retrospective_policy_diagnostics.csv": "decision/evidence/retrospective_policy_diagnostics.csv",
}

SOURCE_CODE_ALLOWLIST = (
    "kdd_benchmark_discovery/kdd069_model_types.py",
    "kdd_benchmark_discovery/kdd069_sequence_models.py",
    "kdd_benchmark_discovery/kdd069_rssm_models.py",
    "kdd_benchmark_discovery/kdd_e01_evaluator.py",
    "kdd_benchmark_discovery/run_kdd_e02_known_value_full.py",
    "kdd_benchmark_discovery/run_kdd100_complete_known_value.py",
    "kdd_benchmark_discovery/run_kdd100r_task_matched_known_value.py",
    "kdd_benchmark_discovery/run_kdd_x02_cross_cohort_policy_benchmark.py",
    "kdd_benchmark_discovery/run_kdd_x08_task_matched_evaluator.py",
    "kdd_benchmark_discovery/run_kdd_x09_promoted_cohort_policy_benchmark.py",
    "configs/kdd100r_task_matched_known_value.json",
    "configs/kdd101_model_free_diagnostics_v5.json",
    "configs/kdd_x02_cross_cohort_policy_benchmark_v1.json",
    "configs/kdd_x08_task_matched_evaluator_v1.json",
    "configs/kdd_x09_promoted_cohort_policy_benchmark_v1.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(path: Path) -> int | str:
    if path.suffix != ".csv":
        return "not_applicable"
    with path.open(newline="", encoding="utf-8") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--world-ehr-source", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    world_ehr_source = args.world_ehr_source.resolve()
    if not (source / "manuscript.tex").is_file():
        raise FileNotFoundError("--source must be the reader-facing manuscript package")

    manifest_rows = []
    for source_name, destination_name in ALLOWLIST.items():
        source_path = source / source_name
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        destination = ROOT / destination_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination)
        manifest_rows.append(
            {
                "path": destination.relative_to(ROOT).as_posix(),
                "sha256": sha256(destination),
                "bytes": destination.stat().st_size,
                "data_rows": csv_rows(destination),
                "aggregate_or_document_only": True,
            }
        )

    manifest = ROOT / "decision/evidence/manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    source_rows = []
    reference_root = ROOT / "decision/reference_code"
    reference_root.mkdir(parents=True, exist_ok=True)
    expected_reference_files = {Path(relative).name for relative in SOURCE_CODE_ALLOWLIST}
    for stale in reference_root.iterdir():
        if stale.is_file() and stale.name != "source_manifest.csv" and stale.name not in expected_reference_files:
            stale.unlink()
    for relative in SOURCE_CODE_ALLOWLIST:
        source_path = world_ehr_source / relative
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        destination = reference_root / Path(relative).name
        shutil.copyfile(source_path, destination)
        source_rows.append(
            {
                "packaged_path": destination.relative_to(ROOT).as_posix(),
                "source_relative_path": relative,
                "sha256": sha256(destination),
                "bytes": destination.stat().st_size,
                "provenance": "byte_identical_source_snapshot",
            }
        )
    source_manifest = ROOT / "decision/reference_code/source_manifest.csv"
    with source_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)


if __name__ == "__main__":
    main()
