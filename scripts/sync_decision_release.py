#!/usr/bin/env python3
"""Sync the allowlisted aggregate manuscript package into the decision release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DOCUMENT_ALLOWLIST = {
    "manuscript.tex": "decision/manuscript/manuscript.tex",
    "manuscript.pdf": "decision/manuscript/manuscript.pdf",
    "references.bib": "decision/manuscript/references.bib",
    "README.md": "decision/manuscript/reader-package.md",
    "benchmark-card.md": "decision/manuscript/benchmark-card.md",
    "contracts.md": "decision/manuscript/contracts.md",
    "credentialed-reconstruction.md": "decision/manuscript/credentialed-reconstruction.md",
    "environment-manifest.md": "decision/manuscript/environment-manifest.md",
    "estimator-planner-cards.md": "decision/manuscript/estimator-planner-cards.md",
    "non-claims-and-limitations.md": "decision/manuscript/non-claims-and-limitations.md",
    "reproducibility.md": "decision/manuscript/reproducibility.md",
    "artifact-hashes.json": "decision/manuscript/reader-package-hashes.json",
    "number-audit.csv": "decision/evidence/number-audit.csv",
    "provenance-manifest.csv": "decision/evidence/source-provenance.csv",
}

FIGURE_ALLOWLIST = {
    "adaptive_all_policy_regret_heatmap.pdf",
    "adaptive_all_policy_regret_heatmap.png",
    "cohort_evaluability_roles.pdf",
    "cohort_evaluability_roles.png",
    "heterogeneous_all_policy_fixed_control_heatmap.pdf",
    "heterogeneous_all_policy_fixed_control_heatmap.png",
    "known_value_all_policy_sensitivity_heatmap.pdf",
    "known_value_all_policy_sensitivity_heatmap.png",
    "known_value_method_groups.pdf",
    "known_value_method_groups.png",
    "known_value_policy_level.pdf",
    "known_value_policy_level.png",
}

TABLE_ALLOWLIST = {
    "action_information_policy_discrimination.csv",
    "adaptive_environment_contract.csv",
    "adaptive_exploitation_gap_all_rows.csv",
    "adaptive_exploitation_summary.csv",
    "adaptive_policy_group_leaders.csv",
    "adaptive_policy_performance_all_methods.csv",
    "adaptive_policy_regret_all_rows.csv",
    "adaptive_policy_true_returns_all_rows.csv",
    "adaptive_vs_best_fixed_gap.csv",
    "all_baseline_transition_leaders.csv",
    "baseline_method_inventory.csv",
    "baseline_surface_atlas.csv",
    "cohort_policy_evaluability.csv",
    "cohort_scale.csv",
    "cohort_uncertainty_leaders.csv",
    "complete_model_performance_by_cohort.csv",
    "complete_model_performance_task_balanced.csv",
    "cross_surface_model_family_rows.csv",
    "cross_surface_rank_stability.csv",
    "cross_cohort_evaluation_layers.csv",
    "ehr_known_value_bridge_coefficients.csv",
    "ehr_to_known_value_contract_matrix.csv",
    "factual_action_information_primary.csv",
    "heterogeneous_all_method_seed_means.csv",
    "heterogeneous_cell_discrimination.csv",
    "heterogeneous_model_exploitation_all_rows.csv",
    "heterogeneous_policy_true_returns_all_rows.csv",
    "heterogeneous_world_model_planner_all_rows.csv",
    "known_value_all_policy_sensitivity.csv",
    "known_value_balanced_core.csv",
    "known_value_cross_task_summary.csv",
    "known_value_max_supported_control.csv",
    "known_value_model_based_vs_model_free.csv",
    "known_value_policy_level.csv",
    "known_value_reference_full_matrix.csv",
    "known_value_scenario_inventory.csv",
    "known_value_task_extension_full_matrix.csv",
    "ope_approved_exact_tuples.csv",
    "ope_gate_breakdown.csv",
    "ope_gate_thresholds.csv",
    "ope_reference_all_tuple_metrics.csv",
    "ope_task_matched_all_tuple_metrics.csv",
    "ope_validation_and_disposition.csv",
    "policy_set_interval_inclusion_diagnostic_only.csv",
    "prediction_policy_consistency.csv",
    "promoted_task_roles.csv",
    "related_work_landscape.csv",
    "retrospective_policy_diagnostics.csv",
    "repeated_dataset_ope_authorization.csv",
    "repeated_dataset_ope_coverage.csv",
    "repeated_dataset_ope_gate_summary.csv",
    "repeated_dataset_ope_rank_and_sign.csv",
    "sepsis_reference_fidelity.csv",
    "task_extension_rank_stability.csv",
    "task_extension_world_model_planner_matrix.csv",
    "task_matched_ope_authorization.csv",
    "uncertainty_planning_consistency.csv",
    "world_model_fidelity_summary.csv",
    "world_model_planner_bridge.csv",
    "world_model_recursive_horizon_metrics.csv",
    "world_model_uncertainty_summary.csv",
}

ALLOWLIST = {
    **DOCUMENT_ALLOWLIST,
    **{
        f"figures/{name}": f"decision/figures/{name}"
        for name in sorted(FIGURE_ALLOWLIST)
    },
    **{
        f"tables/{name}": f"decision/evidence/{name}"
        for name in sorted(TABLE_ALLOWLIST)
    },
}

SOURCE_CODE_ALLOWLIST = (
    "kdd_benchmark_discovery/kdd069_model_types.py",
    "kdd_benchmark_discovery/kdd069_sequence_models.py",
    "kdd_benchmark_discovery/kdd069_rssm_models.py",
    "kdd_benchmark_discovery/kdd098_training.py",
    "kdd_benchmark_discovery/kdd098r_training.py",
    "kdd_benchmark_discovery/kdd_e01_evaluator.py",
    "kdd_benchmark_discovery/run_kdd_e02_known_value_full.py",
    "kdd_benchmark_discovery/run_kdd100_complete_known_value.py",
    "kdd_benchmark_discovery/run_kdd100r_task_matched_known_value.py",
    "kdd_benchmark_discovery/run_kdd_s02_canonical_sepsis_materialization.py",
    "kdd_benchmark_discovery/run_kdd098r_world_models.py",
    "kdd_benchmark_discovery/run_kdd101_model_free_diagnostics.py",
    "kdd_benchmark_discovery/run_kdd_adapt01_adaptive_known_value.py",
    "kdd_benchmark_discovery/run_kdd107_heterogeneous_known_value.py",
    "kdd_benchmark_discovery/run_kdd_ope_rd01_repeated_dataset.py",
    "kdd_benchmark_discovery/run_kdd_bridge01_ehr_known_value.py",
    "kdd_benchmark_discovery/run_kdd_x02_cross_cohort_policy_benchmark.py",
    "kdd_benchmark_discovery/run_kdd_x08_task_matched_evaluator.py",
    "kdd_benchmark_discovery/run_kdd_x09_promoted_cohort_policy_benchmark.py",
    "configs/kdd100r_task_matched_known_value.json",
    "configs/kdd098r_world_model.json",
    "configs/kdd101_model_free_diagnostics_v5.json",
    "configs/kdd_adapt01_adaptive_known_value_v1.json",
    "configs/kdd107_heterogeneous_known_value_v1.json",
    "configs/kdd_ope_rd01_repeated_dataset_v1.json",
    "configs/kdd_bridge01_ehr_known_value_v1.json",
    "configs/kdd_x02_cross_cohort_policy_benchmark_v1.json",
    "configs/kdd_x08_task_matched_evaluator_v1.json",
    "configs/kdd_x09_promoted_cohort_policy_benchmark_v1.json",
    "tests/test_kdd107_heterogeneous_known_value.py",
    "tests/test_kdd_ope_rd01.py",
    "tests/test_kdd_bridge01.py",
)

DOCUMENT_TEXT_REPLACEMENTS = {
    "environment-manifest.md": {
        "/" + "home" + "/cys1102/miniconda3/envs/world-model/bin/python":
            "<world-model-python>",
    },
}

SOURCE_TEXT_REPLACEMENTS = {
    "kdd_benchmark_discovery/run_kdd_s02_canonical_sepsis_materialization.py": {
        'Path("/' + 'data/physionet.org/files/mimiciv/3.1")':
            'Path("<authorized-mimiciv-root>")',
    },
    "kdd_benchmark_discovery/run_kdd098r_world_models.py": {
        'Path("/' + 'data/physionet.org/files/mimiciv/3.1")':
            'Path("<authorized-mimiciv-root>")',
        '"/' + 'home/" in text': '"/" + "home" + "/" in text',
    },
    "kdd_benchmark_discovery/run_kdd101_model_free_diagnostics.py": {
        'Path("/' + 'data/physionet.org/files/mimiciv/3.1")':
            'Path("<authorized-mimiciv-root>")',
    },
    "kdd_benchmark_discovery/run_kdd107_heterogeneous_known_value.py": {
        'RESEARCHFORGE = Path("/' + 'home/cys1102/yunsung/researchforge")':
            'RESEARCHFORGE = Path("<researchforge-root>")',
    },
}


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


def copy_with_replacements(
    source: Path,
    destination: Path,
    replacements: dict[str, str] | None = None,
) -> bool:
    if not replacements:
        shutil.copyfile(source, destination)
        return False
    text = source.read_text(encoding="utf-8")
    for old, new in replacements.items():
        if old not in text:
            raise RuntimeError(f"Expected sanitization target is missing: {source}: {old}")
        text = text.replace(old, new)
    destination.write_text(text, encoding="utf-8")
    return True


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
        copy_with_replacements(
            source_path,
            destination,
            DOCUMENT_TEXT_REPLACEMENTS.get(source_name),
        )
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
        writer = csv.DictWriter(
            handle, fieldnames=list(manifest_rows[0]), lineterminator="\n"
        )
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
        sanitized = copy_with_replacements(
            source_path,
            destination,
            SOURCE_TEXT_REPLACEMENTS.get(relative),
        )
        source_rows.append(
            {
                "packaged_path": destination.relative_to(ROOT).as_posix(),
                "source_relative_path": relative,
                "source_sha256": sha256(source_path),
                "sha256": sha256(destination),
                "bytes": destination.stat().st_size,
                "provenance": (
                    "portable_path_sanitized_source_snapshot"
                    if sanitized
                    else "byte_identical_source_snapshot"
                ),
            }
        )
    source_manifest = ROOT / "decision/reference_code/source_manifest.csv"
    with source_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(source_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(source_rows)


if __name__ == "__main__":
    main()
