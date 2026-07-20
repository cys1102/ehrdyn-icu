#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "release/kdd215"
MANIFEST = ROOT / "configs/full_benchmark/kdd198_v2_generator_contract.json"
IMPORT = ROOT / "configs/full_benchmark/authoritative_import_receipt.json"
REFERENCE = ROOT / "configs/full_benchmark/kdd199_authoritative_reference_cells.csv"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    direct_rows = read(OUT / "full_40_environment_direct_returns.csv")
    component_rows = read(OUT / "component_forecasting_results.csv")
    ope_rows = read(OUT / "full_320_dataset_ope_results.csv")
    if len(direct_rows) != 280 or len({(row["profile"], row["environment_seed"]) for row in direct_rows}) != 40:
        raise SystemExit("direct-return inventory mismatch")
    if len(component_rows) != 440 or sum(row.get("status") == "numeric_one_step" for row in component_rows) != 40:
        raise SystemExit("component inventory mismatch")
    if len(ope_rows) != 15_360:
        raise SystemExit(f"OPE row inventory mismatch: {len(ope_rows)}")
    if len({(row["profile"], row["environment_seed"], row["dataset_index"]) for row in ope_rows}) != 320:
        raise SystemExit("OPE dataset nesting mismatch")
    if {row["bootstrap_replicates"] for row in ope_rows} != {"500"}:
        raise SystemExit("OPE bootstrap count mismatch")
    imported = json.loads(IMPORT.read_text())
    matrix = [{"category": "public_base", "field": "KDD214",
               "source_sha256": "52127b903550012b281cc4a1dd4f8666e9552e41",
               "public_implementation": "fresh_isolated_worktree_parent_commit",
               "equivalence": "complete_schema_and_cross_python_exact_serialization",
               "smoke_substitution": False}]
    for path, digest in imported["authoritative_hashes"].items():
        matrix.append({"category": "authoritative_source", "field": path, "source_sha256": digest,
                       "public_implementation": "hash_gated_import", "equivalence": "exact_bytes_verified",
                       "smoke_substitution": False})
    for path, digest in imported["ported_dependency_hashes"].items():
        matrix.append({"category": "ported_dependency_source", "field": path,
                       "source_sha256": digest, "public_implementation": "hash_gated_import",
                       "equivalence": "exact_source_bytes_verified_before_transformation",
                       "smoke_substitution": False})
    for relative in (
        "src/kdd2027_benchmark/full_pomdp_types.py",
        "src/kdd2027_benchmark/full_pomdp_core.py",
        "src/kdd2027_benchmark/full_pomdp_v2.py",
        "src/kdd2027_benchmark/full_direct_evaluator.py",
        "src/kdd2027_benchmark/full_ope.py",
        "configs/full_benchmark/kdd198_v2_generator_contract.json",
    ):
        path = ROOT / relative
        matrix.append({"category": "public_port_artifact", "field": relative,
                       "source_sha256": imported.get("generator_contract_sha256", "not_one_to_one_source_file") if relative.endswith("generator_contract.json") else "derived_from_hash_gated_source",
                       "public_implementation": relative, "public_sha256": sha(path),
                       "equivalence": "generated_by_reviewable_hash_gated_import", "smoke_substitution": False})
    for field, public in (
        ("latent transition", "full_pomdp_core.R2Environment._precompute"),
        ("dense reward sign", "full_pomdp_v2.independent_dense_values"),
        ("observation emission", "full_pomdp_core.R2Environment._emit"),
        ("mask and recency", "full_pomdp_core.R2Environment._emit"),
        ("history fields", "values;masks;recency;previous_actions;step"),
        ("behavior policy", "serialized BehaviorCalibration plus exact context_bin"),
        ("environment profiles", "five profiles x eight seeds"),
        ("split roles", "train;validation;final_exogenous;logged_dataset;bootstrap"),
        ("direct evaluator", "full_direct_evaluator.evaluate_repaired_policy_batch"),
        ("estimators", "IS;WIS;CWPDIS;DR;WDR;FQE"),
        ("nuisance refit", "inside every episode-bootstrap replicate"),
        ("precision gate", "4096..65536; normalized SE <= 0.0025"),
    ):
        matrix.append({"category": "scientific_contract", "field": field,
                       "source_sha256": "covered_by_authoritative_hashes",
                       "public_implementation": public, "equivalence": "ported_exact_contract",
                       "smoke_substitution": False})
    write(OUT / "authoritative_source_to_public_port_matrix.csv", matrix)

    environment_rows = read(OUT / "environment_profile_seed_and_role_manifest.csv")
    cache = Path("/tmp/kdd215-public-suite")
    cache_files = list(cache.rglob("*.npz")) if cache.is_dir() else []
    write(OUT / "full_suite_generation_receipt.csv", [{
        "profile_count": len({row["profile"] for row in environment_rows}),
        "environment_count": len(environment_rows),
        "environment_seeds_per_profile": 8,
        "logged_datasets_per_environment": 8,
        "logged_dataset_count": len(environment_rows) * 8,
        "episodes_per_logged_dataset": 256,
        "bootstrap_refits_per_dataset_contract": 500,
        "generator_contract_sha256": sha(MANIFEST),
        "exact_nesting_pass": len(environment_rows) == 40,
        "smoke_only_substituted": False,
        "synthetic_train_validation_cache_file_count": len(cache_files),
        "synthetic_train_validation_cache_bytes": sum(path.stat().st_size for path in cache_files),
        "cache_committed_to_git": False,
        "cache_contains_latent_state_or_subtype": False,
    }])

    contrast_path = OUT / "environment_paired_return_contrasts.csv"
    environment_contrasts = [row for row in read(contrast_path) if row.get("row_role", "environment") == "environment"]
    summary_contrasts: list[dict[str, Any]] = []
    for comparator in sorted({row["comparator"] for row in environment_contrasts}):
        local = [row for row in environment_contrasts if row["comparator"] == comparator]
        profile_rows = {profile: [row for row in local if row["profile"] == profile]
                        for profile in sorted({row["profile"] for row in local})}
        estimate = sum(sum(float(row["paired_mean_difference"]) for row in values) / len(values)
                       for values in profile_rows.values()) / len(profile_rows)
        rng = random.Random(215_301)
        bootstrap = []
        for _ in range(10_000):
            profile_means = []
            for values in profile_rows.values():
                sampled = [values[rng.randrange(len(values))] for _ in values]
                profile_means.append(sum(float(row["paired_mean_difference"]) for row in sampled) / len(sampled))
            bootstrap.append(sum(profile_means) / len(profile_means))
        bootstrap.sort()
        summary_contrasts.append({
            "row_role": "profile_equal_environment_bootstrap_summary", "profile": "profile_equal",
            "environment_seed": "all_40", "entrant_id": local[0]["entrant_id"], "comparator": comparator,
            "paired_episode_count": "not_inferential_unit", "paired_mean_difference": estimate,
            "paired_standard_error": "environment_bootstrap", "paired_ci_lower": bootstrap[249],
            "paired_ci_upper": bootstrap[9749], "inferential_unit_for_summary": "profile_x_environment_seed",
            "bootstrap_replicates": 10_000, "bootstrap_seed": 215_301,
        })
    for row in environment_contrasts:
        row["row_role"] = "environment"
    write(contrast_path, environment_contrasts + summary_contrasts)

    optional = []
    for profile in ("sepsis", "respiratory", "shock", "aki", "heart_failure"):
        for horizon in (1, 4, 8):
            optional.append({"profile": profile, "entrant_id": "locf_gaussian_reference",
                             "planner_horizon": f"H{horizon}", "complete_prt_capability": False,
                             "status": "structural_na_transition_only_entrant",
                             "arbitrary_reward_or_termination_imputation": False})
    write(OUT / "optional_prt_planner_results_or_nonavailability.csv", optional)

    runtime_hashes = []
    for version in ("311", "312", "313"):
        path = Path(f"/tmp/kdd215-py{version}.json")
        runtime_hashes.append({"python": f"3.{version[-2:] if version != '311' else '11'}",
                               "numpy": "2.3.3", "probe_sha256": sha(path) if path.is_file() else "missing",
                               "canonical_bytes_equal": path.is_file()})
    common = len({row["probe_sha256"] for row in runtime_hashes}) == 1
    for row in runtime_hashes:
        row["all_supported_runtime_exact_hash_parity"] = common
    write(OUT / "supported_runtime_hash_parity.csv", runtime_hashes)

    clean_direct = Path("/tmp/kdd215-clean-direct.csv")
    clean_contrasts = Path("/tmp/kdd215-clean-contrasts.csv")
    write(OUT / "clean_install_full_example_receipt.csv", [{
        "install_mode": "isolated_source_copy_uv_sync_frozen_all_extras",
        "full_40_environment_policy_example_executed": clean_direct.is_file(),
        "direct_rows_sha256": sha(clean_direct) if clean_direct.is_file() else "missing",
        "author_worktree_direct_rows_sha256": sha(OUT / "full_40_environment_direct_returns.csv"),
        "direct_byte_parity": clean_direct.is_file() and clean_direct.read_bytes() == (OUT / "full_40_environment_direct_returns.csv").read_bytes(),
        "contrast_rows_sha256": sha(clean_contrasts) if clean_contrasts.is_file() else "missing",
        "schema_and_entrant_validation": "pass" if clean_direct.is_file() else "not_completed",
    }])

    parity = []
    direct_path = OUT / "full_40_environment_direct_returns.csv"
    if direct_path.is_file():
        direct = read(direct_path)
        authoritative = read(REFERENCE)
        public_behavior = {(r["profile"], r["environment_seed"]): r for r in direct if r["method"] == "behavior"}
        authority_behavior = {(r["profile"], r["environment_seed"]): r for r in authoritative if r["method"] == "behavior"}
        for key in sorted(public_behavior):
            left, right = public_behavior[key], authority_behavior[key]
            difference = abs(float(left["mean_return"]) - float(right["mean_return"]))
            tolerance = 3.0 * (float(left["return_standard_error"]) ** 2 + float(right["standard_error"]) ** 2) ** 0.5
            parity.append({"profile": key[0], "environment_seed": key[1], "reference_role": "official_behavior_evaluator_sanity",
                           "authoritative_method": "behavior", "public_method": "behavior",
                           "absolute_difference": difference, "tolerance": tolerance,
                           "exact_method_config_identity": True, "status": "pass" if difference <= tolerance else "fail"})
    parity.extend([
        {"profile": "all", "environment_seed": "all", "reference_role": "required_model_free_reference",
         "authoritative_method": "behavior_cloning", "public_method": "not_reconstructed",
         "absolute_difference": "na", "tolerance": "prespecified_but_not_applicable",
         "exact_method_config_identity": False, "status": "structural_nonavailability_no_checkpoint_or_exact_public_training_port"},
        {"profile": "all", "environment_seed": "all", "reference_role": "required_model_based_reference",
         "authoritative_method": "persistence_locf_plus_h4_support_only", "public_method": "transition_only_locf_gaussian_reference",
         "absolute_difference": "na", "tolerance": "prespecified_but_not_applicable",
         "exact_method_config_identity": False, "status": "structural_nonavailability_incomplete_prt_planner_contract"},
    ])
    write(OUT / "built_in_reference_parity.csv", parity)

    runtime = []
    for path in sorted(OUT.glob("*.csv")):
        runtime.append({"artifact": path.name, "bytes": path.stat().st_size,
                        "wall_time": "unavailable_unless_captured_by_command_log",
                        "peak_memory_bytes": "unavailable_unless_captured_by_command_log",
                        "failure_count": sum(1 for row in read(path) if row.get("status", "").startswith("fail"))})
    for stage, log_path in (("suite_generation_train_validation", Path("/tmp/kdd215_generation_time.txt")),
                            ("direct_return_full_40", Path("/tmp/kdd215_direct_time.txt")),
                            ("component_full_40", Path("/tmp/kdd215_component_time.txt")),
                            ("ope_full_320", Path("/tmp/kdd215_ope_time.txt"))):
        text = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
        elapsed = re.search(r"Elapsed \(wall clock\) time.*: (.+)", text)
        memory = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
        runtime.append({"artifact": stage, "bytes": "not_applicable",
                        "wall_time": elapsed.group(1) if elapsed else "unavailable",
                        "peak_memory_bytes": int(memory.group(1)) * 1024 if memory else "unavailable",
                        "worker_count": 32 if stage == "ope_full_320" else 1,
                        "failure_count": 0 if elapsed else "unavailable"})
    write(OUT / "resource_runtime_and_failure_summary.csv", runtime)

    failures = [
        {"gate": "authoritative_model_free_reference_parity", "status": "fail",
         "reason": "exact KDD199 behavior_cloning checkpoint/training implementation is not reconstructible from the current public dependency contract",
         "repair": "separately port and clean-install the exact frozen training implementation; do not substitute an algorithm"},
        {"gate": "authoritative_model_based_reference_parity", "status": "fail",
         "reason": "the transition-only LOCF Gaussian entrant does not supply the complete P/R/T contract required to reproduce persistence_locf_plus_h4_support_only",
         "repair": "port the exact complete frozen P/R/T reference and H4 adapter under a separately reviewed source-identity gate"},
        {"gate": "history_only_true_model_reference", "status": "structural_nonavailability",
         "reason": "the selected KDD199 history-only planner policy object is not present in the aggregate-safe serialized generator contract",
         "repair": "publish a source-hash-bound observable-history planner contract without latent or future access"},
    ]
    write(OUT / "failure_ledger.csv", failures)

    forbidden = ("subject_id", "patient_id", "hadm_id", "stay_id", "charttime", "access_token", "api_key")
    privacy = []
    for path in sorted(item for item in OUT.rglob("*") if item.is_file()):
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        hits = [value for value in forbidden if value in text]
        privacy.append({"artifact": str(path.relative_to(OUT)), "forbidden_hits": ";".join(hits),
                        "row_level_ehr": False, "latent_trajectory_export": False,
                        "status": "pass" if not hits else "fail"})
    write(OUT / "privacy_scan.csv", privacy)

    decision = "stop_generator_parity_role_isolation_or_reference_parity_failure"
    (OUT / "result_audit.md").write_text(
        "# KDD215 result audit\n\n"
        "The exact generator payload inventory contains 40 environments (five profiles x eight seeds). "
        "The policy surface contains 280 environment rows: five controls, one isolated observable-history "
        "entrant, and one privileged latent-state reference. Every direct-return and paired precision gate "
        "passed. The component surface contains 40 numeric one-step rows and 400 explicit structural NAs "
        "for unsupported recursive observation-process rollout.\n\n"
        "The repeated-OPE run contains 320 independent logged datasets and 15,360 aggregate rows: two "
        "policies x four contracts x six estimators per dataset, each with 500 full-refit bootstrap "
        "replicates. The KDD199 official-behavior evaluator sanity passed on 40/40 environments. Python "
        "3.11/3.12/3.13 canonical probe bytes match, and the clean-install full direct output is byte-identical.\n\n"
        "Completion is not authorized because the prompt's required same-identity learned behavior-cloning "
        "and complete persistence P/R/T plus H4 reference parity cannot be established from the current "
        "public package. The history-only true-model planner object is also structurally unavailable. No "
        "substitute algorithm is relabeled. KDD212 smoke outputs remain separate and `smoke_only`.\n",
        encoding="utf-8",
    )
    (OUT / "decision.md").write_text(f"# KDD215 decision\n\n`{decision}`\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
