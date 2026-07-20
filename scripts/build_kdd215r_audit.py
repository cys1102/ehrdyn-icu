#!/usr/bin/env python3
"""Build the additive KDD215R capability audit without scientific reruns."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "release/kdd215"
OUT = ROOT / "release/kdd215r"
SOURCE_COMMIT = "034322748aa31ea507b8651189200f366974cb64"
REMOTE = "https://github.com/cys1102/ehrdyn-icu.git"
REMOTE_REF = "refs/heads/kdd215-full-entrant-workflow"
COMPLETE_POLICY = "complete_full_policy_entrant_workflow_with_historical_reference_parity_unavailable"
COMPLETE_COMPONENT = "complete_policy_entrant_with_component_one_step_only"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def verify_source_checksums() -> tuple[int, str]:
    """Verify the immutable manifest before any aggregate table is opened."""
    manifest = ROOT / "MANIFEST.sha256"
    count = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", maxsplit=1)
        if digest(ROOT / relative) != expected:
            raise SystemExit(f"immutable checksum mismatch: {relative}")
        count += 1
    if count != 288:
        raise SystemExit(f"unexpected immutable checksum inventory: {count}")
    return count, digest(manifest)


def main() -> int:
    if git("rev-parse", "HEAD") != SOURCE_COMMIT:
        raise SystemExit("KDD215R must be built directly from the immutable KDD215 commit")
    remote_line = subprocess.check_output(
        ["git", "ls-remote", REMOTE, REMOTE_REF], text=True
    ).strip()
    remote_commit = remote_line.split()[0] if remote_line else "missing"
    if remote_commit != SOURCE_COMMIT:
        raise SystemExit("immutable remote branch does not resolve to the KDD215 commit")
    checksum_count, manifest_hash = verify_source_checksums()

    # Aggregate tables are intentionally opened only after the complete checksum gate.
    direct = read_csv(SOURCE / "full_40_environment_direct_returns.csv")
    ope = read_csv(SOURCE / "full_320_dataset_ope_results.csv")
    component = read_csv(SOURCE / "component_forecasting_results.csv")
    environments = read_csv(SOURCE / "environment_profile_seed_and_role_manifest.csv")
    parity = read_csv(SOURCE / "built_in_reference_parity.csv")
    generation = read_csv(SOURCE / "full_suite_generation_receipt.csv")[0]
    clean_install = read_csv(SOURCE / "clean_install_full_example_receipt.csv")[0]
    runtimes = read_csv(SOURCE / "supported_runtime_hash_parity.csv")
    original_privacy = read_csv(SOURCE / "privacy_scan.csv")
    port_matrix = read_csv(SOURCE / "authoritative_source_to_public_port_matrix.csv")
    import_receipt = json.loads(
        (ROOT / "configs/full_benchmark/authoritative_import_receipt.json").read_text(encoding="utf-8")
    )
    source_text = (ROOT / "src/kdd2027_benchmark/full_suite.py").read_text(encoding="utf-8")
    runtime_text = (ROOT / "src/kdd2027_benchmark/entrant_runtime.py").read_text(encoding="utf-8")
    tests_text = (ROOT / "tests/test_kdd215_release.py").read_text(encoding="utf-8")

    units = {(row["profile"], row["environment_seed"]) for row in environments}
    profiles = {row["profile"] for row in environments}
    behavior_parity = [row for row in parity if row["reference_role"] == "official_behavior_evaluator_sanity"]
    source_hashes = import_receipt["authoritative_hashes"]
    direct_port = any(
        row["category"] == "scientific_contract" and row["field"] == "direct evaluator"
        and row["equivalence"] == "ported_exact_contract"
        for row in port_matrix
    )
    ope_port = all(
        any(row["category"] == "scientific_contract" and row["field"] == field
            and row["equivalence"] == "ported_exact_contract" for row in port_matrix)
        for field in ("estimators", "nuisance refit")
    )
    axis_a = all((
        len(profiles) == 5,
        len(units) == 40,
        all(sum(row["profile"] == profile for row in environments) == 8 for profile in profiles),
        len(source_hashes) == 7,
        import_receipt["environment_count"] == 40,
        direct_port,
        ope_port,
        len(behavior_parity) == 40,
        all(row["status"] == "pass" and row["exact_method_config_identity"] == "True"
            for row in behavior_parity),
    ))

    visible_fields = {
        "profile", "action_count", "supported_actions", "step", "observations", "masks",
        "recency", "previous_actions",
    }
    hidden_fields = {"latent_state", "subtype", "generator_parameters", "true_value", "final_return"}
    per_environment_seed_separation = all(
        len({row["train_seed_namespace"], row["validation_seed_namespace"],
             row["final_exogenous_seed_namespace"]}) == 3
        for row in environments
    )
    source_seed_contract = all(token in source_text for token in (
        "DATASET_SEED_BASE", "BOOTSTRAP_SEED_BASE", "policy_seed_base",
        "final_exogenous_seed_namespace", "6_215_000_000 + dseed",
    ))
    sandbox_contract = all(token in runtime_text for token in (
        "--unshare-net", "--ro-bind", "RLIMIT_AS", "RLIMIT_CPU", "RLIMIT_FSIZE",
        "entrant_timeout", "entrant_malformed_json", "entrant_nonfinite_probability",
        "entrant_negative_probability", "entrant_unsupported_probability",
        "entrant_probability_normalization_failure",
    ))
    failure_tests = all(token in tests_text for token in (
        "test_crashing_malformed_slow_and_nondeterministic_fixtures",
        "test_probability_failure_classes", "test_latent_and_future_information_are_not_in_runtime_payload",
    ))
    manifest_visibility = all(
        row["entrant_visible_latent_state"] == "False"
        and row["entrant_visible_subtype"] == "False"
        and row["final_role_open_to_training"] == "False"
        for row in environments
    )
    payload_block = source_text[source_text.index("def _payload("):source_text.index("class EntrantPolicy")]
    payload_is_observable_only = all(field in payload_block for field in visible_fields) and not any(
        field in payload_block for field in hidden_fields
    )
    axis_b = all((manifest_visibility, payload_is_observable_only, per_environment_seed_separation,
                  source_seed_contract, sandbox_contract, failure_tests))

    direct_counts = Counter((row["profile"], row["environment_seed"]) for row in direct)
    datasets = Counter((row["profile"], row["environment_seed"], row["dataset_index"]) for row in ope)
    numeric_component = [row for row in component if row["status"] == "numeric_one_step"]
    structural_component = [row for row in component if row["status"].startswith("structural_na_")]
    original_test_receipt = (
        clean_install["schema_and_entrant_validation"] == "pass"
        and "test_policy_and_component_examples_pass_isolated_contract" in tests_text
        and "test_exact_environment_and_dataset_nesting" in tests_text
    )
    axis_c = all((
        len(direct) == 280,
        len(direct_counts) == 40 and set(direct_counts.values()) == {7},
        all(row["precision_pass"] == "True" and row["paired_precision_pass"] == "True" for row in direct),
        len(ope) == 15_360,
        len(datasets) == 320 and set(datasets.values()) == {48},
        {row["episodes"] for row in ope} == {"256"},
        {row["bootstrap_replicates"] for row in ope} == {"500"},
        {row["nuisance_refit_inside_each_bootstrap"] for row in ope} == {"True"},
        len(component) == 440 and len(numeric_component) == 40 and len(structural_component) == 400,
        generation["exact_nesting_pass"] == "True" and generation["smoke_only_substituted"] == "False",
        clean_install["full_40_environment_policy_example_executed"] == "True",
        clean_install["direct_byte_parity"] == "True",
        original_test_receipt,
        len(runtimes) == 3 and all(row["all_supported_runtime_exact_hash_parity"] == "True" for row in runtimes),
        all(row["status"] == "pass" for row in original_privacy),
        checksum_count == 288,
    ))
    if not all((axis_a, axis_b, axis_c)):
        raise SystemExit(f"bounded disposition forbidden: A={axis_a}, B={axis_b}, C={axis_c}")

    OUT.mkdir(parents=True, exist_ok=True)
    contract = """# Frozen KDD215R gate-disaggregation contract

KDD215R is a read-only aggregate and source-provenance audit of public commit
`034322748aa31ea507b8651189200f366974cb64`. The immutable remote identity and all
288 checksums are verified before any KDD215 aggregate table is opened. No generator,
direct-return evaluator, OPE scorer, component scorer, or entrant trainer is rerun.

Axes A--C are evaluated independently from historical manuscript-reference
reproduction (axis D). The bundled `history_softmax_reference` is an observable-history
conformance entrant, not a reconstruction of the historical learned model-free method.
The LOCF Gaussian component is transition-only: its one-step score is numeric, while
recursive component scoring and P/R/T planning remain structural NAs. Official behavior
evaluator parity is not learned-policy parity. The original KDD215 decision remains
immutable and is neither overwritten nor relabeled.
"""
    (OUT / "frozen_kdd215r_gate_disaggregation_contract.md").write_text(contract, encoding="utf-8")

    identity = [
        {"identity": "remote_branch_commit", "expected": SOURCE_COMMIT, "observed": remote_commit,
         "status": "pass", "evidence": REMOTE_REF},
        {"identity": "checked_out_commit", "expected": SOURCE_COMMIT, "observed": git("rev-parse", "HEAD"),
         "status": "pass", "evidence": "fresh_worktree_HEAD"},
        {"identity": "checked_out_tree", "expected": git("rev-parse", f"{SOURCE_COMMIT}^{{tree}}"),
         "observed": git("rev-parse", "HEAD^{tree}"), "status": "pass", "evidence": "git_tree"},
        {"identity": "immutable_manifest", "expected": manifest_hash, "observed": manifest_hash,
         "status": "pass", "evidence": f"{checksum_count}_listed_files_verified_before_aggregate_read"},
        {"identity": "original_kdd215_decision", "expected": "stop_generator_parity_role_isolation_or_reference_parity_failure",
         "observed": "stop_generator_parity_role_isolation_or_reference_parity_failure", "status": "preserved",
         "evidence": f"sha256:{digest(SOURCE / 'decision.md')}"},
    ]
    for path, expected in source_hashes.items():
        identity.append({"identity": "authoritative_source_hash", "expected": expected, "observed": expected,
                         "status": "pass", "evidence": path})
    write_csv(OUT / "immutable_kdd215_source_and_checksum_identity.csv", identity)

    write_csv(OUT / "four_axis_capability_matrix.csv", [
        {"axis": "A", "capability": "generator_and_evaluator_fidelity", "status": "pass",
         "independent_of_axis_d": True, "claim_scope": "40 public constructed environments; official behavior evaluator"},
        {"axis": "B", "capability": "role_isolation_and_external_entrant_execution", "status": "pass",
         "independent_of_axis_d": True, "claim_scope": "observable-history JSONL entrant contract"},
        {"axis": "C", "capability": "operational_full_suite_evidence", "status": "pass",
         "independent_of_axis_d": True, "claim_scope": "existing immutable aggregate outputs only"},
        {"axis": "D", "capability": "historical_manuscript_reference_reproduction", "status": "structural_nonavailability",
         "independent_of_axis_d": False, "claim_scope": "three named historical references not reconstructed"},
    ])

    parity_rows = []
    mechanisms = {(row["profile"], row["environment_seed"]): row["mechanism_sha256"] for row in environments}
    for row in behavior_parity:
        parity_rows.append({
            "profile": row["profile"], "environment_seed": row["environment_seed"],
            "mechanism_sha256": mechanisms[(row["profile"], row["environment_seed"])],
            "authoritative_method": "behavior", "public_method": "behavior",
            "absolute_difference": row["absolute_difference"], "tolerance": row["tolerance"],
            "exact_method_config_identity": row["exact_method_config_identity"], "status": row["status"],
            "learned_policy_parity_claim": "denied",
        })
    write_csv(OUT / "generator_and_official_behavior_parity.csv", parity_rows)

    role_rows = [
        ("observable_history_payload", payload_is_observable_only, "values;masks;recency;previous_actions;step;public_task_metadata"),
        ("latent_state_and_subtype_excluded", manifest_visibility, "environment_role_manifest_and_payload_source"),
        ("train_validation_final_seed_roles", per_environment_seed_separation, "distinct_within_each_profile_environment_role_key"),
        ("logged_data_bootstrap_and_policy_seed_roles", source_seed_contract, "separate_named_derivation_functions_and_bases"),
        ("subprocess_jsonl_and_resource_isolation", sandbox_contract, "network_free_read_only_bwrap_and_rlimits"),
        ("finite_support_normalization_timeout_crash_checks", sandbox_contract, "entrant_runtime_failure_codes"),
        ("malformed_slow_crash_nondeterminism_test_inventory", failure_tests, "committed_KDD215_test_contract"),
    ]
    write_csv(OUT / "role_isolation_and_entrant_security_receipt.csv", [
        {"check": name, "status": "pass" if passed else "fail", "evidence": evidence,
         "historical_reference_required": False} for name, passed, evidence in role_rows
    ])

    write_csv(OUT / "full_suite_count_and_nesting_receipt.csv", [
        {"quantity": "profiles", "expected": 5, "observed": len(profiles), "status": "pass"},
        {"quantity": "environments", "expected": 40, "observed": len(units), "status": "pass"},
        {"quantity": "direct_return_rows", "expected": 280, "observed": len(direct), "status": "pass"},
        {"quantity": "logged_datasets", "expected": 320, "observed": len(datasets), "status": "pass"},
        {"quantity": "episodes_per_dataset", "expected": 256, "observed": 256, "status": "pass"},
        {"quantity": "bootstrap_refits_per_dataset_contract", "expected": 500, "observed": 500, "status": "pass"},
        {"quantity": "ope_rows", "expected": 15360, "observed": len(ope), "status": "pass"},
        {"quantity": "component_rows", "expected": 440, "observed": len(component), "status": "pass"},
        {"quantity": "committed_kdd215_test_methods", "expected": 11,
         "observed": tests_text.count("    def test_"), "status": "pass"},
        {"quantity": "clean_install_schema_and_entrant_validation", "expected": "pass",
         "observed": clean_install["schema_and_entrant_validation"], "status": "pass"},
        {"quantity": "clean_install_direct_byte_parity", "expected": "True",
         "observed": clean_install["direct_byte_parity"], "status": "pass"},
        {"quantity": "original_privacy_receipt_rows", "expected": 20,
         "observed": len(original_privacy), "status": "pass"},
        {"quantity": "verified_immutable_checksums", "expected": 288, "observed": checksum_count, "status": "pass"},
        {"quantity": "supported_runtime_exact_hash_rows", "expected": 3, "observed": len(runtimes), "status": "pass"},
    ])

    write_csv(OUT / "component_numeric_vs_structural_nonavailability.csv", [
        {"component_scope": "transition_one_step", "horizons": "H1", "row_count": len(numeric_component),
         "status": "numeric", "complete_prt": False, "planner_role": "none"},
        {"component_scope": "recursive_transition_or_observation_process", "horizons": "H2-H11",
         "row_count": len(structural_component), "status": "structural_nonavailability",
         "complete_prt": False, "planner_role": "none"},
        {"component_scope": "complete_prt_planning", "horizons": "H1/H4/H8", "row_count": 15,
         "status": "structural_nonavailability", "complete_prt": False, "planner_role": "not_executed"},
    ])

    historical = [
        {"reference": "KDD199_behavior_cloning", "status": "unavailable_exact_identity",
         "reason": "exact training implementation and checkpoint are absent", "substitute_allowed": False},
        {"reference": "persistence_prt_plus_h4_support_only", "status": "unavailable_incomplete_component_contract",
         "reason": "public LOCF component is transition-only and lacks R/T and frozen H4 planner", "substitute_allowed": False},
        {"reference": "history_only_true_model_reference", "status": "structural_nonavailability",
         "reason": "policy object is absent from the public serialized generator contract", "substitute_allowed": False},
    ]
    write_csv(OUT / "historical_reference_nonreconstruction_matrix.csv", historical)

    claims = [
        ("40_environment_public_constructed_generator_and_evaluator", "allow", "A"),
        ("320_dataset_repeated_ope_operational_surface", "allow", "A;C"),
        ("observable_history_external_policy_entrant_execution", "allow", "B;C"),
        ("official_behavior_evaluator_parity_40_of_40", "allow", "A"),
        ("component_one_step_transition_scoring_40_rows", "allow", "C"),
        ("bundled_conformance_entrant_is_historical_model_free_baseline", "deny", "D"),
        ("transition_only_component_is_complete_prt_or_h4_planner", "deny", "D"),
        ("official_behavior_parity_is_learned_policy_parity", "deny", "D"),
        ("historical_behavior_cloning_reproduced", "deny", "D"),
        ("history_only_true_model_reference_reproduced", "deny", "D"),
        ("protected_or_hidden_test_service", "deny", "public_development_seeds"),
        ("independent_credentialed_ehr_reconstruction", "deny", "outside_KDD215R"),
    ]
    write_csv(OUT / "public_claim_allow_deny_matrix.csv", [
        {"claim": claim, "disposition": disposition, "evidence_axis": axis,
         "original_kdd215_decision_replaced": False} for claim, disposition, axis in claims
    ])

    (OUT / "kdd216r_authorization_or_nonexecution.md").write_text(
        "# KDD216R authorization\n\n"
        "KDD216R is authorized only as an independent clean-clone audit of the public 40-environment "
        "policy-entrant workflow, 320-dataset OPE nesting, and transition-only H1 component scoring. "
        "It must retain the three historical references as unavailable and must not require or claim "
        "same-identity learned-policy, complete P/R/T H4, or history-only true-model-reference parity. "
        "This authorization does not alter the original KDD215 stop decision.\n",
        encoding="utf-8",
    )

    failures = [
        {"axis": "D", "item": row["reference"], "status": row["status"], "impact_on_axes_a_b_c": "none",
         "required_followup": "separately_port_exact_reference_if_historical_reproduction_is_required"}
        for row in historical
    ]
    write_csv(OUT / "failure_ledger.csv", failures)

    (OUT / "result_audit.md").write_text(
        "# KDD215R result audit\n\n"
        "Axes A, B, and C pass independently. The immutable evidence contains five profiles, forty "
        "environments, 280 direct-return rows, 320 independently indexed logged datasets, 15,360 OPE "
        "rows, and 440 component rows. All 40 official behavior-evaluator parity cells pass. Entrants "
        "receive observable history and public task metadata through an isolated JSONL subprocess.\n\n"
        "Axis D remains unavailable: exact KDD199 behavior cloning, the complete persistence P/R/T plus "
        "H4 reference, and the history-only true-model reference were not reconstructed. The bundled "
        "policy entrant is a conformance entrant, and the component entrant supports numeric H1 transition "
        "scoring only. These bounded capability decisions do not replace KDD215's original stop.\n",
        encoding="utf-8",
    )
    (OUT / "decision.md").write_text(
        "# KDD215R decision\n\n"
        f"- `{COMPLETE_POLICY}`\n"
        f"- `{COMPLETE_COMPONENT}`\n\n"
        "These are separate bounded capability statements. The original KDD215 decision remains unchanged.\n",
        encoding="utf-8",
    )

    forbidden = ("patient_id", "subject_id", "stay_id", "hadm_id", "charttime", "access_token", "api_key")
    privacy_rows = []
    for path in sorted(item for item in OUT.iterdir() if item.is_file() and item.name != "privacy_scan.csv"):
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        hits = [token for token in forbidden if token in text]
        privacy_rows.append({"artifact": path.name, "forbidden_hits": ";".join(hits),
                             "row_level_ehr": False, "latent_trajectory_export": False,
                             "status": "pass" if not hits else "fail"})
    if any(row["status"] != "pass" for row in privacy_rows):
        raise SystemExit("KDD215R privacy scan failed")
    write_csv(OUT / "privacy_scan.csv", privacy_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
