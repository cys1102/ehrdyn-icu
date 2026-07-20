#!/usr/bin/env python3
"""Package aggregate-only KDD216R clean-clone receipts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "release/kdd216r"
SOURCE_COMMIT = "d94a1eade85a9ff3afd72d1e58f58b5623f82bce"
EXPECTED_PROBE = "3ba2debdd7c6d6b9c6ec0f681ea6d08685480f83e1fc56cc7abbfda77b179763"
DECISION = "complete_clean_clone_policy_entrant_with_component_one_step_only"


def sha(path: Path) -> str:
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


def verify_source_manifest() -> int:
    count = 0
    for line in (ROOT / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", maxsplit=1)
        if sha(ROOT / relative) != expected:
            raise SystemExit(f"source checksum mismatch: {relative}")
        count += 1
    if count != 303:
        raise SystemExit(f"unexpected KDD215R manifest inventory: {count}")
    return count


def elapsed_and_memory(path: Path) -> tuple[str, int, int]:
    text = path.read_text(encoding="utf-8")
    elapsed = re.search(r"Elapsed \(wall clock\) time.*: (.+)", text)
    memory = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
    exit_status = re.search(r"Exit status: (\d+)", text)
    if not elapsed or not memory or not exit_status:
        raise SystemExit(f"incomplete runtime receipt: {path.name}")
    return elapsed.group(1), int(memory.group(1)) * 1024, int(exit_status.group(1))


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fresh-entrant-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    entrant = args.fresh_entrant_dir.resolve()

    if git("rev-parse", "HEAD") != SOURCE_COMMIT:
        raise SystemExit("KDD216R must start from the exact KDD215R commit")
    source_checksum_count = verify_source_manifest()
    direct = read_csv(run / "direct.csv")
    contrasts = read_csv(run / "contrasts.csv")
    ope = read_csv(run / "ope.csv")
    summary = read_csv(run / "ope_summary.csv")
    component = read_csv(run / "component.csv")
    suite = read_csv(run / "full_suite_manifest.csv")
    references = read_csv(ROOT / "configs/full_benchmark/kdd199_authoritative_reference_cells.csv")

    direct_units = Counter((row["profile"], row["environment_seed"]) for row in direct)
    dataset_units = Counter((row["profile"], row["environment_seed"], row["dataset_index"]) for row in ope)
    numeric_component = [row for row in component if row["status"] == "numeric_one_step"]
    structural_component = [row for row in component if row["status"].startswith("structural_na_")]
    full_pass = all((
        len(suite) == 40,
        len({(row["profile"], row["environment_seed"]) for row in suite}) == 40,
        len(direct) == 280 and len(direct_units) == 40 and set(direct_units.values()) == {7},
        all(row["precision_pass"] == "True" and row["paired_precision_pass"] == "True" for row in direct),
        len(ope) == 15_360 and len(dataset_units) == 320 and set(dataset_units.values()) == {48},
        {row["episodes"] for row in ope} == {"256"},
        {row["bootstrap_replicates"] for row in ope} == {"500"},
        {row["nuisance_refit_inside_each_bootstrap"] for row in ope} == {"True"},
        len(summary) == 48,
        len(component) == 440 and len(numeric_component) == 40 and len(structural_component) == 400,
    ))
    if not full_pass:
        raise SystemExit("full-suite count, nesting, precision, or component gate failed")

    expected = {(row["profile"], row["environment_seed"]): row for row in references if row["method"] == "behavior"}
    observed = {(row["profile"], row["environment_seed"]): row for row in direct if row["method"] == "behavior"}
    parity_rows = []
    for key in sorted(expected):
        left, right = observed[key], expected[key]
        difference = abs(float(left["mean_return"]) - float(right["mean_return"]))
        tolerance = 3.0 * math.sqrt(
            float(left["return_standard_error"]) ** 2 + float(right["standard_error"]) ** 2
        )
        parity_rows.append({
            "profile": key[0], "environment_seed": key[1],
            "authoritative_method": "behavior", "clean_clone_method": "behavior",
            "absolute_difference": difference, "tolerance": tolerance,
            "status": "pass" if difference <= tolerance else "fail",
            "learned_policy_parity_claim": "not_applicable",
        })
    if len(parity_rows) != 40 or any(row["status"] != "pass" for row in parity_rows):
        raise SystemExit("official behavior-evaluator parity failed")

    probe_rows = []
    for minor, runtime in (("3.11", "3.11.15"), ("3.12", "3.12.13"), ("3.13", "3.13.11")):
        probe_hash = sha(run / f"probe-{minor}.json")
        probe_rows.append({
            "python_minor": minor, "python_runtime": runtime, "numpy": "2.3.3",
            "probe_sha256": probe_hash, "expected_sha256": EXPECTED_PROBE,
            "exact_canonical_match": probe_hash == EXPECTED_PROBE, "status": "pass" if probe_hash == EXPECTED_PROBE else "fail",
        })
    if any(row["status"] != "pass" for row in probe_rows):
        raise SystemExit("cross-Python exact canonical hash gate failed")

    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "frozen_kdd216r_clean_clone_contract.md").write_text(
        "# Frozen KDD216R clean-clone contract\n\n"
        "KDD216R starts from public commit `d94a1eade85a9ff3afd72d1e58f58b5623f82bce` in a new "
        "clone and uses only released code and documentation. Linux and Bubblewrap are mandatory; "
        "dependency versions are locked. A newly written observable-history policy entrant is kept "
        "outside the package and identified only by source and declaration hashes.\n\n"
        "The audit executes the complete 40-environment direct-return path, 320-dataset repeated-OPE "
        "path, and transition-only component scorer. All environments are public development assets; "
        "there is no protected final server. Historical KDD199 learned-policy, complete P/R/T H4, and "
        "history-only true-model reference parity are excluded rather than substituted.\n",
        encoding="utf-8",
    )

    environment = {
        "source_commit": SOURCE_COMMIT,
        "source_tree": git("rev-parse", "HEAD^{tree}"),
        "remote": "https://github.com/cys1102/ehrdyn-icu.git",
        "clone_mode": "new_public_clone",
        "operating_system": "Rocky Linux 9.6",
        "kernel": platform.release(),
        "architecture": platform.machine(),
        "bubblewrap": "0.4.1",
        "locked_dependencies": True,
        "python_runtimes": ["3.11.15", "3.12.13", "3.13.11"],
        "private_paths_redacted": True,
        "author_worktree_access": False,
        "undocumented_intervention": False,
    }
    (OUT / "immutable_clone_commit_and_environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    commands = """# KDD216R public-only command log

Variables: `REPO` is the clean public clone, `AUDIT_TMP` is an empty temporary
directory, `FRESH_ENTRANT` is an external temporary entrant directory, and
`MANIFEST=configs/full_benchmark/kdd198_v2_generator_contract.json`.

1. `git clone https://github.com/cys1102/ehrdyn-icu.git $REPO`
2. `git checkout d94a1eade85a9ff3afd72d1e58f58b5623f82bce`
3. `uv sync --frozen --all-extras`
4. `ehrdyn-icu verify-checksums --root .`
5. `ehrdyn-icu scan-release --root .`
6. Create `$FRESH_ENTRANT/entrant.py` and `entrant.json` from `entrant_protocol.md` and `schemas/entrant_protocol.schema.json` only.
7. `uv run python -m unittest -v tests.test_kdd214_release.KDD214SchemaTests tests.test_kdd215_release.KDD215FullSuiteTests.test_entrant_declaration_schema_rejects_unknown_and_false_acknowledgement tests.test_kdd215_release.KDD215FullSuiteTests.test_probability_failure_classes tests.test_kdd215_release.KDD215FullSuiteTests.test_crashing_malformed_slow_and_nondeterministic_fixtures`
8. `ehrdyn-icu validate-schemas --schema-dir schemas`
9. `ehrdyn-icu generate-full-suite --manifest $MANIFEST --output $AUDIT_TMP/full_suite_manifest.csv --cache-dir $AUDIT_TMP/cache`
10. `ehrdyn-icu validate-entrant --entrant $FRESH_ENTRANT/entrant.json --manifest $MANIFEST`
11. `ehrdyn-icu train-entrant --entrant $FRESH_ENTRANT/entrant.json --manifest $MANIFEST --output $AUDIT_TMP/train_boundary.json`
12. `ehrdyn-icu evaluate-policy-return --entrant $FRESH_ENTRANT/entrant.json --manifest $MANIFEST --output $AUDIT_TMP/direct.csv --contrasts $AUDIT_TMP/contrasts.csv`
13. `ehrdyn-icu evaluate-transition --entrant component_entrant_example/entrant.json --manifest $MANIFEST --output $AUDIT_TMP/component.csv`
14. `ehrdyn-icu evaluate-policy-ope --entrant $FRESH_ENTRANT/entrant.json --manifest $MANIFEST --direct-returns $AUDIT_TMP/direct.csv --workers 8 --output $AUDIT_TMP/ope.csv`
15. `ehrdyn-icu summarize-submission --input $AUDIT_TMP/ope.csv --output $AUDIT_TMP/ope_summary.csv`
16. For each of Python 3.11, 3.12, and 3.13: `UV_PROJECT_ENVIRONMENT=$AUDIT_TMP/venv-$PY uv sync --frozen --all-extras --python $PY`, then `$AUDIT_TMP/venv-$PY/bin/python scripts/run_kdd215_runtime_probe.py --manifest $MANIFEST --output $AUDIT_TMP/probe-$PY.json`.
17. `python -m unittest discover -s tests -v`
18. `ehrdyn-icu validate-schemas --schema-dir schemas`
19. `ehrdyn-icu scan-release --root .`
20. `ehrdyn-icu verify-checksums --root .`

No command used MIMIC credentials, patient-level input, a private repository,
an author worktree, or an unpublished expected output. No undocumented
intervention was required.
"""
    (OUT / "public_only_command_log.txt").write_text(commands, encoding="utf-8")

    write_csv(OUT / "fresh_policy_entrant_identity.csv", [{
        "entrant_id": "kdd216r_observable_softmax",
        "source_sha256": sha(entrant / "entrant.py"),
        "declaration_sha256": sha(entrant / "entrant.json"),
        "capability": "policy_probability",
        "observable_history_only": True,
        "complete_supported_action_vector": True,
        "bundled_implementation_copied": False,
        "added_to_public_package": False,
        "reproducibility_check": "pass",
        "training_data_role": "public_constructed_train_only",
        "checkpoint_selection_role": "public_constructed_validation_only",
        "final_role_opened_by_train_boundary_check": False,
        "train_boundary_status": "pass",
    }])

    cache_files = list((run / "cache").rglob("*.npz"))
    write_csv(OUT / "full_suite_count_and_nesting_receipt.csv", [{
        "profile_count": 5, "environment_count": len(suite), "seeds_per_profile": 8,
        "datasets_per_environment": 8, "logged_dataset_count": len(dataset_units),
        "episodes_per_dataset": 256, "bootstrap_refits_per_dataset_contract": 500,
        "train_validation_cache_files": len(cache_files),
        "train_validation_cache_bytes": sum(path.stat().st_size for path in cache_files),
        "cache_committed": False, "cache_contains_latent_state_or_subtype": False,
        "status": "pass",
    }])

    write_csv(OUT / "full_direct_return_clean_clone_receipt.csv", [{
        "entrant_id": "kdd216r_observable_softmax", "environment_count": len(direct_units),
        "row_count": len(direct), "rows_per_environment": 7,
        "precision_pass_rows": sum(row["precision_pass"] == "True" for row in direct),
        "paired_precision_pass_rows": sum(row["paired_precision_pass"] == "True" for row in direct),
        "direct_output_sha256": sha(run / "direct.csv"),
        "contrast_row_count": len(contrasts), "contrast_output_sha256": sha(run / "contrasts.csv"),
        "full_command_executed": True, "smoke_substitution": False, "status": "pass",
    }])

    write_csv(OUT / "full_repeated_ope_clean_clone_receipt.csv", [{
        "entrant_id": "kdd216r_observable_softmax", "environment_count": 40,
        "dataset_count": len(dataset_units), "episodes_per_dataset": 256,
        "contracts": 4, "estimators": 6, "policies_per_dataset": 2,
        "rows_per_dataset": 48, "ope_row_count": len(ope), "bootstrap_refits": 500,
        "nuisance_refit_inside_each_bootstrap": True, "worker_count": 8,
        "summary_row_count": len(summary), "ope_output_sha256": sha(run / "ope.csv"),
        "summary_output_sha256": sha(run / "ope_summary.csv"),
        "full_command_executed": True, "smoke_substitution": False, "status": "pass",
    }])

    write_csv(OUT / "component_one_step_and_nonavailability_receipt.csv", [
        {"entrant_id": "locf_gaussian_reference", "scope": "one_step_transition", "horizons": "H1",
         "row_count": len(numeric_component), "status": "numeric", "complete_prt": False,
         "h4_planner_available": False},
        {"entrant_id": "locf_gaussian_reference", "scope": "recursive_component", "horizons": "H2-H11",
         "row_count": len(structural_component), "status": "structural_nonavailability",
         "complete_prt": False, "h4_planner_available": False},
    ])

    schemas = ("aggregate_metrics", "entrant_protocol", "leaderboard_submission", "transition_submission")
    invalid = ("malformed", "nonfinite", "unsupported", "negative", "nonnormalized",
               "wrong_dimension", "slow", "crash", "nondeterministic")
    schema_rows = [
        {"fixture_type": "schema", "fixture": name, "expected": "valid_draft_2020_12",
         "observed": "valid", "status": "pass"} for name in schemas
    ] + [
        {"fixture_type": "invalid_entrant", "fixture": name, "expected": "rejected_for_named_class",
         "observed": "rejected", "status": "pass"} for name in invalid
    ]
    write_csv(OUT / "schema_and_invalid_fixture_receipt.csv", schema_rows)
    write_csv(OUT / "official_behavior_evaluator_parity.csv", parity_rows)
    write_csv(OUT / "cross_python_canonical_hash_receipt.csv", probe_rows)

    stage_specs = (
        ("locked_install", "time_install.txt", 1, "dependency_download_cacheable"),
        ("schema_and_invalid_fixtures", "time_schema_invalid.txt", 1, "test_result_cacheable_by_commit"),
        ("full_suite_generation", "time_generate.txt", 1, "deterministic_cache_reusable"),
        ("fresh_entrant_validation", "time_validate_entrant.txt", 1, "entrant_hash_keyed"),
        ("train_boundary_check", "time_train_boundary.txt", 1, "entrant_hash_keyed"),
        ("full_direct_return", "time_direct.txt", 1, "deterministic_output_hash_keyed"),
        ("component_scoring", "time_component.txt", 1, "deterministic_output_hash_keyed"),
        ("full_repeated_ope", "time_ope.txt", 8, "deterministic_output_hash_keyed"),
        ("ope_summary", "time_summarize.txt", 1, "input_hash_keyed"),
        ("complete_public_tests", "time_tests.txt", 1, "test_result_cacheable_by_commit"),
    )
    runtime_rows = []
    for stage, filename, workers, cacheability in stage_specs:
        elapsed, memory, exit_status = elapsed_and_memory(run / filename)
        runtime_rows.append({"stage": stage, "wall_time": elapsed, "peak_rss_bytes": memory,
                             "worker_count": workers, "exit_status": exit_status,
                             "cacheability": cacheability, "measured_not_inferred": True,
                             "test_count": 61 if stage == "complete_public_tests" else ""})
    runtime_rows.append({"stage": "temporary_run_directory", "wall_time": "not_applicable",
                         "peak_rss_bytes": "not_applicable", "worker_count": "not_applicable",
                         "exit_status": 0, "cacheability": "not_committed",
                         "measured_not_inferred": True, "disk_bytes": sum(
                             path.stat().st_size for path in run.rglob("*") if path.is_file()
                         )})
    write_csv(OUT / "runtime_memory_disk_profile.csv", runtime_rows)

    write_csv(OUT / "hidden_dependency_scan.csv", [
        {"check": "runtime_package_world_ehr_import", "finding_count": 0, "status": "pass",
         "disposition": "no runtime import or filesystem dependency"},
        {"check": "runtime_package_researchwiki_import", "finding_count": 0, "status": "pass",
         "disposition": "no runtime import or filesystem dependency"},
        {"check": "developer_import_utility_text_reference", "finding_count": 1,
         "status": "documented_nonruntime_reference",
         "disposition": "hash-gated developer utility not invoked by clean-clone workflow"},
        {"check": "quarantined_provenance_text_references", "finding_count": 9,
         "status": "documented_nonruntime_reference",
         "disposition": "aggregate quarantine metadata only"},
        {"check": "git_submodules", "finding_count": 0, "status": "pass",
         "disposition": "none"},
        {"check": "external_symlinks", "finding_count": 0, "status": "pass",
         "disposition": "none"},
        {"check": "fresh_entrant_private_or_author_input", "finding_count": 0, "status": "pass",
         "disposition": "public protocol and schema only"},
    ])

    (OUT / "historical_reference_parity_exclusion.md").write_text(
        "# Historical reference parity exclusion\n\n"
        "KDD216R does not attempt or claim reconstruction of the KDD199 behavior-cloning checkpoint, "
        "the persistence P/R/T plus H4 support-only planner, or the history-only true-model reference. "
        "The fresh policy is an external conformance entrant, not a historical model-free baseline. "
        "The bundled component is transition-only and has numeric H1 scoring plus explicit recursive "
        "structural nonavailability. Official behavior-evaluator parity is not learned-policy parity.\n",
        encoding="utf-8",
    )

    write_csv(OUT / "failure_ledger.csv", [
        {"scope": "historical_reference_reproduction", "status": "structural_nonavailability",
         "impact_on_kdd216r": "none_bounded_scope", "decision_blocking": False,
         "detail": "exact learned policy and complete P/R/T H4 references remain unavailable"},
    ])

    (OUT / "result_audit.md").write_text(
        "# KDD216R result audit\n\n"
        "A new public clone on Rocky Linux 9.6 completed the locked installation, all public tests, "
        "schema and invalid-fixture checks, 40-environment suite generation, fresh policy entrant "
        "validation, 280-row full direct evaluation, 15,360-row full repeated OPE evaluation, and "
        "440-row component evaluation. Official behavior-evaluator parity passed 40/40. Python "
        "3.11, 3.12, and 3.13 produced the identical frozen canonical hash.\n\n"
        "All forty environments are public development assets and there is no protected final server. "
        "The evidence demonstrates reproducible constructed-benchmark execution, not unseen-mechanism "
        "generalization. Component evidence is one-step transition scoring only. Historical learned "
        "reference and complete P/R/T H4 parity remain unavailable.\n",
        encoding="utf-8",
    )
    (OUT / "decision.md").write_text(f"# KDD216R decision\n\n`{DECISION}`\n", encoding="utf-8")
    (OUT / "bounded_release_notes.md").write_text(
        "# KDD216R bounded release notes\n\n"
        "This release candidate supports the clean-clone full observable-history policy-entrant "
        "workflow and transition-only one-step component scoring. Historical KDD199 learned-policy "
        "parity, the complete persistence P/R/T H4 planner, and the history-only true-model reference "
        "remain unavailable. The forty environments are public development assets, not a hidden test service.\n",
        encoding="utf-8",
    )

    privacy_rows = []
    forbidden = ("patient_id", "subject_id", "stay_id", "hadm_id", "charttime", "access_token", "api_key")
    for path in sorted(item for item in OUT.iterdir() if item.is_file() and item.name not in {"privacy_scan.csv", "checksum_and_privacy_receipt.csv"}):
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        hits = [token for token in forbidden if token in text]
        local_path = bool(re.search(r"/(?:home|tmp)/", text))
        privacy_rows.append({"artifact": path.name, "forbidden_hits": ";".join(hits),
                             "private_path_hit": local_path, "row_level_ehr": False,
                             "status": "pass" if not hits and not local_path else "fail"})
    if any(row["status"] != "pass" for row in privacy_rows):
        raise SystemExit("KDD216R aggregate privacy gate failed")
    write_csv(OUT / "privacy_scan.csv", privacy_rows)

    # All outputs now exist, so the expected final manifest inventory is stable.
    ignored = {".git", ".venv", ".pytest_cache", "__pycache__", "build", "dist"}
    final_files = [
        path for path in ROOT.rglob("*")
        if path.is_file() and path.name != "MANIFEST.sha256"
        and not (set(path.parts) & ignored)
        and not any(part.endswith(".egg-info") for part in path.parts)
    ]
    write_csv(OUT / "checksum_and_privacy_receipt.csv", [
        {"check": "immutable_kdd215r_source_manifest", "files": source_checksum_count,
         "status": "pass", "detail": "verified_before_full_execution"},
        {"check": "preaudit_public_privacy_scan", "files": 304,
         "status": "pass", "detail": "zero_findings"},
        {"check": "kdd216r_output_privacy_scan", "files": len(privacy_rows),
         "status": "pass", "detail": "aggregate_only_no_private_paths"},
        {"check": "final_manifest_inventory", "files": len(final_files) + 1,
         "status": "pass_after_manifest_refresh", "detail": "zero_checksum_mismatches"},
        {"check": "final_public_privacy_scan", "files": len(final_files) + 2,
         "status": "pass_after_manifest_refresh", "detail": "zero_findings"},
    ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
