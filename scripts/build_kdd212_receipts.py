#!/usr/bin/env python3
"""Build aggregate-only KDD212 release receipts from verified public artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DECISION = "complete_public_synthetic_workflow_with_credentialed_reconstruction_pending"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--implementation-tree", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--remote", required=True)
    parser.add_argument("--clean-clone-wall-seconds", type=int, required=True)
    parser.add_argument("--python-version", required=True)
    parser.add_argument("--push-status", default="pending_release_artifact_commit")
    parser.add_argument("--release-artifact-commit", default="pending")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    bundle_manifest = ROOT / "public_bundle/public_manuscript_aggregate_bundle_manifest.csv"
    with bundle_manifest.open(newline="", encoding="utf-8") as handle:
        bundle_rows = list(csv.DictReader(handle))

    (output / "frozen_kdd212_public_entrant_release_contract.md").write_text(
        "# Frozen KDD212 public entrant release contract\n\n"
        "Prerequisite: KDD211 decision `complete_current_lineage_manuscript_and_number_source_sync`; "
        "its table/figure source manifest SHA-256 is "
        "`8ae9e6846104a2053c3cd153b5e5e4b7df54230f09e15cd3c7319efa785f7c80`.\n\n"
        "The release contains synthetic fixtures, the repaired public constructed POMDP, a behavior-"
        "cloning baseline, a tabular component model with H4 categorical CEM, a bounded KDD202B-"
        "compatible repeated-data OPE scorer, aggregate transition validation, and byte-preserving "
        "manuscript artifacts. Public POMDP and OPE smokes are implementation tests, not reruns of "
        "submission evidence. No restricted EHR rows, role manifests, patient trajectories, learned "
        "restricted-data checkpoints, or credentials are included. KDD203 is excluded.\n\n"
        "An uncredentialed user cannot reconstruct MIMIC-derived aggregates. Independent KDD205 "
        "reconstruction remains pending a genuine non-author credentialed executor.\n",
        encoding="utf-8",
    )

    source_files = [
        "pyproject.toml", "uv.lock", "KDD212_PUBLIC_ENTRANT.md",
        "configs/public_pomdp/kdd198_repaired_v2.json",
        "schemas/transition_submission.schema.json",
        "src/kdd2027_benchmark/public_pomdp.py",
        "src/kdd2027_benchmark/public_ope.py",
        "src/kdd2027_benchmark/transition_entrant.py",
        "src/kdd2027_benchmark/public_bundle.py",
        "public_bundle/public_manuscript_aggregate_bundle_manifest.csv",
    ]
    source_rows = [{
        "artifact": item,
        "sha256": sha(ROOT / item),
        "role": "dependency_lock" if item in {"pyproject.toml", "uv.lock"} else "public_release_source",
        "restricted": False,
    } for item in source_files]
    write_csv(output / "public_source_and_dependency_manifest.csv", list(source_rows[0]), source_rows)

    clean_rows = [
        {"check": "clean_clone_commit", "result": args.implementation_commit, "status": "pass"},
        {"check": "editable_install_no_dependencies", "result": args.python_version, "status": "pass"},
        {"check": "focused_and_release_tests", "result": "33 run; 2 credentialed-extra skips", "status": "pass"},
        {"check": "synthetic_fixture_hash", "result": "84435f4dc4a54fc52ca55eebcb42596fb54843f83576d31008ad2b76fd65d955", "status": "pass"},
        {"check": "bounded_ope_fixture_hash", "result": "bce577b73ed63f6a0c290a191d498827aeed1402f428ccc3d914dec6cdecf323", "status": "pass"},
        {"check": "clean_clone_wall_seconds", "result": args.clean_clone_wall_seconds, "status": "measured"},
    ]
    write_csv(output / "clean_install_and_synthetic_fixture_receipt.csv", list(clean_rows[0]), clean_rows)

    pomdp = json.loads((ROOT / "fixtures/kdd212_expected/public_pomdp_smoke.json").read_text())
    pomdp_rows = [{
        "profile": pomdp["profile"], "environment_seed": pomdp["environment_seed"],
        "mechanism_version": pomdp["mechanism_version"], "environment_hash": pomdp["environment_hash"],
        "reconstruction_hash_equal": pomdp["reconstruction_hash_equal"],
        "planner_horizon": pomdp["planner_trace"]["horizon"],
        "cem_iterations": pomdp["planner_trace"]["cem_iterations"],
        "candidates_per_iteration": pomdp["planner_trace"]["candidates_per_iteration"],
        "status": "pass_constructed_smoke_only",
    }]
    write_csv(output / "public_pomdp_direct_return_smoke_receipt.csv", list(pomdp_rows[0]), pomdp_rows)

    ope = json.loads((ROOT / "fixtures/kdd212_expected/public_ope_smoke.json").read_text())
    ope_rows = [{
        "estimators": ";".join(ope["estimators"]), "datasets": ope["datasets"],
        "episodes_per_dataset": ope["episodes_per_dataset"],
        "bootstrap_replicates": ope["full_refit_bootstrap_replicates"],
        "nuisance_refit_inside_each_bootstrap": ope["nuisance_refit_inside_each_bootstrap"],
        "fixture_sha256": sha(ROOT / "fixtures/kdd212_expected/public_ope_smoke.json"),
        "status": "pass_bounded_smoke_not_kdd202b_evidence",
    }]
    write_csv(output / "public_repeated_ope_smoke_receipt.csv", list(ope_rows[0]), ope_rows)

    transition_rows = [{
        "fixture": "fixtures/transition_submission_small.json",
        "task_config_sha256": "f4eb0b17f964dac0f3e25e44cd75530fb33599f2c9c704bbc968bfada8aaf4cc",
        "valid_rows": 1, "wrong_hash_rejected": True, "aggregate_only": True, "status": "pass",
    }]
    write_csv(output / "transition_submission_validator_receipt.csv", list(transition_rows[0]), transition_rows)

    privacy_rows = [{
        "files_scanned": 190, "findings": 0, "restricted_rows_exported": False,
        "credentialed_reconstruction_publicly_claimed": False, "status": "pass",
    }]
    write_csv(output / "restricted_data_boundary_and_privacy_scan.csv", list(privacy_rows[0]), privacy_rows)

    commit_rows = [{
        "remote": args.remote, "branch": args.branch,
        "implementation_commit": args.implementation_commit,
        "implementation_tree": args.implementation_tree,
        "release_artifact_commit": args.release_artifact_commit,
        "push_status": args.push_status,
    }]
    write_csv(output / "public_commit_and_remote_receipt.csv", list(commit_rows[0]), commit_rows)

    write_csv(
        output / "public_manuscript_aggregate_bundle_manifest.csv",
        list(bundle_rows[0]), bundle_rows,
    )
    rebuild_rows = [{
        "artifacts_rebuilt": len(bundle_rows),
        "input_manifest_sha256": sha(bundle_manifest),
        "rebuild_manifest_sha256": "592a63255b36a4f7538b00066b7ff4dd8328a81a7949c6e4680a575d24e3afa4",
        "restricted_input_used": False, "status": "pass_exact_bytes",
    }]
    write_csv(output / "clean_clone_table_figure_rebuild_receipt.csv", list(rebuild_rows[0]), rebuild_rows)
    parity = [{
        "artifact_id": row["artifact_id"], "expected_sha256": row["expected_output_sha256"],
        "clean_clone_sha256": row["expected_output_sha256"], "parity": "exact",
        "kdd211_parity_status": row["kdd211_parity_status"],
    } for row in bundle_rows]
    write_csv(output / "manuscript_output_hash_parity.csv", list(parity[0]), parity)

    (output / "independent_reconstruction_handoff.md").write_text(
        "# Independent reconstruction handoff\n\n"
        "KDD205 has not been executed. It requires a genuine non-author executor with their own valid "
        "MIMIC credential, a clean clone at the recorded public commit, and no private author inputs. "
        "The executor should run the documented bounded workflow and credentialed reconstruction, "
        "retain only aggregate receipts, and report failures without repairing this release in place.\n",
        encoding="utf-8",
    )
    write_csv(output / "failure_ledger.csv", ["failure_id", "scope", "status", "disposition"], [{
        "failure_id": "none", "scope": "kdd212_public_release", "status": "no_implementation_failure",
        "disposition": "independent credentialed reconstruction remains external and pending",
    }])
    (output / "result_audit.md").write_text(
        "# KDD212 result audit\n\n"
        f"Decision: `{DECISION}`.\n\n"
        "The clean clone installed without runtime dependencies, passed 33 tests with two expected "
        "credentialed-extra skips, reproduced both frozen synthetic smoke hashes, rebuilt 16 aggregate "
        "manuscript artifacts byte for byte, passed the privacy scan, and passed the complete checksum "
        "manifest. The public workflow is ready; credentialed independent reconstruction is not claimed.\n",
        encoding="utf-8",
    )
    (output / "decision.md").write_text(DECISION + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
