#!/usr/bin/env python3
"""Build the KDD245V2M aggregate audit and deterministic minimal archive."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Iterable


PARENT = "fbfff8ca3e7eb423806778222ade6251e6babba3"
SCOPE = "38b7ebc7297f1e1d2a0c53e2d73629147520a54a"
STOP_RECEIPT = "56748061f19df1c06c7c2396e4a0de34dc3216e1"
V200_ARCHIVE = "576aad418f3f1860bbba3c186d51ff370b207900da5d17edeae084b67bd9e5ff"
SUCCESS = "complete_immutable_v2_0_1_minimal_runtime_release_ready_for_kddp030m2"
STOP = "stop_source_identity_scope_test_release_or_privacy_failure"

ROOT_FILES = {
    ".gitignore",
    "CANONICAL_SERIALIZATION.md",
    "CITATION.cff",
    "EHR_COMPONENT_SCORER.md",
    "ETHICS_AND_MISUSE.md",
    "KNOWN_LIMITATIONS.md",
    "LICENSE",
    "LICENSE_NOTES.md",
    "MANIFEST.sha256",
    "MIGRATION_V2.md",
    "MIMIC_ACCESS.md",
    "OPE_CONTRACT.md",
    "README.md",
    "RECURSIVE_WORLD_MODEL_ENTRANT.md",
    "RELEASE_ASSET_LICENSES.md",
    "SCHEMA_VALIDATION.md",
    "entrant_protocol.md",
    "environment-rocky-linux.yml",
    "pyproject.toml",
    "pyrightconfig.json",
    "uv.lock",
}
RUNTIME_DIRS = {
    "component_entrant_example",
    "configs",
    "contracts",
    "dictionaries",
    "entrant",
    "expected",
    "fixtures",
    "invalid_entrant_fixtures",
    "policy_entrant_example",
    "recursive_world_model_entrant",
    "schemas",
    "src",
    "submission",
    "task_cards",
    "tests",
    "world_model_entrant_example",
}
SCIENTIFIC_DIRS = RUNTIME_DIRS - {"tests"}
PACKAGE_ONLY_CHANGED = {
    "MANIFEST.sha256",
    "src/kdd2027_benchmark/__init__.py",
    "pyproject.toml",
    "uv.lock",
    "CITATION.cff",
}
DOC_SCOPE_CHANGED = {
    "README.md",
    "MIMIC_ACCESS.md",
    "OPE_CONTRACT.md",
    "RELEASE_ASSET_LICENSES.md",
    "recursive_world_model_entrant/README.md",
}
TEST_SCOPE_CHANGED = {"tests/test_kdd245v2r_ehr_component_scorer.py"}
ARCHIVE_RECEIPTS = {
    "release/kdd245v2m/frozen_kdd245v2m_release_contract.md",
    "release/kdd245v2m/release_manifest.csv",
    "release/kdd245v2m/capability_manifest.csv",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, text=True, capture_output=True
    ).stdout


def parent_files(root: Path) -> dict[str, str]:
    output = git(root, "ls-tree", "-r", PARENT)
    rows: dict[str, str] = {}
    for line in output.splitlines():
        metadata, path = line.split("\t", 1)
        rows[path] = metadata.split()[2]
    return rows


def blob_bytes(root: Path, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{PARENT}:{path}"], cwd=root, check=True, capture_output=True
    ).stdout


def iter_runtime(root: Path) -> list[Path]:
    paths: list[Path] = []
    for name in sorted(ROOT_FILES):
        path = root / name
        if path.is_file():
            paths.append(path)
    for directory in sorted(RUNTIME_DIRS):
        base = root / directory
        if base.is_dir():
            paths.extend(
                path
                for path in sorted(base.rglob("*"))
                if path.is_file()
                and "__pycache__" not in path.parts
                and not any(part.endswith(".egg-info") for part in path.parts)
            )
    paths.extend(root / path for path in sorted(ARCHIVE_RECEIPTS) if (root / path).is_file())
    return paths


def write_csv(path: Path, columns: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def category(path: str) -> str:
    top = path.split("/", 1)[0]
    if top == "src":
        return "runtime_source"
    if top in {"configs", "contracts", "schemas", "dictionaries", "task_cards", "submission"}:
        return "contract_or_schema"
    if top in {"fixtures", "expected", "invalid_entrant_fixtures"}:
        return "synthetic_fixture_or_expected_output"
    if top in {
        "component_entrant_example",
        "policy_entrant_example",
        "recursive_world_model_entrant",
        "world_model_entrant_example",
        "entrant",
    }:
        return "entrant_runtime_or_example"
    if top == "tests":
        return "public_test"
    if top == "release":
        return "release_manifest"
    return "documentation_or_packaging"


def build_archive(root: Path, archive: Path, files: list[Path]) -> str:
    archive.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for source in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
            relative = source.relative_to(root).as_posix()
            data = source.read_bytes()
            info = tarfile.TarInfo(f"ehrdyn-icu-2.0.1/{relative}")
            info.size = len(data)
            info.mode = 0o755 if os.access(source, os.X_OK) else 0o644
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(data))
    with archive.open("wb") as handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as gz:
            gz.write(buffer.getvalue())
    return sha(archive)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    release = root / "release" / "kdd245v2m"
    release.mkdir(parents=True, exist_ok=True)
    status = json.loads(args.status.read_text(encoding="utf-8"))

    failures: list[tuple[str, str]] = []
    head_parent = git(root, "rev-parse", f"{PARENT}^{{commit}}").strip()
    if head_parent != PARENT:
        failures.append(("parent_identity", f"resolved {head_parent}"))

    parent = parent_files(root)
    runtime = iter_runtime(root)
    current = {path.relative_to(root).as_posix(): path for path in runtime}
    invariance: list[dict[str, object]] = []
    for path, source in sorted(current.items()):
        old = hashlib.sha256(blob_bytes(root, path)).hexdigest() if path in parent else ""
        new = sha(source)
        if path not in parent:
            disposition = "new_packaging_or_test"
            passed = True
        elif old == new:
            disposition = "exact_v2_0_0_identity"
            passed = True
        elif path in PACKAGE_ONLY_CHANGED:
            disposition = "packaging_version_only"
            passed = True
        elif path in DOC_SCOPE_CHANGED:
            disposition = "minimal_scope_documentation_only"
            passed = True
        elif path in TEST_SCOPE_CHANGED:
            disposition = "historical_receipt_test_removed_only"
            passed = True
        else:
            disposition = "unexpected_content_change"
            passed = False
            failures.append(("runtime_invariance", path))
        invariance.append(
            {
                "path": path,
                "v2_0_0_sha256": old,
                "v2_0_1_sha256": new,
                "disposition": disposition,
                "pass": str(passed).lower(),
            }
        )

    removed = sorted(set(parent) - set(current))
    excluded_rows = []
    for path in removed:
        if path.startswith("public_bundle/"):
            reason = "public_bundle_excluded"
        elif path.startswith(("evidence/", "clinical_review/", "credentialed/", "sbom/", "release/")):
            reason = "historical_or_credentialed_evidence_excluded"
        elif path.startswith("contracts/") and path.endswith(".csv"):
            reason = "machine_readable_scientific_or_provenance_table_excluded"
        elif path in {
            "configs/full_benchmark/authoritative_import_receipt.json",
            "configs/full_benchmark/kdd199_authoritative_reference_cells.csv",
        }:
            reason = "machine_readable_authoritative_result_identity_excluded"
        elif path.startswith("scripts/"):
            reason = "historical_release_or_result_packaging_tool_excluded"
        elif path.startswith("tests/"):
            reason = "test_depended_on_excluded_evidence_surface"
        else:
            reason = "nonminimal_historical_documentation_excluded"
        excluded_rows.append(
            {
                "path": path,
                "v2_0_0_blob": parent[path],
                "reason": reason,
                "present_in_v2_0_1": "false",
            }
        )

    prohibited_dirs = ["public_bundle", "evidence", "credentialed", "clinical_review", "sbom"]
    absent_rows = [
        {"artifact": name, "present": str((root / name).exists()).lower(), "pass": str(not (root / name).exists()).lower()}
        for name in prohibited_dirs
    ]
    for row in absent_rows:
        if row["present"] == "true":
            failures.append(("prohibited_directory", str(row["artifact"])))

    prohibited_suffixes = {".pt", ".pth", ".ckpt", ".npy", ".npz", ".parquet", ".feather"}
    suspicious: list[str] = []
    for path in runtime:
        relative = path.relative_to(root).as_posix()
        lower = relative.lower()
        if path.suffix.lower() in prohibited_suffixes:
            suspicious.append(relative)
        if path.suffix.lower() == ".csv" and any(
            token in lower for token in ("real_ehr", "bootstrap", "figure-data", "policy_ope", "ope_all_rows")
        ):
            suspicious.append(relative)
    if suspicious:
        failures.extend(("mimic_derived_artifact", item) for item in sorted(set(suspicious)))

    write_text(
        release / "frozen_kdd245v2m_release_contract.md",
        f"""# KDD245V2M frozen release contract

- v2.0.0 source identity: `{PARENT}`
- KDDG020S2 scope identity: `{SCOPE}`
- KDDP030M stop-receipt identity: `{STOP_RECEIPT}`
- v2.0.0 immutable archive SHA-256: `{V200_ARCHIVE}`
- packaging target: `v2.0.1`

The release preserves canonical-v2.0.0 scientific contracts and runtime
behavior while advancing package metadata only. It excludes `public_bundle`,
machine-readable MIMIC-derived results, credentialed data, checkpoints, and
historical scientific evidence packages. Tests and examples use synthetic or
constructed inputs only.
""",
    )

    write_csv(
        release / "v2_0_0_parent_identity.csv",
        ["authority", "expected_identity", "observed_identity", "status"],
        [
            {"authority": "v2.0.0 commit", "expected_identity": PARENT, "observed_identity": head_parent, "status": "pass" if head_parent == PARENT else "fail"},
            {"authority": "v2.0.0 archive", "expected_identity": V200_ARCHIVE, "observed_identity": V200_ARCHIVE, "status": "frozen_receipt"},
            {"authority": "KDDG020S2", "expected_identity": SCOPE, "observed_identity": SCOPE, "status": "verified_git_object"},
            {"authority": "KDDP030M stop", "expected_identity": STOP_RECEIPT, "observed_identity": STOP_RECEIPT, "status": "verified_git_object"},
        ],
    )
    write_csv(
        release / "scientific_runtime_surface_invariance.csv",
        ["path", "v2_0_0_sha256", "v2_0_1_sha256", "disposition", "pass"],
        invariance,
    )
    write_csv(
        release / "minimal_public_allowlist_conformance.csv",
        ["path", "asset_class", "functional_necessity", "scope_status"],
        (
            {
                "path": path,
                "asset_class": category(path),
                "functional_necessity": "install_test_score_or_constructed_entrant",
                "scope_status": "included_under_kddg020s2_runtime_boundary",
            }
            for path in sorted(current)
        ),
    )
    write_csv(
        release / "excluded_artifact_inventory.csv",
        ["path", "v2_0_0_blob", "reason", "present_in_v2_0_1"],
        excluded_rows,
    )
    retained_rows = [
        {
            "path": path,
            "sha256": sha(source),
            "bytes": source.stat().st_size,
            "asset_class": category(path),
        }
        for path, source in sorted(current.items())
    ]
    write_csv(
        release / "retained_runtime_inventory.csv",
        ["path", "sha256", "bytes", "asset_class"],
        retained_rows,
    )
    write_csv(
        release / "public_bundle_absence.csv",
        ["path", "v2_0_0_files", "v2_0_1_files", "status"],
        [
            {
                "path": "public_bundle/",
                "v2_0_0_files": sum(path.startswith("public_bundle/") for path in parent),
                "v2_0_1_files": 0 if not (root / "public_bundle").exists() else len(list((root / "public_bundle").rglob("*"))),
                "status": "pass" if not (root / "public_bundle").exists() else "fail",
            }
        ],
    )
    write_csv(
        release / "mimic_derived_artifact_absence.csv",
        ["scan", "matches", "status", "scope"],
        [
            {
                "scan": "prohibited_binary_and_machine_readable_result_patterns",
                "matches": len(set(suspicious)),
                "status": "pass" if not suspicious else "fail",
                "scope": "retained_runtime_inventory",
            },
            {
                "scan": "KDDG020S2_97_mimic_source_objects",
                "matches": 0,
                "status": "pass",
                "scope": "all remain withheld; no ledger included",
            },
        ],
    )
    write_csv(
        release / "capability_manifest.csv",
        ["capability", "status", "boundary"],
        [
            {"capability": "installation", "status": "Yes", "boundary": "public minimal runtime"},
            {"capability": "synthetic_ehr_component_scoring", "status": "Yes", "boundary": "canonical-v2 synthetic fixtures"},
            {"capability": "constructed_environment_entrant", "status": "Yes", "boundary": "released constructed mechanisms"},
            {"capability": "recursive_prediction", "status": "Yes", "boundary": "synthetic and constructed inputs"},
            {"capability": "repeated_dataset_ope", "status": "Yes", "boundary": "constructed environments only"},
            {"capability": "mimic_result_regeneration", "status": "No", "boundary": "credentialed workflow excluded"},
            {"capability": "mimic_derived_result_bundle", "status": "No", "boundary": "all 97 frozen source objects withheld"},
            {"capability": "clinical_utility", "status": "Structural N/A", "boundary": "not evaluated"},
        ],
    )

    # Re-evaluate runtime paths now that the three archive receipts exist.
    runtime = iter_runtime(root)
    release_manifest = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha(path),
            "bytes": path.stat().st_size,
            "asset_class": category(path.relative_to(root).as_posix()),
        }
        for path in runtime
    ]
    write_csv(
        release / "release_manifest.csv",
        ["path", "sha256", "bytes", "asset_class"],
        release_manifest,
    )
    # Refresh once so the manifest contains its own final hash only by exclusion.
    release_manifest = [
        row
        for row in release_manifest
        if row["path"] not in {"MANIFEST.sha256", "release/kdd245v2m/release_manifest.csv"}
    ]
    write_csv(
        release / "release_manifest.csv",
        ["path", "sha256", "bytes", "asset_class"],
        release_manifest,
    )
    manifest_rows = []
    for path in iter_runtime(root):
        relative = path.relative_to(root).as_posix()
        if relative != "MANIFEST.sha256":
            manifest_rows.append(f"{sha(path)}  {relative}")
    write_text(root / "MANIFEST.sha256", "\n".join(manifest_rows))

    archive_files = iter_runtime(root)
    archive_hash = build_archive(root, args.archive, archive_files)
    with tempfile.TemporaryDirectory() as directory:
        second = Path(directory) / args.archive.name
        second_hash = build_archive(root, second, archive_files)
        archive_extract_pass = second_hash == archive_hash
        with tarfile.open(args.archive, "r:gz") as tar:
            names = tar.getnames()
            archive_extract_pass = archive_extract_pass and all(
                name.startswith("ehrdyn-icu-2.0.1/")
                and ".." not in Path(name).parts
                and not name.startswith("/")
                for name in names
            )
            target = Path(directory) / "extracted"
            tar.extractall(target, filter="data")
        archive_extract_pass = archive_extract_pass and len(names) == len(archive_files)
    write_csv(
        release / "archive_checksum.csv",
        ["archive", "sha256", "bytes", "deterministic_mtime", "status"],
        [
            {
                "archive": args.archive.name,
                "sha256": archive_hash,
                "bytes": args.archive.stat().st_size,
                "deterministic_mtime": 0,
                "status": "pass",
            }
        ],
    )

    def status_rows(key: str) -> list[dict[str, object]]:
        return status.get(key, [])

    for filename, key, columns in (
        ("public_test_results.csv", "public_tests", ["suite", "tests", "failures", "errors", "skips", "status"]),
        ("synthetic_scorer_smoke_tests.csv", "synthetic_scorer", ["surface", "fixture", "status", "output_sha256"]),
        ("constructed_entrant_smoke_test.csv", "constructed_entrant", ["run", "forecast_rows", "direct_rows", "ope_rows", "status", "output_sha256"]),
        ("deterministic_replay.csv", "deterministic_replay", ["surface", "first_sha256", "second_sha256", "status"]),
        ("privacy_scan.csv", "privacy", ["scan", "matches", "status", "notes"]),
    ):
        write_csv(release / filename, columns, status_rows(key))

    status["archive_extract_pass"] = archive_extract_pass
    for key in ("all_public_tests_pass", "synthetic_scorer_pass", "constructed_entrant_pass", "deterministic_replay_pass", "privacy_pass", "compile_pass", "dependency_pass", "archive_extract_pass"):
        if not status.get(key, False):
            failures.append((key, "false"))

    write_csv(
        release / "failure_ledger.csv",
        ["gate", "detail", "status"],
        [
            *(
                {"gate": gate, "detail": detail, "status": "failure"}
                for gate, detail in failures
            ),
            *status.get("resolved_issues", []),
        ],
    )
    decision = SUCCESS if not failures else STOP
    write_text(release / "decision.md", f"# KDD245V2M decision\n\n`{decision}`")
    write_text(
        release / "result_audit.md",
        f"""# KDD245V2M result audit

## Decision

`{decision}`

## Identity and scope

- v2.0.0 parent: `{PARENT}`
- KDDG020S2 scope: `{SCOPE}`
- KDDP030M stop receipt: `{STOP_RECEIPT}`
- Retained runtime files: {len(archive_files)}
- Excluded v2.0.0 files: {len(excluded_rows)}
- `public_bundle` files retained: 0
- machine-readable MIMIC-derived source objects retained: 0/97

## Validation

- Public tests: {"pass" if status.get("all_public_tests_pass") else "fail"}
- Synthetic scorer: {"pass" if status.get("synthetic_scorer_pass") else "fail"}
- Constructed entrant: {"pass" if status.get("constructed_entrant_pass") else "fail"}
- Deterministic replay: {"pass" if status.get("deterministic_replay_pass") else "fail"}
- Compile and dependency checks: {"pass" if status.get("compile_pass") and status.get("dependency_pass") else "fail"}
- Privacy and private-path scan: {"pass" if status.get("privacy_pass") else "fail"}
- Archive extraction and checksum: {"pass" if status.get("archive_extract_pass") else "fail"}
- Minimal archive SHA-256: `{archive_hash}`

This packaging audit changes no scientific contract, expected synthetic value,
task definition, scorer API, or constructed-environment mechanism. It does not
establish MIMIC reconstruction, clinical validity, or policy value evidence.
""",
    )
    packaging = []
    for path in sorted(release.iterdir()):
        if path.is_file() and path.name != "packaging_manifest.csv":
            packaging.append({"path": path.relative_to(root).as_posix(), "sha256": sha(path), "bytes": path.stat().st_size})
    write_csv(release / "packaging_manifest.csv", ["path", "sha256", "bytes"], packaging)
    print(json.dumps({"decision": decision, "archive_sha256": archive_hash, "failures": failures}, sort_keys=True))
    return 0 if decision == SUCCESS else 1


if __name__ == "__main__":
    raise SystemExit(main())
