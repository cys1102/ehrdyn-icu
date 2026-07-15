"""Validate packaged aggregate successor evidence and privacy receipts."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from ..errors import ReleaseContractError


def verify_evidence(root: Path, manifest: Path) -> dict[str, object]:
    try:
        with manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise ReleaseContractError(f"Cannot read successor evidence manifest: {error}") from error
    if not rows:
        raise ReleaseContractError("Successor evidence manifest is empty")
    root_resolved = root.resolve()
    privacy_receipts: set[str] = set()
    experiments: set[str] = set()
    for row in rows:
        if row.get("aggregate_or_contract_only") != "True":
            raise ReleaseContractError(f"Non-aggregate successor evidence entry: {row.get('source_artifact', '')}")
        path = (root / row["packaged_path"]).resolve()
        if path.parent != root_resolved and root_resolved not in path.parents:
            raise ReleaseContractError("Successor evidence path escapes release root")
        if not path.is_file():
            raise ReleaseContractError(f"Successor evidence is missing: {row['packaged_path']}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ReleaseContractError(f"Successor evidence hash mismatch: {row['packaged_path']}")
        if path.stat().st_size != int(row["bytes"]):
            raise ReleaseContractError(f"Successor evidence byte-count mismatch: {row['packaged_path']}")
        privacy_receipts.add(row["privacy_receipt"])
        experiments.add(row["experiment_id"])
    for relative in privacy_receipts:
        path = root / relative
        with path.open(newline="", encoding="utf-8") as handle:
            checks = list(csv.DictReader(handle))
        if not checks or any(row.get("status") != "pass" for row in checks):
            raise ReleaseContractError(f"Successor evidence privacy receipt failed: {relative}")
    return {
        "evidence_files_verified": len(rows),
        "experiments": sorted(experiments),
        "privacy_receipts_verified": len(privacy_receipts),
        "hash_mismatches": 0,
        "pass": True,
    }
