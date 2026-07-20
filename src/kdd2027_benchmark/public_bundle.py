from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path

from .errors import ReleaseContractError


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rebuild_public_bundle(bundle: Path, output: Path) -> dict[str, object]:
    manifest = bundle / "public_manuscript_aggregate_bundle_manifest.csv"
    if not manifest.is_file():
        raise ReleaseContractError("Public manuscript bundle manifest is missing")
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ReleaseContractError("Public manuscript bundle manifest is empty")
    output.mkdir(parents=True, exist_ok=True)
    receipts = []
    for row in rows:
        source = bundle / row["bundle_path"]
        if not source.is_file() or _sha(source) != row["sha256"]:
            raise ReleaseContractError(f"Public bundle input hash mismatch: {row['artifact_id']}")
        target = output / row["output_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        actual = _sha(target)
        if actual != row["expected_output_sha256"]:
            raise ReleaseContractError(f"Generated public output hash mismatch: {row['artifact_id']}")
        receipts.append((row["artifact_id"], row["output_path"], actual, row["kdd211_parity_status"]))
    receipt = output / "rebuild_manifest.csv"
    with receipt.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["artifact_id", "output_path", "sha256", "kdd211_parity_status"])
        writer.writerows(receipts)
    return {"artifacts_rebuilt": len(receipts), "manifest_sha256": _sha(manifest), "rebuild_manifest_sha256": _sha(receipt), "pass": True}
