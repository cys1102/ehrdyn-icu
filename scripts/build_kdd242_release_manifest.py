#!/usr/bin/env python3
"""Build aggregate-only KDD242 release metadata from frozen public artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KDD235B = ROOT / "release/kdd235b"
KDD242 = ROOT / "release/kdd242"


def rows(name: str) -> list[dict[str, str]]:
    with (KDD235B / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    artifacts = {
        "checkpoint_inventory": KDD235B / "clean_clone_and_external_package_identity.csv",
        "forecast_horizon_metrics": KDD235B / "entrant_forecasting_and_uncertainty_summary.csv",
        "direct_return_summary": KDD235B / "entrant_planner_direct_return_summary.csv",
        "repeated_ope_summary": KDD235B / "entrant_repeated_ope_summary.csv",
        "capability_matrix": KDD242 / "capability_matrix.csv",
        "citation_metadata": ROOT / "CITATION.cff",
    }
    counts = {
        "environment_count": len(rows("clean_clone_and_external_package_identity.csv")),
        "forecast_horizon_rows": len(rows("entrant_forecasting_and_uncertainty_summary.csv")),
        "direct_return_rows": len(rows("entrant_planner_direct_return_summary.csv")),
        "ope_summary_rows": len(rows("entrant_repeated_ope_summary.csv")),
    }
    expected = {
        "environment_count": 40,
        "forecast_horizon_rows": 440,
        "direct_return_rows": 40,
        "ope_summary_rows": 240,
    }
    if counts != expected:
        raise SystemExit(f"KDD235B inventory mismatch: {counts}")
    manifest = {
        "schema_version": "kdd242.release.v1",
        "release": "v1.3.0",
        "name": "EHRDyn-ICU v1.3.0 - Recursive Constructed Benchmark Entrant",
        "public_base_commit": "4739d59392a660cf215c29ebca02fb2f52cd7804",
        "frozen_prerun_commit": "5e6cbd7",
        "counts": counts,
        "artifacts": {
            name: {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}
            for name, path in artifacts.items()
        },
        "leaderboard_status": "demonstration_entrant_not_eligible",
        "claim_boundary": "constructed_benchmark_interface_usability_only",
    }
    KDD242.mkdir(parents=True, exist_ok=True)
    (KDD242 / "release_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksum_paths = [artifacts[name] for name in sorted(artifacts)] + [KDD242 / "release_manifest.json"]
    lines = [f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}" for path in checksum_paths]
    (KDD242 / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
