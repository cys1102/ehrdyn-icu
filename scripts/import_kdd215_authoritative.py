#!/usr/bin/env python3
"""Import the aggregate-safe KDD215 simulator sources with fixed hash gates.

This developer utility takes an explicit world-ehr checkout.  The public tree
never records that local path; only verified source bytes and the aggregate
generator contract are copied.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


EXPECTED = {
    "configs/kdd198_pomdp_specification_audit_v1.json": "8b072d3001728a2a246b459d6182bcb720b1ee1308662d4041f9d700b3454aa4",
    "configs/kdd199_direct_return_precision_refresh_v1.json": "6fffd82596958b4c13c951256e60caa01fb3588b67a867e3870b862acb83e99a",
    "configs/kdd202b_policy_by_estimator_accuracy_v1.json": "d0ae420384358604ae336fdfbcc25f56fb5d2e96132e349ce66571727fddf14d",
    "kdd_benchmark_discovery/kdd198_pomdp_v2.py": "257b540db70e83bd521ad3111ce0f9ba992228672be267e7456b22063f6814bf",
    "kdd_benchmark_discovery/kdd199_repaired_direct_evaluator.py": "6132f697c8ea784b92728d2b53a3392ea4839da16b866a68587874b2dfb9b0fc",
    "kdd_benchmark_discovery/run_kdd199_direct_return_precision_refresh.py": "caf4ef21948cda7733d1d1d5715eb3390780f267becd80df89ff6e98bc21284e",
    "kdd_benchmark_discovery/run_kdd202b_policy_by_estimator_accuracy.py": "c2c982bf6d251fde0c5307cf20aadaebe04d36b11ace8bf40d778d81eb38c700",
}
PORT_DEPENDENCIES = {
    "kdd_benchmark_discovery/kdd165r_pomdp.py": "f655e7b86b9cade99f42c0d35b0c1a9e618c7bdbc35d9b2d90b2f08d2e851d1c",
    "kdd_benchmark_discovery/kdd165r2_pomdp.py": "25492325298cded1ab9cd5f197cf6c5156c0d9bbc74ba80011442d5e0be1af05",
    "kdd_benchmark_discovery/kdd202b_policy_ope.py": "cc36a1013237cdc9973d096605d6b67cb62ca1a11587d35c7ab2e052f1802d0c",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    failures = []
    for relative, expected in {**EXPECTED, **PORT_DEPENDENCIES}.items():
        actual = digest(args.source / relative)
        if actual != expected:
            failures.append(f"{relative}: {actual} != {expected}")
    if failures:
        raise SystemExit("authoritative source mismatch\n" + "\n".join(failures))

    package = args.destination / "src/kdd2027_benchmark"
    types = (args.source / "kdd_benchmark_discovery/kdd165r_pomdp.py").read_text()
    types = types.replace("import torch\n", "")
    before, remainder = types.split("def calibrate_behavior", 1)
    _discard, after = remainder.split("class ObservationHistoryPolicy", 1)
    types = before + "class ObservationHistoryPolicy" + after
    write(package / "full_pomdp_types.py", types)

    core = (args.source / "kdd_benchmark_discovery/kdd165r2_pomdp.py").read_text()
    core = core.replace("from .kdd165r_pomdp import", "from .full_pomdp_types import")
    write(package / "full_pomdp_core.py", core)

    mechanism = (args.source / "kdd_benchmark_discovery/kdd198_pomdp_v2.py").read_text()
    mechanism = mechanism.replace("from .kdd165r2_pomdp import", "from .full_pomdp_core import")
    write(package / "full_pomdp_v2.py", mechanism)

    evaluator = (args.source / "kdd_benchmark_discovery/kdd199_repaired_direct_evaluator.py").read_text()
    evaluator = evaluator.replace("from typing import Any, Callable", "from dataclasses import dataclass\nfrom typing import Any, Callable")
    evaluator = evaluator.replace(
        "from .kdd166_pomdp_benchmark import SyntheticData\n",
        "@dataclass(frozen=True)\nclass SyntheticData:\n"
        "    observed: np.ndarray\n    masks: np.ndarray\n    deltas: np.ndarray\n"
        "    actions: np.ndarray\n    next_observed: np.ndarray\n"
        "    behavior_probability: np.ndarray\n    rewards: np.ndarray\n"
        "    done: np.ndarray\n    valid: np.ndarray\n    subtypes: np.ndarray\n\n",
    )
    evaluator = evaluator.replace("from .kdd198_pomdp_v2 import", "from .full_pomdp_v2 import")
    write(package / "full_direct_evaluator.py", evaluator)

    ope = (args.source / "kdd_benchmark_discovery/kdd202b_policy_ope.py").read_text()
    ope = ope.replace("from .kdd199_repaired_direct_evaluator import", "from .full_direct_evaluator import")
    ope = ope.replace(
        "from .kdd202a_ope_reference import DENOMINATOR_FLOOR, TIE_TOLERANCE",
        "DENOMINATOR_FLOOR = 1e-12\nTIE_TOLERANCE = 1e-12",
    )
    write(package / "full_ope.py", ope)

    manifest = args.source / "results/kdd198_pomdp_specification_audit_20260719_084254/complete_generator_parameter_and_seed_manifest.json"
    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    if len(manifest_value.get("accepted_old_environments", [])) != 40:
        raise SystemExit("generator contract does not contain 40 environments")
    write(
        args.destination / "configs/full_benchmark/kdd198_v2_generator_contract.json",
        json.dumps(manifest_value, sort_keys=True, separators=(",", ":")) + "\n",
    )
    source_receipt = {
        "authoritative_hashes": EXPECTED,
        "ported_dependency_hashes": PORT_DEPENDENCIES,
        "generator_contract_sha256": digest(manifest),
        "environment_count": 40,
        "patient_level_data_included": False,
    }
    write(
        args.destination / "configs/full_benchmark/authoritative_import_receipt.json",
        json.dumps(source_receipt, sort_keys=True, separators=(",", ":")) + "\n",
    )
    direct_path = args.source / "results/kdd199_direct_return_precision_refresh_20260719_103144/incremental_episode_precision_by_policy_environment.csv"
    with direct_path.open(newline="", encoding="utf-8") as handle:
        direct = list(csv.DictReader(handle))
    selected_methods = {
        "behavior", "behavior_cloning", "persistence_locf_plus_h4_support_only"
    }
    latest: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in direct:
        if row["method"] not in selected_methods:
            continue
        key = (row["profile"], row["environment_seed"], row["method"])
        if key not in latest or int(row["cumulative_episode_count"]) > int(latest[key]["cumulative_episode_count"]):
            latest[key] = row
    reference = args.destination / "configs/full_benchmark/kdd199_authoritative_reference_cells.csv"
    reference.parent.mkdir(parents=True, exist_ok=True)
    fields = ["profile", "environment_seed", "method", "cumulative_episode_count", "mean_return", "standard_error", "generator_return_range"]
    with reference.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        for row in sorted(latest.values(), key=lambda value: (value["profile"], int(value["environment_seed"]), value["method"])):
            writer.writerow({key: row[key] for key in fields})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
