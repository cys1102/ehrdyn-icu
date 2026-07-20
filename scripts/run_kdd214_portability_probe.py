#!/usr/bin/env python3
"""Generate the frozen cross-Python aggregate serialization probe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from kdd2027_benchmark.canonical import canonical_bytes, write_canonical_json  # noqa: E402
from kdd2027_benchmark.evaluator import evaluate_fixture  # noqa: E402
from kdd2027_benchmark.fixture import generate_fixture  # noqa: E402
from kdd2027_benchmark.public_ope import run_public_ope_smoke  # noqa: E402
from kdd2027_benchmark.public_pomdp import run_public_pomdp_smoke  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    fixture = args.output / "synthetic_fixture.csv"
    generate_fixture(fixture, episodes=3, seed=21403)
    config = ROOT / "configs/public_pomdp/kdd198_repaired_v2.json"
    payload = {
        "aggregate_metrics": evaluate_fixture(fixture, "kdd2027_sepsis_vasopressor_3bin"),
        "ope_smoke": run_public_ope_smoke(config, "aki", 21401, 2, 16, 2, 21411),
        "pomdp_smoke": run_public_pomdp_smoke(config, "aki", 21401, 16, 21407),
        "probe_contract": {
            "canonical_decimal_places": 12,
            "schema_release": "kdd214",
        },
    }
    write_canonical_json(args.output / "computed_smoke.json", payload)
    write_canonical_json(args.output / "unrounded_float_repr.json", _float_reprs(payload))
    sys.stdout.buffer.write(canonical_bytes({
        "computed": "computed_smoke.json",
        "pass": True,
        "python_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
    }))
    return 0


def _float_reprs(value: object, path: tuple[object, ...] = ()) -> dict[str, str]:
    rows: dict[str, str] = {}
    if isinstance(value, float):
        rows[_pointer(path)] = repr(value)
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            rows.update(_float_reprs(item, path + (index,)))
    elif isinstance(value, dict):
        for key, item in value.items():
            rows.update(_float_reprs(item, path + (key,)))
    return rows


def _pointer(parts: tuple[object, ...]) -> str:
    return "/" + "/".join(str(part).replace("~", "~0").replace("/", "~1") for part in parts)


if __name__ == "__main__":
    raise SystemExit(main())
