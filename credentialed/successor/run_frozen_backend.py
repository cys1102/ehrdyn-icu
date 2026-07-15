#!/usr/bin/env python3
"""Verify, then optionally invoke the exact frozen RV01R/RV02R backend.

The default verify-only path reads source code only. Actual construction,
training, or sealed evaluation requires an explicit stage and config.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from kdd2027_benchmark.errors import ReleaseContractError  # noqa: E402
from kdd2027_benchmark.rv.source import verify_backend_source  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify or invoke the frozen KDD-RV successor backend.")
    parser.add_argument("--backend-root", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--stage", choices=("construction", "train", "evaluate"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--test-opening-receipt", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        receipt = verify_backend_source(args.backend_root)
        if args.verify_only:
            print(json.dumps({**receipt, "execution": "not_requested"}, sort_keys=True))
            return 0
        if args.stage is None or args.config is None:
            raise ReleaseContractError("Execution requires both --stage and --config")
        if args.stage == "evaluate" and args.test_opening_receipt is None:
            raise ReleaseContractError("Sealed evaluation requires --test-opening-receipt")
        if args.stage != "evaluate" and args.test_opening_receipt is not None:
            raise ReleaseContractError("The test-opening receipt is valid only for sealed evaluation")
        module = {
            "construction": "kdd_rv01r.run",
            "train": "kdd_rv02r.train",
            "evaluate": "kdd_rv02r.evaluate",
        }[args.stage]
        command = [sys.executable, "-m", module, "--config", str(args.config)]
        if args.test_opening_receipt is not None:
            command.extend(("--test-opening-receipt", str(args.test_opening_receipt)))
        environment = dict(os.environ)
        current_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = str(args.backend_root) + (os.pathsep + current_pythonpath if current_pythonpath else "")
        completed = subprocess.run(command, cwd=args.backend_root, env=environment, check=False)
        return completed.returncode
    except ReleaseContractError as error:
        print(f"KDD-RV successor contract error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
