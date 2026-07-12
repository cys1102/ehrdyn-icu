from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from . import BENCHMARK_VERSION
from .baseline import BASELINES, run_baseline
from .config import validate_config_directory, validate_task_config
from .errors import ReleaseContractError
from .evaluator import evaluate_fixture
from .fixture import generate_fixture
from .privacy import scan_release, verify_checksums
from .report import write_aggregate_report
from .split import deterministic_split
from .submission import validate_submission

JSON_INDENT: Final = 2


@dataclass(slots=True)
class CliArgs(argparse.Namespace):
    command: str = ""
    output: Path = Path()
    episodes: int = 8
    seed: int = 3408
    config: Path | None = None
    config_dir: Path | None = None
    fixture: Path = Path()
    task_config: Path = Path()
    baseline: str = ""
    output_fixture: Path = Path()
    input_dir: Path = Path()
    entity_key: str = ""
    root: Path = Path()
    submission: Path = Path()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KDD 2027 frozen benchmark clean-room utilities.")
    _ = parser.add_argument("--version", action="version", version=BENCHMARK_VERSION)
    commands = parser.add_subparsers(dest="command", required=True)
    fixture = commands.add_parser("generate-fixture", help="Generate a deterministic synthetic fixture.")
    _ = fixture.add_argument("--output", type=Path, required=True)
    _ = fixture.add_argument("--episodes", type=int, default=8)
    _ = fixture.add_argument("--seed", type=int, default=3408)
    validate = commands.add_parser("validate-config", help="Validate one task config or the frozen config directory.")
    _ = validate.add_argument("--config", type=Path)
    _ = validate.add_argument("--config-dir", type=Path)
    evaluate = commands.add_parser("evaluate", help="Evaluate schema-compatible synthetic predictions.")
    _ = evaluate.add_argument("--fixture", type=Path, required=True)
    _ = evaluate.add_argument("--task-config", type=Path, required=True)
    _ = evaluate.add_argument("--output", type=Path, required=True)
    baseline = commands.add_parser("baseline", help="Run a deterministic synthetic forecasting baseline.")
    _ = baseline.add_argument("--fixture", type=Path, required=True)
    _ = baseline.add_argument("--task-config", type=Path, required=True)
    _ = baseline.add_argument("--baseline", choices=sorted(BASELINES), required=True)
    _ = baseline.add_argument("--output-fixture", type=Path, required=True)
    _ = baseline.add_argument("--output", type=Path, required=True)
    report = commands.add_parser("aggregate-report", help="Render aggregate metric JSON files as Markdown.")
    _ = report.add_argument("--input-dir", type=Path, required=True)
    _ = report.add_argument("--output", type=Path, required=True)
    split = commands.add_parser("split", help="Assign an opaque local entity key to a deterministic split.")
    _ = split.add_argument("--entity-key", required=True)
    scan = commands.add_parser("scan-release", help="Fail closed on restricted release artifacts.")
    _ = scan.add_argument("--root", type=Path, required=True)
    checksums = commands.add_parser("verify-checksums", help="Verify the frozen public artifact manifest.")
    _ = checksums.add_argument("--root", type=Path, required=True)
    submission = commands.add_parser("validate-submission", help="Validate an aggregate leaderboard submission.")
    _ = submission.add_argument("--submission", type=Path, required=True)
    _ = submission.add_argument("--config-dir", type=Path, required=True)
    return parser


def parse_args() -> CliArgs:
    return build_parser().parse_args(namespace=CliArgs())


def main() -> int:
    args = parse_args()
    try:
        return _dispatch(args)
    except ReleaseContractError as error:
        print(f"KDD2027 contract error: {error}")
        return 2


def _dispatch(args: CliArgs) -> int:
    if args.command == "generate-fixture":
        print(json.dumps({"rows": generate_fixture(args.output, args.episodes, args.seed), "synthetic": True}))
    elif args.command == "validate-config":
        if bool(args.config) == bool(args.config_dir):
            raise ReleaseContractError("Choose exactly one of --config or --config-dir")
        count = 1 if args.config is not None else len(validate_config_directory(_required_path(args.config_dir)))
        if args.config is not None:
            _ = validate_task_config(args.config)
        print(json.dumps({"valid_configs": count, "benchmark_version": BENCHMARK_VERSION}))
    elif args.command == "evaluate":
        config = validate_task_config(args.task_config)
        _write_json(args.output, evaluate_fixture(args.fixture, str(config["task_id"])))
    elif args.command == "baseline":
        config = validate_task_config(args.task_config)
        _write_json(args.output, run_baseline(args.fixture, args.output_fixture, str(config["task_id"]), args.baseline))
    elif args.command == "aggregate-report":
        print(json.dumps({"aggregate_records": write_aggregate_report(args.input_dir, args.output)}))
    elif args.command == "split":
        print(json.dumps({"split": deterministic_split(args.entity_key)}))
    elif args.command == "scan-release":
        print(json.dumps(scan_release(args.root), sort_keys=True))
    elif args.command == "verify-checksums":
        print(json.dumps(verify_checksums(args.root), sort_keys=True))
    elif args.command == "validate-submission":
        configs = validate_config_directory(args.config_dir)
        task_ids = {str(config["task_id"]) for config in configs}
        print(json.dumps(validate_submission(args.submission, task_ids), sort_keys=True))
    return 0


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(json.dumps(value, indent=JSON_INDENT, sort_keys=True) + "\n", encoding="utf-8")


def _required_path(path: Path | None) -> Path:
    if path is None:
        raise ReleaseContractError("Required path was not provided")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
