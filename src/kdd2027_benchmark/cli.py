from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from . import BENCHMARK_VERSION
from .baseline import BASELINES, run_baseline
from .canonical import canonical_bytes, write_canonical_json
from .config import validate_config_directory, validate_task_config
from .errors import ReleaseContractError
from .evaluator import evaluate_fixture, evaluate_predictions
from .fixture import generate_fixture
from .manifest import validate_paper_manifests
from .privacy import scan_release, verify_checksums
from .report import write_aggregate_report
from .split import deterministic_split
from .schema import validate_schema_directory
from .submission import validate_submission
from .public_bundle import rebuild_public_bundle
from .public_ope import run_public_ope_smoke
from .public_pomdp import run_public_pomdp_smoke
from .transition_entrant import validate_transition_submission

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
    predictions: Path = Path()
    episode_key_column: str = "local_episode_key"
    task_manifest: Path = Path()
    contract_manifest: Path = Path()
    evidence: Path = Path()
    profile: str = "aki"
    environment_seed: int = 21201
    datasets: int = 4
    bootstrap: int = 8
    bundle: Path = Path()
    schema_dir: Path = Path()


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
    evaluate_local = commands.add_parser(
        "evaluate-local",
        help="Aggregate a credentialed local prediction file without exporting row keys.",
    )
    _ = evaluate_local.add_argument("--predictions", type=Path, required=True)
    _ = evaluate_local.add_argument("--episode-key-column", default="local_episode_key")
    _ = evaluate_local.add_argument("--task-config", type=Path, required=True)
    _ = evaluate_local.add_argument("--output", type=Path, required=True)
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
    schemas = commands.add_parser("validate-schemas", help="Validate every released JSON Schema as Draft 2020-12.")
    _ = schemas.add_argument("--schema-dir", type=Path, required=True)
    submission = commands.add_parser("validate-submission", help="Validate an aggregate leaderboard submission.")
    _ = submission.add_argument("--submission", type=Path, required=True)
    _ = submission.add_argument("--config-dir", type=Path, required=True)
    manifest = commands.add_parser(
        "validate-manifest",
        help="Validate paper task and contract mappings against public evidence.",
    )
    _ = manifest.add_argument("--task-manifest", type=Path, required=True)
    _ = manifest.add_argument("--contract-manifest", type=Path, required=True)
    _ = manifest.add_argument("--evidence", type=Path, required=True)
    pomdp = commands.add_parser("pomdp-smoke", help="Run the repaired public constructed-POMDP entrant smoke.")
    _ = pomdp.add_argument("--config", type=Path, required=True)
    _ = pomdp.add_argument("--profile", required=True)
    _ = pomdp.add_argument("--environment-seed", type=int, default=21201)
    _ = pomdp.add_argument("--episodes", type=int, default=64)
    _ = pomdp.add_argument("--seed", type=int, default=3408)
    _ = pomdp.add_argument("--output", type=Path, required=True)
    ope = commands.add_parser("ope-smoke", help="Run the bounded KDD202B-compatible repeated-dataset OPE smoke.")
    _ = ope.add_argument("--config", type=Path, required=True)
    _ = ope.add_argument("--profile", required=True)
    _ = ope.add_argument("--environment-seed", type=int, default=21201)
    _ = ope.add_argument("--datasets", type=int, default=4)
    _ = ope.add_argument("--episodes", type=int, default=64)
    _ = ope.add_argument("--bootstrap", type=int, default=8)
    _ = ope.add_argument("--seed", type=int, default=3411)
    _ = ope.add_argument("--output", type=Path, required=True)
    transition = commands.add_parser("validate-transition-submission", help="Validate aggregate transition rows and task/version hashes.")
    _ = transition.add_argument("--submission", type=Path, required=True)
    _ = transition.add_argument("--config-dir", type=Path, required=True)
    rebuild = commands.add_parser("rebuild-public-bundle", help="Deterministically materialize public manuscript tables and figures.")
    _ = rebuild.add_argument("--bundle", type=Path, required=True)
    _ = rebuild.add_argument("--output", type=Path, required=True)
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
        _print_json({"rows": generate_fixture(args.output, args.episodes, args.seed), "synthetic": True})
    elif args.command == "validate-config":
        if bool(args.config) == bool(args.config_dir):
            raise ReleaseContractError("Choose exactly one of --config or --config-dir")
        count = 1 if args.config is not None else len(validate_config_directory(_required_path(args.config_dir)))
        if args.config is not None:
            _ = validate_task_config(args.config)
        _print_json({"valid_configs": count, "benchmark_version": BENCHMARK_VERSION})
    elif args.command == "evaluate":
        config = validate_task_config(args.task_config)
        _write_json(args.output, evaluate_fixture(args.fixture, str(config["task_id"])))
    elif args.command == "evaluate-local":
        config = validate_task_config(args.task_config)
        _write_json(
            args.output,
            evaluate_predictions(args.predictions, str(config["task_id"]), args.episode_key_column),
        )
    elif args.command == "baseline":
        config = validate_task_config(args.task_config)
        _write_json(args.output, run_baseline(args.fixture, args.output_fixture, str(config["task_id"]), args.baseline))
    elif args.command == "aggregate-report":
        _print_json({"aggregate_records": write_aggregate_report(args.input_dir, args.output)})
    elif args.command == "split":
        _print_json({"split": deterministic_split(args.entity_key)})
    elif args.command == "scan-release":
        _print_json(scan_release(args.root))
    elif args.command == "verify-checksums":
        _print_json(verify_checksums(args.root))
    elif args.command == "validate-schemas":
        _print_json({"schemas": validate_schema_directory(args.schema_dir), "pass": True})
    elif args.command == "validate-submission":
        _print_json(validate_submission(args.submission, _required_path(args.config_dir)))
    elif args.command == "validate-manifest":
        _print_json(validate_paper_manifests(args.task_manifest, args.contract_manifest, args.evidence))
    elif args.command == "pomdp-smoke":
        _write_json(args.output, run_public_pomdp_smoke(_required_path(args.config), args.profile, args.environment_seed, args.episodes, args.seed))
    elif args.command == "ope-smoke":
        _write_json(args.output, run_public_ope_smoke(_required_path(args.config), args.profile, args.environment_seed, args.datasets, args.episodes, args.bootstrap, args.seed))
    elif args.command == "validate-transition-submission":
        _print_json(validate_transition_submission(args.submission, _required_path(args.config_dir)))
    elif args.command == "rebuild-public-bundle":
        _print_json(rebuild_public_bundle(args.bundle, args.output))
    return 0


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    write_canonical_json(path, value)


def _print_json(value: object) -> None:
    sys.stdout.buffer.write(canonical_bytes(value))


def _required_path(path: Path | None) -> Path:
    if path is None:
        raise ReleaseContractError("Required path was not provided")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
