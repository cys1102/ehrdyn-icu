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
from .evaluator import evaluate_fixture, evaluate_predictions
from .fixture import generate_fixture
from .manifest import validate_paper_manifests
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
    predictions: Path = Path()
    episode_key_column: str = "local_episode_key"
    task_manifest: Path = Path()
    contract_manifest: Path = Path()
    evidence: Path = Path()
    normalization: Path = Path()
    normalization_output: Path = Path()
    evaluation_contract: Path = Path()
    contract_output: Path = Path()
    subjects: int = 4
    transitions: int = 3
    cluster_key_column: str = "local_subject_key"
    sequence_key_column: str = "local_sequence_key"
    bootstrap_replicates: int = 1000
    backend_root: Path = Path()
    evidence_manifest: Path = Path()
    dist_dir: Path = Path()


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
    rv_validate = commands.add_parser("rv-validate-config", help="Validate the isolated five-task successor contract.")
    _ = rv_validate.add_argument("--config-dir", type=Path, required=True)
    _ = rv_validate.add_argument("--contract-manifest", type=Path, required=True)
    rv_split = commands.add_parser("rv-split", help="Assign a local decimal entity key using the frozen successor hash.")
    _ = rv_split.add_argument("--entity-key", required=True)
    rv_fixture = commands.add_parser("rv-generate-fixture", help="Generate successor synthetic predictions and train-normalization receipt.")
    _ = rv_fixture.add_argument("--output", type=Path, required=True)
    _ = rv_fixture.add_argument("--normalization-output", type=Path, required=True)
    _ = rv_fixture.add_argument("--contract-output", type=Path, required=True)
    _ = rv_fixture.add_argument("--subjects", type=int, default=4)
    _ = rv_fixture.add_argument("--transitions", type=int, default=3)
    _ = rv_fixture.add_argument("--seed", type=int, default=3408)
    rv_evaluate_fixture = commands.add_parser("rv-evaluate-fixture", help="Evaluate successor synthetic predictions.")
    _ = rv_evaluate_fixture.add_argument("--fixture", type=Path, required=True)
    _ = rv_evaluate_fixture.add_argument("--normalization", type=Path, required=True)
    _ = rv_evaluate_fixture.add_argument("--evaluation-contract", type=Path, required=True)
    _ = rv_evaluate_fixture.add_argument("--output", type=Path, required=True)
    _ = rv_evaluate_fixture.add_argument("--bootstrap-replicates", type=int, default=1000)
    rv_evaluate_local = commands.add_parser(
        "rv-evaluate-local",
        help="Aggregate successor sealed-test predictions without exporting local keys.",
    )
    _ = rv_evaluate_local.add_argument("--predictions", type=Path, required=True)
    _ = rv_evaluate_local.add_argument("--normalization", type=Path, required=True)
    _ = rv_evaluate_local.add_argument("--evaluation-contract", type=Path, required=True)
    _ = rv_evaluate_local.add_argument("--output", type=Path, required=True)
    _ = rv_evaluate_local.add_argument("--cluster-key-column", default="local_subject_key")
    _ = rv_evaluate_local.add_argument("--sequence-key-column", default="local_sequence_key")
    _ = rv_evaluate_local.add_argument("--bootstrap-replicates", type=int, default=1000)
    rv_submission = commands.add_parser(
        "rv-validate-submission",
        help="Validate an evaluator-produced successor aggregate receipt.",
    )
    _ = rv_submission.add_argument("--submission", type=Path, required=True)
    _ = rv_submission.add_argument("--config-dir", type=Path, required=True)
    rv_source = commands.add_parser("rv-verify-source", help="Verify an exact frozen world-ehr RV backend checkout.")
    _ = rv_source.add_argument("--backend-root", type=Path, required=True)
    rv_evidence = commands.add_parser("rv-verify-evidence", help="Verify packaged successor aggregate evidence and privacy receipts.")
    _ = rv_evidence.add_argument("--root", type=Path, required=True)
    _ = rv_evidence.add_argument("--manifest", dest="evidence_manifest", type=Path, required=True)
    rv_distributions = commands.add_parser(
        "rv-audit-distributions",
        help="Hash and privacy-scan built successor wheel/source archives.",
    )
    _ = rv_distributions.add_argument("--dist-dir", type=Path, required=True)
    decision_validate = commands.add_parser(
        "decision-validate",
        help="Validate the decision-benchmark contracts, aggregate evidence, and exact dispositions.",
    )
    _ = decision_validate.add_argument("--root", type=Path, required=True)
    decision_smoke = commands.add_parser(
        "decision-smoke",
        help="Run the unrestricted synthetic known-value, planner, CRN, and OPE smoke test.",
    )
    _ = decision_smoke.add_argument("--output", type=Path, required=True)
    _ = decision_smoke.add_argument("--seed", type=int, default=3408)
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
        print(json.dumps({"aggregate_records": write_aggregate_report(args.input_dir, args.output)}))
    elif args.command == "split":
        print(json.dumps({"split": deterministic_split(args.entity_key)}))
    elif args.command == "scan-release":
        print(json.dumps(scan_release(args.root), sort_keys=True))
    elif args.command == "verify-checksums":
        print(json.dumps(verify_checksums(args.root), sort_keys=True))
    elif args.command == "validate-submission":
        configs = validate_config_directory(_required_path(args.config_dir))
        print(json.dumps(validate_submission(args.submission, configs), sort_keys=True))
    elif args.command == "validate-manifest":
        print(
            json.dumps(
                validate_paper_manifests(args.task_manifest, args.contract_manifest, args.evidence),
                sort_keys=True,
            )
        )
    elif args.command == "rv-validate-config":
        from .rv.contract import validate_config_directory as validate_rv_config_directory
        from .rv.contract import validate_contract_manifest

        configs = validate_rv_config_directory(_required_path(args.config_dir))
        _ = validate_contract_manifest(args.contract_manifest)
        print(json.dumps({"valid_configs": len(configs), "successor": True}, sort_keys=True))
    elif args.command == "rv-split":
        from .rv.split import subject_role as rv_subject_role

        print(json.dumps({"role": rv_subject_role(args.entity_key)}, sort_keys=True))
    elif args.command == "rv-generate-fixture":
        from .rv.fixture import generate_fixture as generate_rv_fixture

        print(
            json.dumps(
                generate_rv_fixture(
                    args.output,
                    args.normalization_output,
                    args.contract_output,
                    subjects=args.subjects,
                    transitions=args.transitions,
                    seed=args.seed,
                ),
                sort_keys=True,
            )
        )
    elif args.command == "rv-evaluate-fixture":
        from .rv.evaluator import evaluate_predictions as evaluate_rv_predictions

        _write_json(
            args.output,
            evaluate_rv_predictions(
                args.fixture,
                args.normalization,
                args.evaluation_contract,
                cluster_key_column="synthetic_subject_key",
                sequence_key_column="synthetic_sequence_key",
                synthetic=True,
                bootstrap_replicates=args.bootstrap_replicates,
            ),
        )
    elif args.command == "rv-evaluate-local":
        from .rv.evaluator import evaluate_predictions as evaluate_rv_predictions

        _write_json(
            args.output,
            evaluate_rv_predictions(
                args.predictions,
                args.normalization,
                args.evaluation_contract,
                cluster_key_column=args.cluster_key_column,
                sequence_key_column=args.sequence_key_column,
                bootstrap_replicates=args.bootstrap_replicates,
            ),
        )
    elif args.command == "rv-validate-submission":
        from .rv.contract import validate_config_directory as validate_rv_config_directory
        from .rv.submission import validate_submission as validate_rv_submission

        configs = validate_rv_config_directory(_required_path(args.config_dir))
        print(json.dumps(validate_rv_submission(args.submission, configs), sort_keys=True))
    elif args.command == "rv-verify-source":
        from .rv.source import verify_backend_source

        print(json.dumps(verify_backend_source(args.backend_root), sort_keys=True))
    elif args.command == "rv-verify-evidence":
        from .rv.evidence import verify_evidence

        print(json.dumps(verify_evidence(args.root, args.evidence_manifest), sort_keys=True))
    elif args.command == "rv-audit-distributions":
        from .privacy import audit_distribution_archives

        print(json.dumps(audit_distribution_archives(args.dist_dir), sort_keys=True))
    elif args.command == "decision-validate":
        from .decision import validate_decision_release

        print(json.dumps(validate_decision_release(args.root), sort_keys=True))
    elif args.command == "decision-smoke":
        from .decision import run_smoke

        result = run_smoke(seed=args.seed)
        _write_json(args.output, result)
        print(json.dumps(result, sort_keys=True))
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
