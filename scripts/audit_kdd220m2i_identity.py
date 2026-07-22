#!/usr/bin/env python3
"""Canonicalize the frozen seven-file credentialed aggregate candidate."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


FILES = (
    "aggregate_receipt.json",
    "icu_time_order_eligibility_aggregate.csv",
    "nullable_key_and_timestamp_exclusion_aggregate.csv",
    "respiratory_action_filter_aggregate.csv",
    "runtime_resource_aggregate.json",
    "stage_resource_instrumentation_aggregate.json",
    "streaming_instrumentation_aggregate.csv",
)
RUNTIME_FIELDS = {
    "wall_seconds", "maximum_resident_set_size_kib", "temporary_disk_bytes",
}
STAGE_RESOURCE_FIELDS = {
    "elapsed_seconds", "temporary_disk_high_water_bytes", "rss_entry_kib",
    "rss_exit_kib", "peak_rss_kib",
}
STAGE_FIELDS = STAGE_RESOURCE_FIELDS | {
    "stage", "rows_read", "rows_retained", "chunk_size",
}
STREAMING_FIELDS = (
    "table", "rows_read", "rows_retained", "chunks_processed",
    "maximum_retained_rows_per_chunk", "effective_chunk_size",
    "compression_encoding", "scan_count",
)
STREAMING_TABLES = (
    "hosp/labevents", "hosp/microbiologyevents", "hosp/prescriptions",
    "icu/chartevents", "icu/inputevents", "icu/outputevents",
    "icu/procedureevents",
)
EXACT_CSV = {
    "icu_time_order_eligibility_aggregate.csv": (
        ("category", "count", "precedence"), ("category",),
    ),
    "nullable_key_and_timestamp_exclusion_aggregate.csv": (
        ("table", "field", "reason", "excluded_count"),
        ("table", "field", "reason"),
    ),
    "respiratory_action_filter_aggregate.csv": (
        ("candidate_transitions", "retained_transitions",
         "excluded_missing_action_transitions", "candidate_episodes",
         "retained_episodes", "excluded_empty_episodes"),
        ("candidate_transitions",),
    ),
}
RESOURCE_SENTINEL = "__KDD220M2I_EXECUTION_RESOURCE__"
ENCODING_SENTINEL = "__KDD220M2I_EQUIVALENT_ENCODING__"
AGGREGATE_SCHEMA_SHA256 = "1f229de694f995b9a500270f3e411dab8bddfee46b15ac038bdb1a21b561f117"
STAGE_SCHEMA_SHA256 = "e5ae77679ccd52e0616f3d0a4d8694e0fac4da8c1a55491042f5bad610573119"


class IdentityError(RuntimeError):
    """Raised when an input lies outside the frozen canonical contract."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IdentityError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_object)
    if not isinstance(value, dict):
        raise IdentityError(f"top-level JSON object required: {path.name}")
    return value


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def validate_schema_file(path: Path, expected_sha256: str) -> dict[str, Any]:
    data = path.read_bytes()
    if sha256_bytes(data) != expected_sha256:
        raise IdentityError(f"schema identity mismatch: {path.name}")
    schema = json.loads(data)
    Draft202012Validator.check_schema(schema)
    return schema


def validate_m2l(release: Path) -> None:
    if (release / "decision.md").read_text(encoding="utf-8").strip() != (
        "complete_single_encoding_view_ready_for_kdd220m2s2"
    ):
        raise IdentityError("KDD220M2L decision does not authorize encoding canonicalization")
    expected_header = (
        "table", "plain_exists", "compressed_exists", "duplicate_stream_equal",
        "selected_encoding", "required_header_valid",
    )
    with (release / "input_layout_preflight.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected_header:
            raise IdentityError("KDD220M2L preflight schema mismatch")
        rows = list(reader)
    if len(rows) != 14 or len({row["table"] for row in rows}) != 14:
        raise IdentityError("KDD220M2L table inventory or uniqueness failure")
    for row in rows:
        if row["duplicate_stream_equal"] != "true" or row["required_header_valid"] != "true":
            raise IdentityError("KDD220M2L did not prove equivalent encoding and header validity")


def inventory(root: Path) -> None:
    names = tuple(sorted(path.name for path in root.iterdir() if path.is_file()))
    if names != tuple(sorted(FILES)):
        raise IdentityError("candidate file inventory mismatch")


def raw_tree(root: Path) -> str:
    inventory(root)
    digest = hashlib.sha256()
    for name in sorted(FILES):
        digest.update(name.encode("utf-8"))
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def _read_csv(path: Path, fields: tuple[str, ...], unique: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != fields:
            raise IdentityError(f"CSV schema mismatch: {path.name}")
        rows = list(reader)
    keys = [tuple(row[field] for field in unique) for row in rows]
    if len(keys) != len(set(keys)):
        raise IdentityError(f"duplicate CSV rows: {path.name}")
    if any(set(row) != set(fields) or None in row for row in rows):
        raise IdentityError(f"CSV additional or missing field: {path.name}")
    return rows


def _write_csv(fields: tuple[str, ...], rows: list[dict[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def canonicalize_bundle(
    root: Path,
    *,
    m2l_release: Path,
    aggregate_schema_path: Path,
    stage_schema_path: Path,
    aggregate_schema_sha256: str = AGGREGATE_SCHEMA_SHA256,
    stage_schema_sha256: str = STAGE_SCHEMA_SHA256,
) -> dict[str, bytes]:
    inventory(root)
    validate_m2l(m2l_release)
    aggregate_schema = validate_schema_file(aggregate_schema_path, aggregate_schema_sha256)
    stage_schema = validate_schema_file(stage_schema_path, stage_schema_sha256)
    output: dict[str, bytes] = {}

    receipt = load_json(root / "aggregate_receipt.json")
    Draft202012Validator(aggregate_schema).validate(receipt)
    receipt = copy.deepcopy(receipt)
    if [row.get("table") for row in receipt["streaming"]] != list(STREAMING_TABLES):
        raise IdentityError("aggregate receipt streaming inventory or order mismatch")
    for row in receipt["streaming"]:
        if set(row) != set(STREAMING_FIELDS):
            raise IdentityError("aggregate receipt streaming field mismatch")
        row["compression_encoding"] = ENCODING_SENTINEL
    output["aggregate_receipt.json"] = canonical_json(receipt)

    runtime = load_json(root / "runtime_resource_aggregate.json")
    if set(runtime) != RUNTIME_FIELDS | {"status"} or runtime["status"] != "complete":
        raise IdentityError("runtime resource field or status mismatch")
    for field in RUNTIME_FIELDS:
        runtime[field] = RESOURCE_SENTINEL
    output["runtime_resource_aggregate.json"] = canonical_json(runtime)

    stage = load_json(root / "stage_resource_instrumentation_aggregate.json")
    Draft202012Validator(stage_schema).validate(stage)
    if set(stage) != {"schema_version", "status", "stages", "privacy"}:
        raise IdentityError("stage instrumentation top-level field mismatch")
    names: list[str] = []
    for row in stage["stages"]:
        if set(row) != STAGE_FIELDS:
            raise IdentityError("stage instrumentation field mismatch")
        names.append(row["stage"])
        for field in STAGE_RESOURCE_FIELDS:
            row[field] = RESOURCE_SENTINEL
    if len(names) != len(set(names)):
        raise IdentityError("duplicate stage rows")
    output["stage_resource_instrumentation_aggregate.json"] = canonical_json(stage)

    streaming = _read_csv(
        root / "streaming_instrumentation_aggregate.csv", STREAMING_FIELDS, ("table",),
    )
    if [row["table"] for row in streaming] != list(STREAMING_TABLES):
        raise IdentityError("streaming CSV inventory or order mismatch")
    for row in streaming:
        row["compression_encoding"] = ENCODING_SENTINEL
    output["streaming_instrumentation_aggregate.csv"] = _write_csv(STREAMING_FIELDS, streaming)

    for name, (fields, unique) in EXACT_CSV.items():
        _read_csv(root / name, fields, unique)
        output[name] = (root / name).read_bytes()
    if set(output) != set(FILES):
        raise IdentityError("canonical output inventory mismatch")
    return output


def canonical_tree(files: dict[str, bytes]) -> str:
    if set(files) != set(FILES):
        raise IdentityError("canonical tree inventory mismatch")
    digest = hashlib.sha256()
    for name in sorted(FILES):
        digest.update(name.encode("utf-8"))
        digest.update(files[name])
    return digest.hexdigest()


def _pointer_escape(value: object) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def json_differences(left: object, right: object, pointer: str = "") -> list[str]:
    if type(left) is not type(right):
        return [pointer or "/"]
    if isinstance(left, dict):
        if set(left) != set(right):
            return [pointer or "/"]
        result: list[str] = []
        for key in sorted(left):
            result.extend(json_differences(left[key], right[key], f"{pointer}/{_pointer_escape(key)}"))
        return result
    if isinstance(left, list):
        if len(left) != len(right):
            return [pointer or "/"]
        result = []
        for index, (one, two) in enumerate(zip(left, right)):
            result.extend(json_differences(one, two, f"{pointer}/{index}"))
        return result
    return [] if left == right else [pointer or "/"]


def classify_differences(left: Path, right: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name in FILES:
        a = (left / name).read_bytes()
        b = (right / name).read_bytes()
        if a == b:
            rows.append({"filename": name, "pointer_or_column": "/", "equality": "equal", "classification": "exact", "justification": "none"})
            continue
        if name in {"runtime_resource_aggregate.json", "stage_resource_instrumentation_aggregate.json", "aggregate_receipt.json"}:
            for pointer in json_differences(load_json(left / name), load_json(right / name)):
                if name == "runtime_resource_aggregate.json" and pointer.removeprefix("/") in RUNTIME_FIELDS:
                    classification, justification = "allowed_telemetry", "frozen_runtime_resource_allowlist"
                elif name == "stage_resource_instrumentation_aggregate.json" and pointer.split("/")[-1] in STAGE_RESOURCE_FIELDS:
                    classification, justification = "allowed_telemetry", "frozen_stage_resource_allowlist"
                elif name == "aggregate_receipt.json" and pointer.startswith("/streaming/") and pointer.endswith("/compression_encoding"):
                    classification, justification = "equivalent_encoding", "KDD220M2L_byte_stream_equality"
                else:
                    classification, justification = "unclassified", "none"
                rows.append({"filename": name, "pointer_or_column": pointer, "equality": "different", "classification": classification, "justification": justification})
        elif name == "streaming_instrumentation_aggregate.csv":
            a_rows = _read_csv(left / name, STREAMING_FIELDS, ("table",))
            b_rows = _read_csv(right / name, STREAMING_FIELDS, ("table",))
            if len(a_rows) != len(b_rows):
                rows.append({"filename": name, "pointer_or_column": "row_inventory", "equality": "different", "classification": "unclassified", "justification": "none"})
                continue
            for index, (one, two) in enumerate(zip(a_rows, b_rows)):
                for field in STREAMING_FIELDS:
                    if one[field] == two[field]:
                        continue
                    allowed = field == "compression_encoding"
                    rows.append({"filename": name, "pointer_or_column": f"row_{index}:{field}", "equality": "different", "classification": "equivalent_encoding" if allowed else "unclassified", "justification": "KDD220M2L_byte_stream_equality" if allowed else "none"})
        else:
            rows.append({"filename": name, "pointer_or_column": "raw_bytes", "equality": "different", "classification": "unclassified", "justification": "none"})
    return rows


def audit(args: argparse.Namespace) -> dict[str, object]:
    left, right = Path(args.m1), Path(args.m2s2)
    left_raw, right_raw = raw_tree(left), raw_tree(right)
    left_files = canonicalize_bundle(left, m2l_release=Path(args.m2l_release), aggregate_schema_path=Path(args.aggregate_schema), stage_schema_path=Path(args.stage_schema))
    right_files = canonicalize_bundle(right, m2l_release=Path(args.m2l_release), aggregate_schema_path=Path(args.aggregate_schema), stage_schema_path=Path(args.stage_schema))
    differences = classify_differences(left, right)
    per_file = []
    for name in FILES:
        left_raw_bytes = (left / name).read_bytes()
        right_raw_bytes = (right / name).read_bytes()
        per_file.append({
            "filename": name,
            "m1_raw_sha256": sha256_bytes(left_raw_bytes),
            "m2s2_raw_sha256": sha256_bytes(right_raw_bytes),
            "m1_bytes": len(left_raw_bytes),
            "m2s2_bytes": len(right_raw_bytes),
            "m1_canonical_sha256": sha256_bytes(left_files[name]),
            "m2s2_canonical_sha256": sha256_bytes(right_files[name]),
        })
    result = {
        "m1_raw_tree_sha256": left_raw,
        "m2s2_raw_tree_sha256": right_raw,
        "m1_canonical_tree_sha256": canonical_tree(left_files),
        "m2s2_canonical_tree_sha256": canonical_tree(right_files),
        "differences": differences,
        "per_file": per_file,
        "difference_count": sum(row["equality"] == "different" for row in differences),
        "unclassified_count": sum(row["classification"] == "unclassified" for row in differences),
    }
    result["canonical_exact"] = result["m1_canonical_tree_sha256"] == result["m2s2_canonical_tree_sha256"]
    Path(args.output).write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    if args.classification_csv:
        with Path(args.classification_csv).open("w", newline="", encoding="utf-8") as handle:
            fields = ("filename", "pointer_or_column", "equality", "classification", "justification")
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(differences)
    if args.per_file_csv:
        with Path(args.per_file_csv).open("w", newline="", encoding="utf-8") as handle:
            fields = ("filename", "m1_raw_sha256", "m2s2_raw_sha256", "m1_bytes", "m2s2_bytes", "m1_canonical_sha256", "m2s2_canonical_sha256")
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(per_file)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m1", required=True)
    parser.add_argument("--m2s2", required=True)
    parser.add_argument("--m2l-release", required=True)
    parser.add_argument("--aggregate-schema", required=True)
    parser.add_argument("--stage-schema", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--classification-csv")
    parser.add_argument("--per-file-csv")
    args = parser.parse_args()
    result = audit(args)
    if result["unclassified_count"] or not result["canonical_exact"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
