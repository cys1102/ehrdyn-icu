from __future__ import annotations

import copy
import gzip
import hashlib
import inspect
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

from kdd2027_benchmark.current_five_task.reconstruct import reconstruct
from tests.test_kdd217ar3a import make_fixture


ROOT = Path(__file__).resolve().parents[1]
RECEIPT_SCHEMA = ROOT / "schemas" / "credentialed_aggregate_receipt.schema.json"
RESOURCE_SCHEMA = ROOT / "schemas" / "stage_resource_instrumentation.schema.json"
AR6_FIXTURE_RECEIPT_SHA256 = "8f125bbb524132b79db68fa4a28b4bf04aa5438ffc742ebb979bb15dec405f8a"
KDD245V2R_FIXTURE_RECEIPT_SHA256 = "a5ab65e8658ae23f7256131dc297ade71f32408f6949e0a6aa1e10c075839ee8"
AR6_SCIENTIFIC_SURFACE_SHA256 = "f412a1c03d2325339543628384c4aad14dd0ffbf92160a4d51d11c4c42b750a3"


def _uncompress_fixture(root: Path) -> None:
    for compressed in sorted(root.rglob("*.csv.gz")):
        plain = compressed.with_suffix("")
        with gzip.open(compressed, "rb") as source, plain.open("wb") as target:
            shutil.copyfileobj(source, target)
        compressed.unlink()


class KDD220AR7BoundedMemoryTests(unittest.TestCase):
    def _run_fixture(self, chunk_rows: int, *, uncompressed: bool = False) -> tuple[dict, bytes, dict]:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        mimic = make_fixture(root)
        if uncompressed:
            _uncompress_fixture(mimic)
        output = root / "aggregate-output"
        receipt = reconstruct(
            mimic,
            output,
            RECEIPT_SCHEMA,
            source_hashes={"fixture": "a" * 64},
            chunk_rows=chunk_rows,
        )
        stage = json.loads((output / "stage_resource_instrumentation_aggregate.json").read_text())
        return receipt, (output / "aggregate_receipt.json").read_bytes(), stage

    def test_ar6_fixture_receipt_and_scientific_surface_are_exact(self) -> None:
        receipt, receipt_bytes, _ = self._run_fixture(3)
        if sys.version_info[:2] == (3, 11):
            constructor_source = ROOT / "src/kdd2027_benchmark/current_five_task/reconstruct.py"
            if (
                hashlib.sha256(constructor_source.read_bytes()).hexdigest()
                == "914356315ba3a489b6d97f5b2770b3c87e1b770e3ff735a6198a9a064f69592f"
            ):
                self.assertEqual(__import__("pandas").__version__, "3.0.3")
                expected = KDD245V2R_FIXTURE_RECEIPT_SHA256
            else:
                expected = AR6_FIXTURE_RECEIPT_SHA256
            self.assertEqual(hashlib.sha256(receipt_bytes).hexdigest(), expected)
        self.assertEqual(
            receipt["contracts"]["scientific_surface_sha256"],
            AR6_SCIENTIFIC_SURFACE_SHA256,
        )
        self.assertEqual(
            [task["task_id"] for task in receipt["tasks"]],
            ["sepsis", "respiratory_support", "shock", "aki", "heart_failure"],
        )

    def test_chunk_and_compression_scientific_surface_invariance(self) -> None:
        hashes = {
            self._run_fixture(chunk_rows)[0]["contracts"]["scientific_surface_sha256"]
            for chunk_rows in (25_000, 50_000, 100_000, 250_000)
        }
        hashes.add(self._run_fixture(25_000, uncompressed=True)[0]["contracts"]["scientific_surface_sha256"])
        self.assertEqual(hashes, {AR6_SCIENTIFIC_SURFACE_SHA256})

    def test_stage_resource_schema_and_required_stages(self) -> None:
        _, _, stage = self._run_fixture(25_000)
        schema = json.loads(RESOURCE_SCHEMA.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(stage)
        names = [row["stage"] for row in stage["stages"]]
        for name in (
            "core_loading", "anchor_construction", "array_construction",
            "task_aggregation:sepsis", "task_aggregation:respiratory_support",
            "task_aggregation:shock", "task_aggregation:aki",
            "task_aggregation:heart_failure", "schema_validation", "receipt_writing",
        ):
            self.assertIn(name, names)
        for table in (
            "hosp/labevents", "hosp/microbiologyevents", "hosp/prescriptions",
            "icu/chartevents", "icu/inputevents", "icu/outputevents", "icu/procedureevents",
        ):
            self.assertTrue(any(name.startswith(f"high_volume_scan:{table}:") for name in names))
        invalid = copy.deepcopy(stage)
        invalid["stages"][0]["patient_id"] = 1
        with self.assertRaises(ValidationError):
            Draft202012Validator(schema).validate(invalid)

    def test_no_retained_frame_list_or_full_copy_concat_in_streaming_scan(self) -> None:
        module = __import__(
            "kdd2027_benchmark.current_five_task.reconstruct",
            fromlist=["_stream_events"],
        )
        source = inspect.getsource(module._stream_events)
        self.assertNotIn("selected_frames", source)
        self.assertNotIn("pd.concat", source)
        self.assertIn("_PartitionStore", source)

    def test_controlled_stop_cleans_private_temporary_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "controlled-stop"
            with self.assertRaises(Exception):
                reconstruct(root / "missing-input", output, RECEIPT_SCHEMA)
            self.assertTrue((output / "controlled_stop_receipt.json").is_file())
            self.assertTrue((output / "stage_resource_instrumentation_aggregate.json").is_file())
            self.assertEqual(list(root.glob(".kdd220ar7-*")), [])


if __name__ == "__main__":
    unittest.main()
