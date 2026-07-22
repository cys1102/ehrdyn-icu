from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from kdd2027_benchmark.current_five_task.reconstruct import reconstruct
from scripts.audit_kdd220m2i_identity import (
    IdentityError, canonical_tree, canonicalize_bundle, classify_differences,
)
from tests.test_kdd217ar3a import make_fixture


ROOT = Path(__file__).resolve().parents[1]
AGGREGATE_SCHEMA = ROOT / "schemas" / "credentialed_aggregate_receipt.schema.json"
STAGE_SCHEMA = ROOT / "schemas" / "stage_resource_instrumentation.schema.json"
M2L = ROOT / "release" / "kdd220m2l"


class KDD220M2ICanonicalIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        mimic = make_fixture(root)
        cls.base = root / "candidate"
        reconstruct(mimic, cls.base, AGGREGATE_SCHEMA, source_hashes={"fixture": "a" * 64}, chunk_rows=3)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def copy_bundle(self) -> tuple[tempfile.TemporaryDirectory, Path]:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        target = Path(directory.name) / "candidate"
        shutil.copytree(self.base, target)
        return directory, target

    def canonical(self, root: Path, *, m2l: Path = M2L, aggregate_schema: Path = AGGREGATE_SCHEMA):
        return canonicalize_bundle(root, m2l_release=m2l, aggregate_schema_path=aggregate_schema, stage_schema_path=STAGE_SCHEMA)

    def test_allowed_telemetry_changes_preserve_identity(self) -> None:
        _, changed = self.copy_bundle()
        runtime = json.loads((changed / "runtime_resource_aggregate.json").read_text())
        runtime.update(wall_seconds=999.0, maximum_resident_set_size_kib=999, temporary_disk_bytes=999)
        (changed / "runtime_resource_aggregate.json").write_text(json.dumps(runtime, sort_keys=True, separators=(",", ":")) + "\n")
        stage = json.loads((changed / "stage_resource_instrumentation_aggregate.json").read_text())
        for row in stage["stages"]:
            for field in ("elapsed_seconds", "temporary_disk_high_water_bytes", "rss_entry_kib", "rss_exit_kib", "peak_rss_kib"):
                row[field] = 999
        (changed / "stage_resource_instrumentation_aggregate.json").write_text(json.dumps(stage, sort_keys=True, separators=(",", ":")) + "\n")
        self.assertEqual(canonical_tree(self.canonical(self.base)), canonical_tree(self.canonical(changed)))

    def test_scientific_or_unclassified_change_breaks_identity(self) -> None:
        _, changed = self.copy_bundle()
        receipt = json.loads((changed / "aggregate_receipt.json").read_text())
        receipt["tasks"][0]["subjects"] += 1
        (changed / "aggregate_receipt.json").write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n")
        self.assertNotEqual(canonical_tree(self.canonical(self.base)), canonical_tree(self.canonical(changed)))
        differences = classify_differences(self.base, changed)
        self.assertTrue(any(row["classification"] == "unclassified" for row in differences))

    def test_missing_file_and_schema_change_stop(self) -> None:
        _, missing = self.copy_bundle()
        (missing / "respiratory_action_filter_aggregate.csv").unlink()
        with self.assertRaises(IdentityError):
            self.canonical(missing)
        directory = tempfile.TemporaryDirectory(); self.addCleanup(directory.cleanup)
        schema = Path(directory.name) / "schema.json"
        value = json.loads(AGGREGATE_SCHEMA.read_text()); value["title"] += " changed"
        schema.write_text(json.dumps(value))
        with self.assertRaises(IdentityError):
            self.canonical(self.base, aggregate_schema=schema)

    def test_unclassified_runtime_field_stops(self) -> None:
        _, changed = self.copy_bundle()
        runtime = json.loads((changed / "runtime_resource_aggregate.json").read_text())
        runtime["unknown"] = 1
        (changed / "runtime_resource_aggregate.json").write_text(json.dumps(runtime))
        with self.assertRaises(IdentityError):
            self.canonical(changed)

    def test_encoding_requires_valid_kdd220m2l_receipt(self) -> None:
        directory = tempfile.TemporaryDirectory(); self.addCleanup(directory.cleanup)
        invalid = Path(directory.name) / "kdd220m2l"
        shutil.copytree(M2L, invalid)
        (invalid / "decision.md").write_text("stop_input_encoding_views_not_equivalent_or_incomplete\n")
        with self.assertRaises(IdentityError):
            self.canonical(self.base, m2l=invalid)


if __name__ == "__main__":
    unittest.main()
