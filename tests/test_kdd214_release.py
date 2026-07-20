from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from kdd2027_benchmark.canonical import canonical_bytes, write_canonical_json
from kdd2027_benchmark.errors import ReleaseContractError
from kdd2027_benchmark.report import write_aggregate_report
from kdd2027_benchmark.schema import SCHEMA_FILES, schema_path, validate_schema_directory
from kdd2027_benchmark.submission import validate_submission
from kdd2027_benchmark.transition_entrant import validate_transition_submission


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs/tasks"
LEADERBOARD = ROOT / "submission/leaderboard_submission_template.json"
TRANSITION = ROOT / "fixtures/transition_submission_small.json"


class KDD214SchemaTests(unittest.TestCase):
    def test_complete_released_schema_inventory_is_draft_202012_valid(self):
        rows = validate_schema_directory(ROOT / "schemas")
        self.assertEqual({row["schema"] for row in rows}, set(SCHEMA_FILES.values()))
        self.assertEqual({row["draft"] for row in rows}, {"2020-12"})

    def test_prior_valid_fixtures_pass_with_required_identity_fields(self):
        self.assertEqual(validate_submission(LEADERBOARD, CONFIGS)["valid_rows"], 1)
        self.assertEqual(validate_transition_submission(TRANSITION, CONFIGS)["valid_rows"], 1)

    def test_missing_submission_id_fails_schema_required(self):
        value = self._transition()
        del value["submission_id"]
        self._transition_fails(value, "/ [required]")

    def test_unknown_top_level_and_metric_fields_fail_additional_properties(self):
        transition = self._transition()
        transition["unknown"] = 1
        self._transition_fails(transition, "/ [additionalProperties]")
        leaderboard = self._leaderboard()
        leaderboard["rows"][0]["metrics"]["unknown"] = 0.0
        self._leaderboard_fails(leaderboard, "/rows/0/metrics [additionalProperties]")

    def test_malformed_hash_and_source_commit_fail_patterns(self):
        value = self._transition()
        value["rows"][0]["task_config_sha256"] = "abc"
        self._transition_fails(value, "/rows/0/task_config_sha256 [pattern]")
        value = self._transition()
        value["rows"][0]["source_commit"] = "not-a-commit"
        self._transition_fails(value, "/rows/0/source_commit [pattern]")

    def test_wrong_task_hash_is_a_semantic_failure(self):
        value = self._transition()
        value["rows"][0]["task_config_sha256"] = "0" * 64
        self._transition_fails(value, "Transition task hash mismatch")

    def test_wrong_schema_horizon_and_metric_fail_const_or_enum(self):
        value = self._transition()
        value["schema_version"] = "future"
        self._transition_fails(value, "/schema_version [const]")
        value = self._transition()
        value["rows"][0]["horizon"] = "H99"
        self._transition_fails(value, "/rows/0/horizon [enum]")
        value = self._transition()
        value["rows"][0]["metric_name"] = "accuracy"
        self._transition_fails(value, "/rows/0/metric_name [enum]")

    def test_nonfinite_values_fail_at_metric_pointer(self):
        for value in (math.nan, math.inf, -math.inf):
            fixture = self._transition()
            fixture["rows"][0]["metric_value"] = value
            self._transition_fails(fixture, "/rows/0/metric_value [number]")

    def test_nonpositive_count_and_false_acknowledgement_fail_schema(self):
        for count in (0, -1):
            value = self._transition()
            value["rows"][0]["observed_target_count"] = count
            self._transition_fails(value, "/rows/0/observed_target_count [minimum]")
        value = self._transition()
        value["rows"][0]["claim_boundary_acknowledged"] = False
        self._transition_fails(value, "/rows/0/claim_boundary_acknowledged [const]")

    def test_duplicate_metric_identity_and_count_inconsistency_fail_semantics(self):
        value = self._transition()
        value["rows"].append(copy.deepcopy(value["rows"][0]))
        self._transition_fails(value, "Duplicate transition metric identity")
        value = self._transition()
        second = copy.deepcopy(value["rows"][0])
        second["metric_name"] = "mae"
        second["observed_target_count"] = 127
        value["rows"].append(second)
        self._transition_fails(value, "Observed target count mismatch")

    def test_wrong_array_or_object_type_fails_schema(self):
        value = self._transition()
        value["rows"] = {}
        self._transition_fails(value, "/rows [type]")
        value = self._leaderboard()
        value["rows"][0]["metrics"] = []
        self._leaderboard_fails(value, "/rows/0/metrics [type]")

    def test_valid_looking_enum_violation_is_rejected(self):
        value = self._leaderboard()
        value["rows"][0]["implementation_fidelity"] = "native"
        self._leaderboard_fails(value, "/rows/0/implementation_fidelity [enum]")

    def test_aggregate_report_is_bound_to_aggregate_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid = {"overall_rmse": 1.0, "unknown": 2}
            write_canonical_json(root / "invalid.json", invalid)
            with self.assertRaisesRegex(ReleaseContractError, r"\[additionalProperties\]|\[required\]"):
                write_aggregate_report(root, root / "report.md")

    def _transition(self) -> dict:
        return json.loads(TRANSITION.read_text(encoding="utf-8"))

    def _leaderboard(self) -> dict:
        return json.loads(LEADERBOARD.read_text(encoding="utf-8"))

    def _transition_fails(self, value: dict, expected: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.json"
            path.write_text(json.dumps(value, allow_nan=True), encoding="utf-8")
            with self.assertRaisesRegex(ReleaseContractError, self._escape(expected)):
                validate_transition_submission(path, CONFIGS)

    def _leaderboard_fails(self, value: dict, expected: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.json"
            path.write_text(json.dumps(value, allow_nan=True), encoding="utf-8")
            with self.assertRaisesRegex(ReleaseContractError, self._escape(expected)):
                validate_submission(path, CONFIGS)

    @staticmethod
    def _escape(value: str) -> str:
        import re
        return re.escape(value)


class KDD214CanonicalTests(unittest.TestCase):
    def test_canonical_bytes_sort_quantize_and_end_with_lf(self):
        value = {"z": 1.23456789012349, "a": "é", "nested": [True, -0.0]}
        self.assertEqual(
            canonical_bytes(value),
            b'{"a":"\xc3\xa9","nested":[true,0],"z":1.234567890123}\n',
        )

    def test_canonical_writer_rejects_every_nonfinite_value(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaisesRegex(ReleaseContractError, "nonfinite"):
                canonical_bytes({"metric": value})

    def test_quantization_error_is_tighter_than_reporting_precision(self):
        value = 0.12345678901249
        encoded = json.loads(canonical_bytes({"value": value}))
        self.assertLessEqual(abs(encoded["value"] - value), 5e-13)
        self.assertLess(5e-13, 1e-6)

    def test_released_schema_paths_are_the_single_authority(self):
        for name, filename in SCHEMA_FILES.items():
            self.assertEqual(schema_path(name).name, filename)


if __name__ == "__main__":
    unittest.main()
