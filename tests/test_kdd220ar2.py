from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from kdd2027_benchmark.current_five_task.contracts import ContractError, assert_unique
from kdd2027_benchmark.current_five_task.reconstruct import (
    _read,
    classify_icu_time_eligibility,
    reconstruct,
)
from tests.test_kdd217ar3a import make_fixture


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "schemas" / "credentialed_aggregate_receipt.schema.json"


def official_shape_rows() -> list[dict[str, object]]:
    return [
        {"subject_id": 1, "hadm_id": 11, "stay_id": 101, "intime": "2020-01-01 00:00:00", "outtime": "2020-01-01 01:00:00"},
        {"subject_id": 2, "hadm_id": 12, "stay_id": 102, "intime": "", "outtime": "2020-01-01 04:00:00"},
        {"subject_id": 3, "hadm_id": 13, "stay_id": 103, "intime": "2020-01-01 00:00:00", "outtime": ""},
        {"subject_id": 4, "hadm_id": 14, "stay_id": 104, "intime": "2020-01-01 00:00:00", "outtime": "2020-01-01 00:00:00"},
        {"subject_id": 5, "hadm_id": 15, "stay_id": 105, "intime": "2020-01-01 04:00:00", "outtime": "2020-01-01 00:00:00"},
    ]


class KDD220AR2EligibilityTests(unittest.TestCase):
    def test_mutually_exclusive_time_order_reasons(self) -> None:
        parsed = pd.DataFrame(official_shape_rows())
        parsed["intime"] = pd.to_datetime(parsed["intime"], errors="coerce")
        parsed["outtime"] = pd.to_datetime(parsed["outtime"], errors="coerce")
        valid_before = parsed.iloc[[0]].reset_index(drop=True)
        retained, audit = classify_icu_time_eligibility(parsed)
        pd.testing.assert_frame_equal(retained.reset_index(drop=True), valid_before)
        self.assertEqual(audit, {
            "total_merged_icu_stays_considered": 5,
            "valid_time_order_stays": 1,
            "missing_intime": 1,
            "missing_outtime": 1,
            "equal_intime_outtime": 1,
            "reversed_time_order": 1,
        })

    def test_no_universal_minimum_stay_length_is_added(self) -> None:
        frame = pd.DataFrame([official_shape_rows()[0]])
        frame["intime"] = pd.to_datetime(frame["intime"])
        frame["outtime"] = pd.to_datetime(frame["outtime"])
        retained, _ = classify_icu_time_eligibility(frame)
        self.assertEqual(len(retained), 1)

    def test_nonempty_malformed_timestamp_still_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "icustays.csv"
            rows = official_shape_rows()[:1]
            rows[0]["intime"] = "not-a-timestamp"
            pd.DataFrame(rows).to_csv(path, index=False)
            with self.assertRaises(ContractError):
                _read(path, dates=("intime", "outtime"))

    def test_duplicate_primary_identifier_still_fails(self) -> None:
        frame = pd.DataFrame(official_shape_rows()[:1] * 2)
        with self.assertRaises(ContractError):
            assert_unique(frame, ("stay_id",), "icustays")

    def test_five_task_fixture_emits_audit_without_contract_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = make_fixture(Path(directory))
            output = Path(directory) / "output"
            receipt = reconstruct(root, output, SCHEMA, source_hashes={"fixture": "a" * 64})
            self.assertEqual([row["task_id"] for row in receipt["tasks"]], [
                "sepsis", "respiratory_support", "shock", "aki", "heart_failure"
            ])
            with (output / "icu_time_order_eligibility_aggregate.csv").open(newline="", encoding="utf-8") as handle:
                audit = {row["category"]: int(row["count"]) for row in csv.DictReader(handle)}
            self.assertEqual(audit["total_merged_icu_stays_considered"], audit["valid_time_order_stays"])
            self.assertEqual(sum(audit[key] for key in (
                "missing_intime", "missing_outtime", "equal_intime_outtime", "reversed_time_order"
            )), 0)


if __name__ == "__main__":
    unittest.main()
