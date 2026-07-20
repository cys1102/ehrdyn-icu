from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import numpy as np

from kdd2027_benchmark.current_five_task.contracts import ContractError, TASKS, validate_layout
from kdd2027_benchmark.current_five_task.reconstruct import (
    DEFAULT_CHUNK_ROWS,
    HIGH_VOLUME_TABLES,
    _stream_events,
    build_anchors,
    build_arrays,
    build_transitions,
    encode_actions,
    finalize_sepsis_anchors,
    kdigo_creatinine_events,
    load_core_with_time_audit,
    reconstruct,
    scan_creatinine,
)
from tests.test_kdd217ar3a import make_fixture


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "schemas" / "credentialed_aggregate_receipt.schema.json"


def _creatinine_rows() -> list[dict[str, object]]:
    return [
        {"subject_id": "", "hadm_id": "", "charttime": "not-a-time", "itemid": 99999, "valuenum": 9.0},
        {"subject_id": 101, "hadm_id": 1001, "charttime": "2020-01-01 00:00:00", "itemid": 50912, "valuenum": 1.0},
        {"subject_id": "", "hadm_id": 1001, "charttime": "2020-01-01 01:00:00", "itemid": 50912, "valuenum": 2.0},
        {"subject_id": 101, "hadm_id": "", "charttime": "2020-01-01 02:00:00", "itemid": 50912, "valuenum": 2.0},
        {"subject_id": 101, "hadm_id": 1001, "charttime": "", "itemid": 50912, "valuenum": 2.0},
        {"subject_id": 101, "hadm_id": 1001, "charttime": "2020-01-02 23:00:00", "itemid": 50912, "valuenum": 1.31},
        {"subject_id": 101, "hadm_id": 1002, "charttime": "2020-01-07 00:00:00", "itemid": 50912, "valuenum": 1.5},
        {"subject_id": 101, "hadm_id": 1002, "charttime": "2020-01-07 00:00:00", "itemid": 50912, "valuenum": 1.6},
        {"subject_id": 101, "hadm_id": 1002, "charttime": "2020-01-07 01:00:00", "itemid": 99999, "valuenum": 9.0},
    ]


def _write_labs(path: Path, rows: list[dict[str, object]], *, compressed: bool) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False, compression="gzip" if compressed else None)


class KDD220AR3AKIAndStreamingTests(unittest.TestCase):
    def test_authoritative_two_window_events(self) -> None:
        rows = pd.DataFrame([
            {"subject_id": 1, "hadm_id": 10, "charttime": pd.Timestamp("2020-01-01"), "value": 1.0, "__source_order": 0},
            {"subject_id": 1, "hadm_id": 10, "charttime": pd.Timestamp("2020-01-02 23:00"), "value": 1.31, "__source_order": 1},
            {"subject_id": 2, "hadm_id": 21, "charttime": pd.Timestamp("2020-01-01"), "value": 1.0, "__source_order": 2},
            {"subject_id": 2, "hadm_id": 21, "charttime": pd.Timestamp("2020-01-07"), "value": 1.5, "__source_order": 3},
            {"subject_id": 3, "hadm_id": 30, "charttime": pd.Timestamp("2020-01-01"), "value": 1.0, "__source_order": 4},
            {"subject_id": 3, "hadm_id": 30, "charttime": pd.Timestamp("2020-01-07"), "value": 1.31, "__source_order": 5},
            {"subject_id": 4, "hadm_id": 40, "charttime": pd.Timestamp("2020-01-01"), "value": 1.0, "__source_order": 6},
            {"subject_id": 4, "hadm_id": 41, "charttime": pd.Timestamp("2020-01-02"), "value": 2.0, "__source_order": 7},
        ])
        events = kdigo_creatinine_events(rows)
        self.assertEqual(events["anchor_source"].tolist(), ["creatinine_delta_48h", "creatinine_ratio_7d"])
        self.assertEqual(events["hadm_id"].tolist(), [10, 21])
        self.assertNotIn(30, events["hadm_id"].tolist())
        self.assertNotIn(41, events["hadm_id"].tolist())

    def test_nullable_keys_chunk_boundary_and_ties_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labevents.csv"
            _write_labs(path, _creatinine_rows(), compressed=False)
            audit_small: dict[tuple[str, str, str], int] = {}
            audit_large: dict[tuple[str, str, str], int] = {}
            small = scan_creatinine(path, chunk_rows=2, audit=audit_small)
            large = scan_creatinine(path, chunk_rows=7, audit=audit_large)
            pd.testing.assert_frame_equal(small, large)
            self.assertEqual(audit_small, audit_large)
            self.assertEqual(len(small), 4)
            tied = small[small["charttime"].eq(pd.Timestamp("2020-01-07"))]
            self.assertEqual(tied["value"].tolist(), [1.5, 1.6])
            self.assertEqual(audit_small[("hosp/labevents", "subject_id", "missing_required_identifier")], 1)
            self.assertEqual(audit_small[("hosp/labevents", "hadm_id", "missing_required_identifier")], 1)
            self.assertEqual(audit_small[("hosp/labevents", "charttime", "missing_required_timestamp")], 1)

    def test_compressed_and_uncompressed_scan_parity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plain, compressed = root / "labs.csv", root / "labs.csv.gz"
            rows = _creatinine_rows()
            _write_labs(plain, rows, compressed=False)
            _write_labs(compressed, rows, compressed=True)
            left = scan_creatinine(plain, chunk_rows=3, audit={})
            right = scan_creatinine(compressed, chunk_rows=5, audit={})
            pd.testing.assert_frame_equal(left, right)

    def test_stay_item_and_interval_timestamp_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chart = root / "chart.csv"
            pd.DataFrame([
                {"stay_id": 1, "charttime": "2020-01-01", "itemid": 220045, "valuenum": 80, "valueuom": "bpm"},
                {"stay_id": "", "charttime": "2020-01-01", "itemid": 220045, "valuenum": 80, "valueuom": "bpm"},
                {"stay_id": 1, "charttime": "2020-01-01", "itemid": "", "valuenum": 80, "valueuom": "bpm"},
                {"stay_id": 1, "charttime": "", "itemid": 220045, "valuenum": 80, "valueuom": "bpm"},
            ]).to_csv(chart, index=False)
            audit: dict[tuple[str, str, str], int] = {}
            selected = _stream_events(
                chart, table="icu/chartevents",
                columns=("stay_id", "charttime", "itemid", "valuenum", "valueuom"),
                dates=("charttime",), required_ids=("stay_id", "itemid"), required_times=("charttime",),
                audit=audit, chunk_rows=2, itemids={220045}, key="stay_id", eligible_keys={1},
            )
            self.assertEqual(len(selected), 1)
            self.assertEqual(audit[("icu/chartevents", "stay_id", "missing_required_identifier")], 1)
            self.assertEqual(audit[("icu/chartevents", "itemid", "missing_required_identifier")], 1)
            self.assertEqual(audit[("icu/chartevents", "charttime", "missing_required_timestamp")], 1)

            intervals = root / "inputs.csv"
            pd.DataFrame([
                {"stay_id": 1, "starttime": "2020-01-01", "endtime": "", "itemid": 221906},
                {"stay_id": 1, "starttime": "", "endtime": "", "itemid": 221906},
            ]).to_csv(intervals, index=False)
            interval_audit: dict[tuple[str, str, str], int] = {}
            selected = _stream_events(
                intervals, table="icu/inputevents",
                columns=("stay_id", "starttime", "endtime", "itemid"),
                dates=("starttime", "endtime"), required_ids=("stay_id", "itemid"), required_times=("starttime",),
                audit=interval_audit, chunk_rows=1, itemids={221906}, key="stay_id", eligible_keys={1},
            )
            self.assertEqual(len(selected), 1)
            self.assertTrue(pd.isna(selected.iloc[0]["endtime"]))
            self.assertEqual(interval_audit[("icu/inputevents", "starttime", "missing_required_timestamp")], 1)

    def test_malformed_nonmissing_identifier_and_timestamp_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad_id = root / "bad-id.csv"
            rows = [_creatinine_rows()[1].copy()]; rows[0]["hadm_id"] = "not-an-id"
            _write_labs(bad_id, rows, compressed=False)
            with self.assertRaisesRegex(ContractError, "hosp/labevents.hadm_id"):
                scan_creatinine(bad_id, chunk_rows=1, audit={})
            bad_time = root / "bad-time.csv"
            rows = [_creatinine_rows()[1].copy()]; rows[0]["charttime"] = "not-a-time"
            _write_labs(bad_time, rows, compressed=False)
            with self.assertRaises(ContractError):
                scan_creatinine(bad_time, chunk_rows=1, audit={})

    def test_full_five_task_output_is_chunk_size_invariant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = make_fixture(Path(directory))
            first, second = Path(directory) / "first", Path(directory) / "second"
            reconstruct(root, first, SCHEMA, source_hashes={"fixture": "a" * 64}, chunk_rows=2)
            reconstruct(root, second, SCHEMA, source_hashes={"fixture": "a" * 64}, chunk_rows=17)
            for name in (
                "aggregate_receipt.json",
                "icu_time_order_eligibility_aggregate.csv",
                "nullable_key_and_timestamp_exclusion_aggregate.csv",
            ):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

    def test_full_internal_interface_is_chunk_size_invariant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = make_fixture(Path(directory)); paths = validate_layout(root)
            snapshots = []
            for chunk_rows in (2, 17):
                audit: dict[tuple[str, str, str], int] = {}
                stays, diagnoses, _ = load_core_with_time_audit(paths, audit)
                candidates = build_anchors(paths, stays, diagnoses, chunk_rows=chunk_rows, audit=audit)
                arrays = build_arrays(paths, candidates, chunk_rows=chunk_rows, audit=audit)
                candidates = finalize_sepsis_anchors(candidates, arrays)
                transitions = build_transitions(candidates)
                actions = {}
                for task in TASKS:
                    local = transitions[transitions["task"].eq(task)].reset_index(drop=True)
                    actions[task] = encode_actions(task, local, arrays)[0]
                snapshots.append((candidates, transitions, arrays, actions, audit))
            left, right = snapshots
            pd.testing.assert_frame_equal(left[0], right[0])
            pd.testing.assert_frame_equal(left[1], right[1])
            for name in left[2]:
                np.testing.assert_allclose(left[2][name], right[2][name], rtol=0, atol=0, equal_nan=True)
            for task in TASKS:
                np.testing.assert_array_equal(left[3][task], right[3][task])
            self.assertEqual(left[4], right[4])

    def test_high_volume_sources_have_no_unbounded_read_fallback(self) -> None:
        source = (ROOT / "src/kdd2027_benchmark/current_five_task/reconstruct.py").read_text(encoding="utf-8")
        for table in HIGH_VOLUME_TABLES:
            self.assertNotIn(f'_read(paths["{table}"]', source)
        self.assertEqual(DEFAULT_CHUNK_ROWS, 250000)
        self.assertIn("chunksize=chunk_rows", source)


if __name__ == "__main__":
    unittest.main()
