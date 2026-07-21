from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator, ValidationError

from kdd2027_benchmark.current_five_task.contracts import (
    FEATURE_INDEX,
    FEATURE_NAMES,
    ContractError,
    fit_train_positive_edges,
)
from kdd2027_benchmark.current_five_task.reconstruct import (
    HIGH_VOLUME_TABLE_ORDER,
    _stream_events,
    encode_actions,
    filter_respiratory_action_transitions,
    reconstruct,
)
from tests.test_kdd217ar3a import make_fixture


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "schemas" / "credentialed_aggregate_receipt.schema.json"
STOP_SCHEMA = ROOT / "schemas" / "credentialed_controlled_stop_receipt.schema.json"


class KDD220AR4RespiratoryAndStreamingTests(unittest.TestCase):
    def test_partial_missing_actions_are_filtered_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = make_fixture(Path(directory))
            chart_path = root / "icu" / "chartevents.csv.gz"
            chart = pd.read_csv(chart_path)
            respiratory_stay = int(chart.loc[chart["itemid"].eq(220339), "stay_id"].iloc[0])
            times = pd.to_datetime(chart["charttime"])
            bins = ((times - pd.Timestamp("2020-01-01")) / pd.Timedelta(hours=4)).astype(int)
            remove_peep = chart["stay_id"].eq(respiratory_stay) & chart["itemid"].eq(220339) & bins.isin((7, 11, 16))
            remove_fio2 = chart["stay_id"].eq(respiratory_stay) & chart["itemid"].eq(226754) & bins.eq(9)
            chart = chart.loc[~(remove_peep | remove_fio2)]
            chart.to_csv(chart_path, index=False, compression="gzip")
            output = Path(directory) / "output"
            receipt = reconstruct(root, output, SCHEMA, chunk_rows=3)
            respiratory = next(row for row in receipt["tasks"] if row["task_id"] == "respiratory_support")
            self.assertEqual(respiratory["episodes"], 1)
            self.assertEqual(respiratory["decisions"], 6)
            audit = pd.read_csv(output / "respiratory_action_filter_aggregate.csv").iloc[0]
            self.assertEqual(int(audit["candidate_transitions"]), 10)
            self.assertEqual(int(audit["retained_transitions"]), 6)
            self.assertEqual(int(audit["excluded_missing_action_transitions"]), 4)
            self.assertEqual(int(audit["excluded_empty_episodes"]), 0)

    def test_missing_action_positions_filter_without_realignment(self) -> None:
        specifications = {
            0: ("both", "peep_missing", "fio2_missing", "both_missing"),
            1: ("both_missing", "both", "both"),
            2: ("both", "peep_missing", "both"),
            3: ("both", "both", "fio2_missing"),
            4: ("peep_missing", "both_missing"),
        }
        rows = []
        values = np.full((5, 6, len(FEATURE_NAMES)), np.nan, dtype=np.float32)
        masks = np.zeros_like(values, dtype=bool)
        peep = np.full((5, 6), np.nan, dtype=np.float32)
        peep_observed = np.zeros((5, 6), dtype=bool)
        expected_valid = []
        counter = 0
        for episode, kinds in specifications.items():
            for relative, kind in enumerate(kinds):
                action = relative + 1
                rows.append({
                    "task": "respiratory_support", "role": "train",
                    "episode_idx": episode, "relative_transition": relative,
                    "state_idx": action - 1, "action_idx": action,
                    "target_idx": action + 1, "synthetic_subject": episode + 100,
                })
                if kind in {"both", "fio2_missing"}:
                    peep[episode, action] = 4.0 + counter
                    peep_observed[episode, action] = True
                if kind in {"both", "peep_missing"}:
                    values[episode, action, FEATURE_INDEX["fio2"]] = 25.0 + counter
                    masks[episode, action, FEATURE_INDEX["fio2"]] = True
                expected_valid.append(kind == "both")
                counter += 1
        transitions = pd.DataFrame(rows)
        original = transitions.copy(deep=True)
        arrays = {"values": values, "masks": masks, "peep": peep, "peep_observed": peep_observed}
        actions, contract = encode_actions("respiratory_support", transitions, arrays)
        valid = np.asarray(expected_valid, dtype=bool)
        np.testing.assert_array_equal(contract["valid_action_mask"], valid)
        np.testing.assert_array_equal(actions[~valid], np.full((~valid).sum(), -1, dtype=np.int16))
        self.assertFalse(np.any(actions[~valid] == 0))

        retained, retained_actions, audit = filter_respiratory_action_transitions(
            transitions, actions, contract["valid_action_mask"]
        )
        pd.testing.assert_frame_equal(transitions, original)
        pd.testing.assert_frame_equal(retained, original.loc[valid].reset_index(drop=True))
        self.assertEqual(audit, {
            "candidate_transitions": 15,
            "retained_transitions": 7,
            "excluded_missing_action_transitions": 8,
            "candidate_episodes": 5,
            "retained_episodes": 4,
            "excluded_empty_episodes": 1,
        })
        self.assertNotIn(4, retained["episode_idx"].tolist())
        self.assertTrue(((retained["state_idx"] + 1) == retained["action_idx"]).all())
        self.assertTrue(((retained["action_idx"] + 1) == retained["target_idx"]).all())
        self.assertTrue(np.all((retained_actions >= 0) & (retained_actions < 25)))
        self.assertEqual(retained["role"].unique().tolist(), ["train"])
        self.assertEqual(retained["synthetic_subject"].tolist(), original.loc[valid, "synthetic_subject"].tolist())

    def test_cutpoints_use_train_positive_observed_only(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 999.0, 1000.0, -5.0])
        observed = np.array([1, 1, 1, 1, 1, 0, 1], dtype=bool)
        roles = np.array(["train", "train", "train", "train", "validation", "train", "train"])
        self.assertEqual(fit_train_positive_edges(values, observed, roles), (1.75, 2.5, 3.25))

    def test_retained_actions_fail_closed_on_mask_or_range_drift(self) -> None:
        transitions = pd.DataFrame([{
            "episode_idx": 0, "role": "train", "relative_transition": 0,
            "state_idx": 0, "action_idx": 1, "target_idx": 2,
        }])
        with self.assertRaisesRegex(ContractError, "mask and encoded missingness"):
            filter_respiratory_action_transitions(transitions, np.array([0]), np.array([False]))
        with self.assertRaisesRegex(ContractError, "K=25"):
            filter_respiratory_action_transitions(transitions, np.array([25]), np.array([True]))

    def test_streaming_counts_chunk_and_compression_are_explicit(self) -> None:
        frame = pd.DataFrame([
            {"stay_id": 1, "charttime": f"2020-01-01 0{hour}:00:00", "itemid": item, "valuenum": 80, "valueuom": "bpm"}
            for hour, item in enumerate((220045, 220045, 999999, 220045, 220045))
        ])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plain, compressed = root / "chart.csv", root / "chart.csv.gz"
            frame.to_csv(plain, index=False)
            frame.to_csv(compressed, index=False, compression="gzip")
            outputs = []
            audits = []
            for path, chunk_rows in ((plain, 2), (compressed, 4)):
                stream: dict[str, dict[str, int | str]] = {}
                outputs.append(_stream_events(
                    path, table="icu/chartevents",
                    columns=("stay_id", "charttime", "itemid", "valuenum", "valueuom"),
                    dates=("charttime",), required_ids=("stay_id", "itemid"),
                    required_times=("charttime",), audit={}, chunk_rows=chunk_rows,
                    itemids={220045}, key="stay_id", eligible_keys={1},
                    sort_by=("stay_id", "charttime", "itemid"), streaming_audit=stream,
                ))
                audits.append(stream["icu/chartevents"])
            pd.testing.assert_frame_equal(outputs[0], outputs[1])
            self.assertEqual([audit["rows_read"] for audit in audits], [5, 5])
            self.assertEqual([audit["rows_retained"] for audit in audits], [4, 4])
            self.assertEqual([audit["chunks_processed"] for audit in audits], [3, 2])
            self.assertEqual([audit["effective_chunk_size"] for audit in audits], [2, 4])
            self.assertEqual([audit["compression_encoding"] for audit in audits], ["csv", "csv_gz"])

    def test_success_receipt_streaming_schema_and_nonrespiratory_nonregression(self) -> None:
        expected = {
            "sepsis": (1, 1, 10, "3224ba7456ee7a4b650b0b06ba681c3861a26a530d84ad058012b74848f9ad4a"),
            "shock": (5, 5, 49, "77b8b92ea7df8d4f9ecf8743e8cbad93b8379b13fc75fa05b8d75659a3080e3e"),
            "aki": (1, 1, 11, "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"),
            "heart_failure": (1, 1, 10, "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = make_fixture(Path(directory))
            output = Path(directory) / "output"
            receipt = reconstruct(root, output, SCHEMA, source_hashes={"fixture": "a" * 64}, chunk_rows=3)
            validator = Draft202012Validator(json.loads(SCHEMA.read_text()))
            validator.validate(receipt)
            self.assertEqual(receipt["schema_version"], "1.1.0")
            self.assertEqual([row["table"] for row in receipt["streaming"]], list(HIGH_VOLUME_TABLE_ORDER))
            self.assertTrue(all(row["scan_count"] >= 1 for row in receipt["streaming"]))
            self.assertTrue(all(row["effective_chunk_size"] == 3 for row in receipt["streaming"]))
            by_task = {row["task_id"]: row for row in receipt["tasks"]}
            for task, values in expected.items():
                row = by_task[task]
                self.assertEqual((row["subjects"], row["episodes"], row["decisions"], row["cutpoint_hash"]), values)
            self.assertTrue((output / "streaming_instrumentation_aggregate.csv").is_file())
            self.assertTrue((output / "respiratory_action_filter_aggregate.csv").is_file())
            self.assertTrue((output / "runtime_resource_aggregate.json").is_file())
            invalid = copy.deepcopy(receipt)
            invalid["streaming"][0]["private_path"] = "/not/allowed"
            with self.assertRaises(ValidationError):
                validator.validate(invalid)
            wrong_table = copy.deepcopy(receipt)
            wrong_table["streaming"][0]["table"] = "icu/not_a_released_table"
            with self.assertRaises(ValidationError):
                validator.validate(wrong_table)

    def test_controlled_stop_is_schema_bound_and_has_no_free_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = make_fixture(Path(directory))
            chart_path = root / "icu" / "chartevents.csv.gz"
            chart = pd.read_csv(chart_path)
            chart = chart[~chart["itemid"].isin((220339, 226754))]
            chart.to_csv(chart_path, index=False, compression="gzip")
            output = Path(directory) / "stopped"
            with self.assertRaises(ContractError):
                reconstruct(root, output, SCHEMA, chunk_rows=3)
            stop = json.loads((output / "controlled_stop_receipt.json").read_text())
            schema = json.loads(STOP_SCHEMA.read_text())
            Draft202012Validator.check_schema(schema)
            Draft202012Validator(schema).validate(stop)
            self.assertEqual(stop["failure_code"], "contract_error")
            self.assertEqual([row["table"] for row in stop["streaming"]], list(HIGH_VOLUME_TABLE_ORDER))
            invalid = copy.deepcopy(stop)
            invalid["error_message"] = "not allowed"
            with self.assertRaises(ValidationError):
                Draft202012Validator(schema).validate(invalid)


if __name__ == "__main__":
    unittest.main()
