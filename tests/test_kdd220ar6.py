from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator, ValidationError

from kdd2027_benchmark.current_five_task.contracts import FEATURE_INDEX, FEATURE_NAMES
from kdd2027_benchmark.current_five_task.reconstruct import (
    aggregate_respiratory_action_bins,
    apply_kdd201_to_frozen_actions,
    encode_actions,
    filter_target_observed_transitions,
    reconstruct,
    respiratory_chart_surfaces,
    sustained_hypotension_from_chart,
)
from tests.test_kdd217ar3a import make_fixture


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "schemas" / "credentialed_aggregate_receipt.schema.json"


class KDD220AR6DualSurfaceAndOrderTests(unittest.TestCase):
    def _events(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"episode_idx": 0, "bin": 1, "itemid": 223835, "valuenum": 0.40, "valueuom": ""},
            {"episode_idx": 0, "bin": 2, "itemid": 223835, "valuenum": 40, "valueuom": ""},
            {"episode_idx": 0, "bin": 3, "itemid": 226754, "valuenum": 40, "valueuom": "%"},
            {"episode_idx": 0, "bin": 4, "itemid": 226754, "valuenum": 40, "valueuom": ""},
            {"episode_idx": 0, "bin": 5, "itemid": 226754, "valuenum": 20, "valueuom": "%"},
            {"episode_idx": 0, "bin": 6, "itemid": 227010, "valuenum": 101, "valueuom": "%"},
            {"episode_idx": 0, "bin": 7, "itemid": 223835, "valuenum": .30, "valueuom": ""},
            {"episode_idx": 0, "bin": 7, "itemid": 226754, "valuenum": 50, "valueuom": "%"},
            {"episode_idx": 0, "bin": 7, "itemid": 229280, "valuenum": 70, "valueuom": "%"},
            {"episode_idx": 0, "bin": 8, "itemid": 220339, "valuenum": 0, "valueuom": "cmH2O"},
            {"episode_idx": 0, "bin": 9, "itemid": 224700, "valuenum": 30, "valueuom": "cmH2O"},
            {"episode_idx": 0, "bin": 10, "itemid": 220339, "valuenum": -1, "valueuom": "cmH2O"},
            {"episode_idx": 0, "bin": 11, "itemid": 220339, "valuenum": 31, "valueuom": "cmH2O"},
        ])

    def test_item_level_dual_surface_contract(self) -> None:
        parsed = respiratory_chart_surfaces(self._events())
        for index in (0, 1):
            self.assertEqual(parsed.iloc[index].legacy_fio2_value, 40.0)
            self.assertTrue(np.isnan(parsed.iloc[index].safe_fio2_value))
        self.assertEqual(parsed.iloc[2].legacy_fio2_value, 40.0)
        self.assertEqual(parsed.iloc[2].safe_fio2_value, 40.0)
        self.assertEqual(parsed.iloc[3].legacy_fio2_value, 40.0)
        self.assertTrue(np.isnan(parsed.iloc[3].safe_fio2_value))
        self.assertTrue(np.isnan(parsed.iloc[4].legacy_fio2_value))
        self.assertTrue(np.isnan(parsed.iloc[4].safe_fio2_value))
        self.assertTrue(np.isnan(parsed.iloc[5].legacy_fio2_value))
        self.assertTrue(np.isnan(parsed.iloc[5].safe_fio2_value))

    def test_joint_median_and_peep_boundaries(self) -> None:
        rows = aggregate_respiratory_action_bins(self._events()).set_index("bin")
        self.assertEqual(rows.loc[7, "legacy_fio2"], 50.0)
        self.assertEqual(rows.loc[7, "safe_fio2"], 60.0)
        self.assertEqual(rows.loc[8, "legacy_peep"], 0.0)
        self.assertEqual(rows.loc[9, "legacy_peep"], 30.0)
        self.assertFalse(rows.loc[10, "legacy_peep_observed"])
        self.assertFalse(rows.loc[11, "legacy_peep_observed"])

    def test_action_observed_safe_state_unobserved(self) -> None:
        transitions = pd.DataFrame([{"episode_idx": 0, "action_idx": 1, "target_idx": 2, "role": "train"}])
        arrays = {
            "values": np.full((1, 3, len(FEATURE_NAMES)), np.nan),
            "masks": np.zeros((1, 3, len(FEATURE_NAMES)), dtype=bool),
            "pre_repair_masks": np.zeros((1, 3, len(FEATURE_NAMES)), dtype=bool),
            "peep": np.array([[np.nan, 8.0, np.nan]]),
            "peep_observed": np.array([[False, True, False]]),
            "fio2_action": np.array([[np.nan, 40.0, np.nan]]),
            "fio2_action_observed": np.array([[False, True, False]]),
        }
        actions, contract = encode_actions("respiratory_support", transitions, arrays)
        self.assertTrue(contract["valid_action_mask"][0])
        self.assertGreaterEqual(int(actions[0]), 0)
        self.assertFalse(arrays["masks"][0, 1, FEATURE_INDEX["fio2"]])

    def test_original_membership_is_not_repaired_target_mask(self) -> None:
        transitions = pd.DataFrame([{"episode_idx": 0, "target_idx": 1}])
        pre = np.zeros((1, 2, len(FEATURE_NAMES)), dtype=bool)
        final = np.zeros_like(pre)
        pre[0, 1, FEATURE_INDEX["fio2"]] = True
        retained = filter_target_observed_transitions(
            transitions, {"pre_repair_masks": pre, "masks": final}
        )
        self.assertEqual(len(retained), 1)

    def test_cutpoints_precede_filters_and_kdd201_only_subsets(self) -> None:
        transitions = pd.DataFrame([
            {"episode_idx": 0, "action_idx": i, "target_idx": i + 1, "role": "train", "anchor_source": "structured_ventilation_chart_event", "relative_transition": i - 1}
            for i in range(1, 5)
        ] + [{"episode_idx": 1, "action_idx": 1, "target_idx": 2, "role": "validation", "anchor_source": "structured_ventilation_chart_event", "relative_transition": 0}])
        shape = (2, 6)
        peep = np.full(shape, np.nan); fio2 = np.full(shape, np.nan)
        peep[0, 1:5] = [1, 2, 3, 4]; fio2[0, 1:5] = [30, 40, 50, 60]
        peep[1, 1] = fio2[1, 1] = 999
        arrays = {"peep": peep, "peep_observed": np.isfinite(peep), "fio2_action": fio2, "fio2_action_observed": np.isfinite(fio2)}
        actions, contract = encode_actions("respiratory_support", transitions, arrays)
        self.assertEqual(contract["edges"], [(1.75, 2.5, 3.25), (37.5, 45.0, 52.5)])
        retained, retained_actions, removed = apply_kdd201_to_frozen_actions(transitions, actions)
        self.assertEqual(removed, 2)
        np.testing.assert_array_equal(retained_actions, actions[[1, 2, 3]])

    def test_expanded_schema_and_five_task_digest_surface(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mimic = make_fixture(Path(directory))
            output = Path(directory) / "output"
            receipt = reconstruct(mimic, output, SCHEMA, source_hashes={"fixture": "a" * 64}, chunk_rows=3)
            validator = Draft202012Validator(json.loads(SCHEMA.read_text()))
            validator.validate(receipt)
            self.assertEqual(receipt["schema_version"], "1.2.0")
            for task in receipt["tasks"]:
                self.assertEqual([row["role"] for row in task["role_summaries"]], ["train", "validation", "historical_other", "all_roles"])
                self.assertEqual(len(task["role_summaries"][-1]["digests"]), 17)
                self.assertEqual(task["role_summaries"][-1]["decisions"], task["decisions"])
                self.assertEqual(len(task["transition_stages"]["role_counts"]), 4)
            invalid = copy.deepcopy(receipt)
            invalid["tasks"][0]["role_summaries"][0]["subject_ids"] = [1]
            with self.assertRaises(ValidationError):
                validator.validate(invalid)

    def test_shock_uses_frozen_logical_source_chunks_not_reader_chunks(self) -> None:
        stays = pd.DataFrame([{
            "stay_id": 1, "intime": pd.Timestamp("2020-01-01"),
            "outtime": pd.Timestamp("2020-01-03"),
        }])
        chart = pd.DataFrame([
            {"stay_id": 1, "charttime": pd.Timestamp("2020-01-01 01:00"), "itemid": 220052, "valuenum": 60, "__source_order": 999_999},
            {"stay_id": 1, "charttime": pd.Timestamp("2020-01-01 02:00"), "itemid": 220052, "valuenum": 60, "__source_order": 1_000_000},
        ])
        split = sustained_hypotension_from_chart(chart, stays, authoritative_chunk_rows=1_000_000)
        self.assertTrue(split.empty)
        chart.loc[1, "__source_order"] = 999_998
        retained = sustained_hypotension_from_chart(chart, stays, authoritative_chunk_rows=1_000_000)
        self.assertEqual(retained["stay_id"].tolist(), [1])


if __name__ == "__main__":
    unittest.main()
