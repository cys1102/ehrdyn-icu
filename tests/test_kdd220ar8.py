from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from kdd2027_benchmark.current_five_task.contracts import validate_layout
from kdd2027_benchmark.current_five_task.reconstruct import (
    _first_anchor,
    _interval_filter,
    build_anchors,
    classify_icu_time_eligibility,
    filter_respiratory_action_transitions,
    load_core_with_time_audit,
)
from tests.test_kdd217ar3a import make_fixture


class KDD220AR8LocalizedRespiratorySourceRepairTests(unittest.TestCase):
    def test_positive_input_oxygenation_support_is_a_respiratory_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = make_fixture(Path(directory))
            items_path = root / "icu" / "d_items.csv.gz"
            items = pd.read_csv(items_path)
            items.loc[len(items)] = {
                "itemid": 999003,
                "label": "High Flow Nasal Cannula",
                "abbreviation": "HFNC",
                "linksto": "inputevents",
                "category": "Resp",
                "unitname": "L/min",
            }
            items.to_csv(items_path, index=False, compression="gzip")

            procedures_path = root / "icu" / "procedureevents.csv.gz"
            procedures = pd.read_csv(procedures_path)
            procedures = procedures.loc[~procedures["itemid"].eq(225792)]
            procedures.to_csv(procedures_path, index=False, compression="gzip")

            inputs_path = root / "icu" / "inputevents.csv.gz"
            inputs = pd.read_csv(inputs_path)
            respiratory_stay = 2002
            respiratory = inputs.loc[inputs["stay_id"].eq(respiratory_stay)].iloc[0].copy()
            respiratory["itemid"] = 999003
            respiratory["starttime"] = "2020-01-02 00:00:00"
            respiratory["endtime"] = "2020-01-02 04:00:00"
            respiratory["amount"] = 1.0
            respiratory["rate"] = 1.0
            inputs = pd.concat([inputs, respiratory.to_frame().T], ignore_index=True)
            inputs.to_csv(inputs_path, index=False, compression="gzip")

            paths = validate_layout(root)
            stays, diagnoses, _ = load_core_with_time_audit(paths, {})
            candidates = build_anchors(paths, stays, diagnoses, chunk_rows=3, audit={})
            respiratory_candidates = candidates.loc[
                candidates["task_id"].eq("respiratory_support")
                & candidates["stay_id"].eq(respiratory_stay)
            ]
            self.assertEqual(len(respiratory_candidates), 1)
            row = respiratory_candidates.iloc[0]
            self.assertEqual(row["anchor_source"], "structured_respiratory_support_event")
            self.assertEqual(row["anchor_time"], pd.Timestamp("2020-01-02 00:00:00"))

    def test_anchor_boundaries_duplicates_invalid_stays_and_missing_actions(self) -> None:
        anchors = pd.DataFrame([
            {"stay_id": 1, "anchor_time": pd.Timestamp("2020-01-01 04:00"), "anchor_source": "later_source"},
            {"stay_id": 1, "anchor_time": pd.Timestamp("2020-01-01 04:00"), "anchor_source": "stable_first_source"},
            {"stay_id": 1, "anchor_time": pd.Timestamp("2020-01-01 08:00"), "anchor_source": "latest_source"},
        ])
        selected = _first_anchor(anchors)
        self.assertEqual(selected.iloc[0]["anchor_source"], "later_source")

        bounded_stays = pd.DataFrame([{
            "stay_id": 1,
            "intime": pd.Timestamp("2020-01-01 00:00"),
            "outtime": pd.Timestamp("2020-01-02 00:00"),
        }])
        boundary_events = pd.DataFrame([
            {"stay_id": 1, "starttime": pd.Timestamp("2020-01-01 00:00"), "source_order": 0},
            {"stay_id": 1, "starttime": pd.Timestamp("2020-01-01 00:00"), "source_order": 1},
            {"stay_id": 1, "starttime": pd.Timestamp("2020-01-02 00:00"), "source_order": 2},
        ])
        bounded = _interval_filter(boundary_events, bounded_stays, "starttime")
        self.assertEqual(bounded["source_order"].tolist(), [0, 1])

        stays = pd.DataFrame([
            {"intime": pd.Timestamp("2020-01-01"), "outtime": pd.Timestamp("2020-01-02")},
            {"intime": pd.NaT, "outtime": pd.Timestamp("2020-01-02")},
            {"intime": pd.Timestamp("2020-01-01"), "outtime": pd.NaT},
            {"intime": pd.Timestamp("2020-01-01"), "outtime": pd.Timestamp("2020-01-01")},
            {"intime": pd.Timestamp("2020-01-02"), "outtime": pd.Timestamp("2020-01-01")},
        ])
        retained, audit = classify_icu_time_eligibility(stays)
        self.assertEqual(len(retained), 1)
        self.assertEqual(
            {key: audit[key] for key in ("missing_intime", "missing_outtime", "equal_intime_outtime", "reversed_time_order")},
            {"missing_intime": 1, "missing_outtime": 1, "equal_intime_outtime": 1, "reversed_time_order": 1},
        )

        transitions = pd.DataFrame([
            {"episode_idx": 1, "relative_transition": 0, "state_idx": 0, "action_idx": 1, "target_idx": 2},
            {"episode_idx": 1, "relative_transition": 1, "state_idx": 1, "action_idx": 2, "target_idx": 3},
        ])
        filtered, actions, counts = filter_respiratory_action_transitions(
            transitions, np.asarray([-1, 7], dtype=np.int16), np.asarray([False, True])
        )
        self.assertEqual(filtered["relative_transition"].tolist(), [1])
        self.assertEqual(actions.tolist(), [7])
        self.assertEqual(counts["excluded_missing_action_transitions"], 1)


if __name__ == "__main__":
    unittest.main()
