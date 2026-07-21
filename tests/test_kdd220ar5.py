from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from kdd2027_benchmark.current_five_task.contracts import (
    FEATURE_INDEX,
    FEATURE_NAMES,
    apply_kdd201_temporal_repair,
    compact_lineage_role,
    extraction_post_hours,
    subject_role,
    validate_layout,
)
from kdd2027_benchmark.current_five_task.lineage_source_port import (
    blood_culture_events,
    compact_lineage_stays,
    sepsis_sofa_filter,
    kdd097_interval_bins,
    large_lineage_stays,
    match_suspected_infections,
)
from kdd2027_benchmark.current_five_task.reconstruct import (
    build_anchors,
    encode_actions,
    filter_target_observed_transitions,
    load_core,
    reconstruct,
)
from kdd2027_benchmark.current_five_task.runtime_config import load_runtime_config
from tests.test_kdd217ar3a import make_fixture


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "schemas" / "credentialed_aggregate_receipt.schema.json"


class KDD220AR5CurrentLineagePortTests(unittest.TestCase):
    def test_task_specific_stay_eligibility_has_no_unfrozen_filters(self) -> None:
        rows = pd.DataFrame({
            "subject_id": [1, 2, 3, 4], "anchor_age": [60, 60, 60, 17],
            "gender": ["F", None, "M", "F"],
            "dischtime": pd.to_datetime(["2020-01-04", "2020-01-04", None, "2020-01-04"]),
            "discharge_location": ["HOSPICE", "HOME", "HOME", "HOME"],
            "intime": pd.to_datetime(["2020-01-01"] * 4),
            "outtime": pd.to_datetime(["2020-01-03", "2020-01-03", "2020-01-03", "2020-01-03"]),
        })
        self.assertEqual(compact_lineage_stays(rows)["subject_id"].tolist(), [1, 2, 3])
        self.assertEqual(large_lineage_stays(rows)["subject_id"].tolist(), [1])
        self.assertEqual(large_lineage_stays(rows).iloc[0]["discharge_location"], "HOSPICE")

    def test_runtime_config_names_the_frozen_sepsis_manifest_gate(self) -> None:
        contract = load_runtime_config()["cohort_parameters"]["sepsis"]
        self.assertEqual(contract, {
            "source": "blood_culture_suspected_infection_manifest",
            "culture_scope": "blood",
            "maximum_observed_sofa_min": 2,
            "preserve_suspected_infection_onset": True,
        })

    def test_e060_blood_match_preserves_prior_culture_anchor_shift(self) -> None:
        events = pd.DataFrame([
            {"micro_specimen_id": 1, "subject_id": 11, "hadm_id": 21, "chartdate": pd.Timestamp("2020-01-01"), "charttime": pd.Timestamp("2020-01-01 04:00"), "spec_type_desc": "URINE", "org_itemid": 1, "org_name": "synthetic"},
            {"micro_specimen_id": 2, "subject_id": 11, "hadm_id": 21, "chartdate": pd.Timestamp("2020-01-01"), "charttime": pd.Timestamp("2020-01-01 08:00"), "spec_type_desc": "BLOOD CULTURE", "org_itemid": 2, "org_name": "synthetic"},
        ])
        cultures = blood_culture_events(events)
        self.assertEqual(cultures["micro_specimen_id"].tolist(), [2])
        antibiotics = pd.DataFrame([{"subject_id": 11, "hadm_id": 21, "stay_id": 31, "antibiotic_time": pd.Timestamp("2020-01-02 00:00")}])
        matched = match_suspected_infections(antibiotics, cultures)
        self.assertEqual(matched.iloc[0]["suspected_infection_time"], pd.Timestamp("2020-01-01 08:00"))

    def test_e060_sofa_gate_preserves_onset_and_excludes_hf_overlap(self) -> None:
        candidates = pd.DataFrame([
            {"task_id": "sepsis", "stay_id": 1, "episode_idx": 0, "anchor_time": pd.Timestamp("2020-01-02")},
            {"task_id": "heart_failure", "stay_id": 1, "episode_idx": 1, "anchor_time": pd.Timestamp("2020-01-02")},
            {"task_id": "sepsis", "stay_id": 2, "episode_idx": 2, "anchor_time": pd.Timestamp("2020-01-02")},
            {"task_id": "heart_failure", "stay_id": 3, "episode_idx": 3, "anchor_time": pd.Timestamp("2020-01-02")},
        ])
        values = np.full((4, 18, len(FEATURE_NAMES)), np.nan)
        masks = np.zeros_like(values, dtype=bool)
        values[0, 4, FEATURE_INDEX["sofa_proxy"]] = 2
        masks[0, 4, FEATURE_INDEX["sofa_proxy"]] = True
        values[2, 4, FEATURE_INDEX["sofa_proxy"]] = 1
        masks[2, 4, FEATURE_INDEX["sofa_proxy"]] = True
        result = sepsis_sofa_filter(candidates, values, masks, sofa_index=FEATURE_INDEX["sofa_proxy"])
        self.assertEqual(result[["task_id", "stay_id"]].to_records(index=False).tolist(), [("sepsis", 1), ("heart_failure", 3)])
        self.assertEqual(result.iloc[0]["anchor_time"], pd.Timestamp("2020-01-02"))

    def test_respiratory_action_is_independent_of_safe_feature_mask(self) -> None:
        transitions = pd.DataFrame([{"episode_idx": 0, "action_idx": 1, "role": "train"}])
        values = np.full((1, 3, len(FEATURE_NAMES)), np.nan)
        masks = np.zeros_like(values, dtype=bool)
        arrays = {
            "values": values, "masks": masks,
            "peep": np.array([[np.nan, 8.0, np.nan]]),
            "peep_observed": np.array([[False, True, False]]),
            "fio2_action": np.array([[np.nan, 40.0, np.nan]]),
            "fio2_action_observed": np.array([[False, True, False]]),
        }
        action, contract = encode_actions("respiratory_support", transitions, arrays)
        self.assertTrue(0 <= int(action[0]) < 25)
        self.assertEqual(contract["valid_action_mask"].tolist(), [True])
        self.assertFalse(masks[0, 1, FEATURE_INDEX["fio2"]])

    def test_target_filter_uses_any_safe_forecasting_observation(self) -> None:
        transitions = pd.DataFrame([
            {"episode_idx": 0, "target_idx": 1},
            {"episode_idx": 0, "target_idx": 2},
        ])
        masks = np.zeros((1, 3, len(FEATURE_NAMES)), dtype=bool)
        masks[0, 2, FEATURE_INDEX["heart_rate"]] = True
        retained = filter_target_observed_transitions(transitions, {"masks": masks})
        self.assertEqual(retained["target_idx"].tolist(), [2])

    def test_kdd201_removes_only_named_disposition_three_anchor_bins(self) -> None:
        values = (0, 1, 2)
        for source in (
            "structured_ventilation_chart_event", "structured_respiratory_support_event",
            "vasopressor_support", "time_stamped_rrt_start",
            "current_stay_decongestion_prescription", "current_stay_decongestion_input",
        ):
            np.testing.assert_array_equal(apply_kdd201_temporal_repair(source, values), [False, True, True])
        for source in ("sustained_hypotension", "creatinine_delta_48h", "creatinine_ratio_7d", "E060_suspected_infection_plus_SOFA_ge2_scaffold"):
            np.testing.assert_array_equal(apply_kdd201_temporal_repair(source, values), [True, True, True])

    def test_kdd097_interval_end_bin_and_temporal_contract(self) -> None:
        start = pd.Timestamp("2020-01-01 01:00")
        end = pd.Timestamp("2020-01-01 04:00")
        self.assertEqual(list(kdd097_interval_bins(start, end, pd.Timestamp("2020-01-01"), bin_hours=4, n_steps=30)), [0, 1])
        self.assertEqual(extraction_post_hours("respiratory_support"), 96)
        self.assertEqual(extraction_post_hours("sepsis"), 48)

    def test_subject_role_scopes_are_frozen(self) -> None:
        subjects = np.arange(1, 250)
        large = [subject_role(int(value)) for value in subjects]
        compact = [compact_lineage_role(int(value)) for value in subjects]
        self.assertTrue(set(large) <= {"train", "validation", "historical_other"})
        self.assertTrue(set(compact) <= {"train", "validation", "historical_other"})
        self.assertTrue(any(left != right for left, right in zip(large, compact)))

    def test_five_task_fixture_includes_hf_without_prior_diagnosis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mimic = make_fixture(Path(directory))
            diagnoses = mimic / "hosp" / "diagnoses_icd.csv.gz"
            frame = pd.read_csv(diagnoses).iloc[0:0]
            frame.to_csv(diagnoses, index=False, compression="gzip")
            paths = validate_layout(mimic)
            stays, loaded_diagnoses = load_core(paths)
            candidates = build_anchors(paths, stays, loaded_diagnoses, chunk_rows=3)
            hf = candidates[candidates["task_id"].eq("heart_failure")]
            self.assertFalse(hf.empty)
            self.assertTrue(hf["anchor_source"].isin({"current_stay_decongestion_prescription", "current_stay_decongestion_input"}).all())
            output = Path(directory) / "output"
            receipt = reconstruct(mimic, output, SCHEMA, chunk_rows=3)
            self.assertEqual([row["task_id"] for row in receipt["tasks"]], ["sepsis", "respiratory_support", "shock", "aki", "heart_failure"])
            self.assertEqual([row["action_count"] for row in receipt["tasks"]], [25, 25, 25, 4, 2])


if __name__ == "__main__":
    unittest.main()
