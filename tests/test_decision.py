from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from kdd2027_benchmark.decision import run_smoke, validate_decision_release
from kdd2027_benchmark.decision.ope import (
    per_decision_is,
    trajectory_is,
    weighted_is,
    weighted_per_decision_is,
)


ROOT = Path(__file__).resolve().parents[1]


class DecisionReleaseTests(unittest.TestCase):
    def test_release_manifest_and_paper_counts_validate(self):
        receipt = validate_decision_release(ROOT)
        self.assertTrue(receipt["pass"])
        self.assertEqual(receipt["complete_transition_rows"], 36)
        self.assertEqual(receipt["current_transition_rows"], 18)
        self.assertEqual(receipt["scale_qualified_cohorts"], 6)
        self.assertEqual(receipt["current_numeric_cohorts"], 3)
        self.assertEqual(receipt["adaptive_policy_summary_rows"], 136)
        self.assertEqual(receipt["adaptive_policy_seed_rows"], 700)
        self.assertEqual(receipt["heterogeneous_policy_seed_rows"], 4080)
        self.assertEqual(receipt["heterogeneous_planner_rows"], 2880)
        self.assertEqual(receipt["heterogeneous_discriminative_cells"], 11)
        self.assertEqual(receipt["current_policy_seed_rows"], 3060)
        self.assertEqual(receipt["current_planner_rows"], 2160)
        self.assertEqual(receipt["current_discriminative_cells"], 8)
        self.assertEqual(receipt["historical_known_value_policy_rows"], 2448)
        self.assertEqual(receipt["historical_ope_tuple_rows"], 16128)
        self.assertEqual(receipt["repeated_dataset_ope_tuples"], 1728)
        self.assertEqual(receipt["current_repeated_dataset_ope_tuples"], 1296)
        self.assertEqual(receipt["current_heterogeneous_repeated_dataset_ope_tuples"], 2592)
        self.assertEqual(receipt["repeated_dataset_adaptive_approved"], 0)
        self.assertEqual(receipt["repeated_dataset_null_approved"], 40)
        self.assertEqual(receipt["aki_task_matched_tier2_approved"], 236)
        self.assertEqual(receipt["retrospective_ehr_policy_value"], "not_executed")

    def test_contract_has_six_world_model_and_four_policy_tasks(self):
        contract = json.loads((ROOT / "decision/contracts/task_contracts.json").read_text(encoding="utf-8"))
        self.assertEqual(len(contract["ehr_world_model_tasks"]), 6)
        self.assertEqual(len(contract["known_value_policy_tasks"]), 4)
        self.assertEqual(
            {row["task"] for row in contract["known_value_policy_tasks"]},
            {"respiratory", "shock", "aki_rrt", "heart_failure"},
        )

    def test_synthetic_smoke_checks_null_crn_planner_and_ope(self):
        receipt = run_smoke(seed=7)
        self.assertTrue(receipt["pass"])
        self.assertEqual(receipt["null_response_maximum_paired_return_gap"], 0.0)
        planner = receipt["planner_audit"]
        self.assertEqual(planner["iterations"], 3)
        self.assertEqual(planner["candidate_sequences"], 64)
        self.assertFalse(planner["support_mask_bypass"])
        self.assertTrue(planner["receding_horizon_first_action_only"])

    def test_ope_estimators_reduce_to_on_policy_sample_return(self):
        rewards = np.array([[1.0, 0.5], [0.0, 1.0], [0.25, 0.25]], dtype=np.float64)
        probabilities = np.full_like(rewards, 0.5)
        expected = float(np.mean(rewards @ np.array([1.0, 0.99])))
        for estimator in (trajectory_is, weighted_is, per_decision_is, weighted_per_decision_is):
            result = estimator(rewards, probabilities, probabilities)
            self.assertAlmostEqual(result["estimate"], expected)
            self.assertAlmostEqual(result["ess"], 3.0)


if __name__ == "__main__":
    unittest.main()
