from __future__ import annotations

import json
import unittest

import numpy as np

from kdd_benchmark_discovery import run_kdd_adapt01_adaptive_known_value as adapt
from kdd_benchmark_discovery import run_kdd_bridge01_ehr_known_value as bridge


class EHRKnownValueBridgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(bridge.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        cls.adapt_config = json.loads(adapt.DEFAULT_CONFIG.read_text(encoding="utf-8"))

    def test_frozen_sources_and_model_family_map(self) -> None:
        bridge.verify_sources(self.config)
        self.assertEqual(len(self.config["shared_model_family_map"]), 4)
        self.assertFalse(self.config["task_map"]["aki"]["primary_bridge_eligible"])
        self.assertFalse(self.config["task_map"]["heart_failure"]["primary_bridge_eligible"])

    def test_same_capacity_fit_is_newly_initialized(self) -> None:
        profile = self.adapt_config["tasks"]["aki_rrt"]
        env, _, features = adapt.build_environment(
            "aki_rrt", profile, "adaptive_composite", self.adapt_config["mechanisms"]
        )
        train, _ = adapt.logged_offline(env, features, 8, 81101, float(profile["missingness"]))
        validation, _ = adapt.logged_offline(env, features, 4, 81102, float(profile["missingness"]))
        budget = dict(self.config["known_value_training"])
        budget.update(max_epochs=2, min_epochs=1, patience=1, batch_size=8)
        fit, receipt = bridge.fit_known_value_world_model(
            "grud_world_model", train, validation, env.n_states, env.n_actions, 3408, budget, 2
        )
        self.assertFalse(receipt["ehr_weights_reused"])
        self.assertEqual(fit.model.cell.hidden_size, 48)
        self.assertEqual(len(fit.fingerprint), 64)

    def test_exploratory_correlation_handles_ties(self) -> None:
        import pandas as pd

        frame = pd.DataFrame(
            {
                "x": [1.0, 1.0, 2.0, 3.0],
                "y": [3.0, 2.0, 1.0, 0.0],
                "primary_bridge_eligible": [True, True, False, False],
            }
        )
        rows = bridge._correlation_rows(frame, "x", "y", "test", self.config)
        self.assertEqual(rows[0]["sample_unit"], "task_model_family")
        self.assertFalse(rows[0]["confirmatory_inference"])
        self.assertTrue(np.isnan(rows[1]["spearman_rho"]))


if __name__ == "__main__":
    unittest.main()
