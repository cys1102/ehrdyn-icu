from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from kdd_benchmark_discovery import run_kdd107_heterogeneous_known_value as k107
from kdd_benchmark_discovery import run_kdd115_heterogeneous_repeated_dataset_ope as k115


class KDD115ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(k115.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        cls.k107_config = json.loads(k115.KDD107_CONFIG.read_text(encoding="utf-8"))

    def test_frozen_sources_and_control_policy_truth_match(self) -> None:
        parity = k115.verify_frozen_contract(self.config)
        self.assertEqual(len(parity["policy_parity_rows"]), 80)
        self.assertTrue(all(row["exact_return_absolute_difference"] <= 1e-12 for row in parity["policy_parity_rows"]))

    def test_six_target_policies_are_support_masked(self) -> None:
        expected = set(self.config["target_policies"])
        for task in self.config["logged_dataset_replicates"]:
            env = k107.build_environment(task, "state_dependent_optimum", self.k107_config)
            policies = k115.target_policies(env, task, self.k107_config)
            self.assertEqual(set(policies), expected)
            for policy in policies.values():
                self.assertEqual(float(np.sum(policy * (~env.support)[None])), 0.0)

    def test_new_logged_dataset_seeds_are_unique_and_disjoint(self) -> None:
        seeds = k115.seed_manifest(self.config)
        self.assertEqual(len(seeds), 16)
        overlap = seeds[[
            "prior_logged_dataset_overlap",
            "optimization_seed_overlap",
            "evaluation_seed_overlap",
            "bootstrap_seed_overlap",
        ]]
        self.assertFalse(overlap.to_numpy().any())
        all_values = []
        for row in seeds.itertuples():
            all_values.extend(range(row.logged_dataset_seed_start, row.logged_dataset_seed_end + 1))
        self.assertEqual(len(all_values), len(set(all_values)))
        self.assertEqual(len(all_values), 4400)

    def test_single_cell_smoke_has_exact_grid_and_repeat_count(self) -> None:
        dataset_rows, policy_rows = k115._cell_worker(
            "aki_rrt", "interior_optimum", 2, 0, 2, self.config, self.k107_config
        )
        self.assertEqual(len(dataset_rows), 2 * 3 * 9 * 4 * 2)
        self.assertEqual(len(policy_rows), 3 * 9 * 4 * 2 * 6)
        self.assertEqual({row["trials"] for row in policy_rows}, {2})
        self.assertEqual({row["maximum_unsupported_action_mass"] for row in dataset_rows}, {0.0})

    def test_wilson_zero_success_roundoff_does_not_fail_interval_parity(self) -> None:
        low, high = k115.rd01.wilson_interval(0, 200)
        self.assertGreaterEqual(low, 0.0)
        frame = pd.DataFrame([{
            "repeated_dataset_empirical_coverage": 0.0,
            "binomial_ci_low": low,
            "binomial_ci_high": high,
        }])
        self.assertTrue(k115.interval_parity_valid(frame))


if __name__ == "__main__":
    unittest.main()
