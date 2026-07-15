from __future__ import annotations

import json
import unittest

import numpy as np

from kdd_benchmark_discovery import kdd_e01_evaluator as e01
from kdd_benchmark_discovery import run_kdd_adapt01_adaptive_known_value as adapt
from kdd_benchmark_discovery import run_kdd_e02_known_value_full as e02
from kdd_benchmark_discovery import run_kdd_ope_rd01_repeated_dataset as rd01


class RepeatedDatasetOPEContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.adapt_config = json.loads(adapt.DEFAULT_CONFIG.read_text(encoding="utf-8"))

    def test_cached_nuisance_implementation_matches_frozen_e02(self) -> None:
        env, layout, _ = adapt.build_environment(
            "aki_rrt",
            self.adapt_config["tasks"]["aki_rrt"],
            "adaptive_composite",
            self.adapt_config["mechanisms"],
        )
        data = e01.generate_logged_data(env, n=32, seed=7401001)
        learned, _ = e02.fit_tabular_model(env, data)
        policy = rd01.target_policies(env, layout, "aki_rrt")["severity_rule"]
        denominator = e01.denominator_full_probabilities(env, data, "exact_behavior")
        bootstrap = e02._bootstrap_counts(32)
        q, v = e01.exact_qv(learned, policy, 2)
        expected = e02.ope_with_bootstrap(
            env, learned, data, policy, denominator, 2, 20.0, bootstrap
        )
        actual = rd01._ope_with_cached_nuisance(
            env, data, policy, denominator, 2, 20.0, bootstrap, q, v
        )
        self.assertEqual(set(actual), set(expected))
        for estimator in expected:
            np.testing.assert_allclose(
                actual[estimator], expected[estimator], rtol=1e-12, atol=1e-12, equal_nan=True
            )

    def test_target_policy_set_is_fixed_and_support_masked(self) -> None:
        expected = {
            "empirical_behavior",
            "random_supported",
            "minimum_supported_action",
            "maximum_supported_action",
            "severity_rule",
            "exact_dynamic_programming_oracle",
        }
        for task, profile in self.adapt_config["tasks"].items():
            env, layout, _ = adapt.build_environment(
                task, profile, "adaptive_composite", self.adapt_config["mechanisms"]
            )
            policies = rd01.target_policies(env, layout, task)
            self.assertEqual(set(policies), expected)
            for policy in policies.values():
                self.assertEqual(float(np.sum(policy * (~env.support)[None])), 0.0)

    def test_wilson_interval_and_rank_tie_handling(self) -> None:
        low, high = rd01.wilson_interval(180, 200)
        self.assertAlmostEqual(low, 0.8505941875672826)
        self.assertAlmostEqual(high, 0.9343295513309041)
        spearman, pairwise = rd01._rank_metrics(np.ones(3), np.array([0.0, 1.0, 2.0]))
        self.assertTrue(np.isnan(spearman))
        self.assertTrue(np.isnan(pairwise))


if __name__ == "__main__":
    unittest.main()
