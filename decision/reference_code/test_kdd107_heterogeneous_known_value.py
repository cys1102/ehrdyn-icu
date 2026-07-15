from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from kdd_benchmark_discovery import kdd_e01_evaluator as e01
from kdd_benchmark_discovery.run_kdd107_heterogeneous_known_value import (
    DEFAULT_CONFIG,
    MECHANISM_FAMILIES,
    build_environment,
    learned_planner,
    mechanism_audit,
    preflight,
)


class KDD107EnvironmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(Path(DEFAULT_CONFIG).read_text(encoding="utf-8"))

    def test_frozen_preflight_passes(self) -> None:
        checks = preflight(self.config)
        self.assertTrue(checks)
        self.assertTrue(all(row["pass"] for row in checks))

    def test_all_environment_and_mechanism_gates_pass(self) -> None:
        for task in self.config["task_profiles"]:
            for mechanism in MECHANISM_FAMILIES:
                with self.subTest(task=task, mechanism=mechanism):
                    env = build_environment(task, mechanism, self.config)
                    self.assertTrue(np.allclose(env.transition.sum(axis=-1), 1.0))
                    self.assertTrue(np.allclose(env.behavior.sum(axis=-1), 1.0))
                    self.assertFalse(np.any(env.behavior[:, ~env.support] > 0))
                    audit = mechanism_audit(task, mechanism, env, self.config)
                    self.assertTrue(audit["support_valid"])
                    self.assertTrue(audit["mechanism_property_pass"])
                    self.assertGreater(audit["weak_overlap_strata"], 0)
                    self.assertGreater(audit["strong_overlap_strata"], 0)

    def test_null_response_is_exactly_action_invariant(self) -> None:
        tolerance = float(self.config["evaluation_contract"]["null_tolerance"])
        for task in self.config["task_profiles"]:
            with self.subTest(task=task):
                env = build_environment(task, "null_response", self.config)
                policies = [env.behavior, e01.h1_exhaustive_policy(env), e01.backward_induction(env)[1]]
                values = [e01.evaluate_policy_exact(env, policy) for policy in policies]
                self.assertLessEqual(max(values) - min(values), tolerance)

    def test_oracle_never_has_negative_regret(self) -> None:
        tolerance = float(self.config["evaluation_contract"]["negative_regret_tolerance"])
        for task in self.config["task_profiles"]:
            for mechanism in MECHANISM_FAMILIES:
                with self.subTest(task=task, mechanism=mechanism):
                    env = build_environment(task, mechanism, self.config)
                    oracle, _, _ = e01.backward_induction(env)
                    for policy in (env.behavior, e01.h1_exhaustive_policy(env)):
                        self.assertGreaterEqual(
                            oracle - e01.evaluate_policy_exact(env, policy), -tolerance
                        )

    def test_cem_handles_binary_weak_overlap_with_state_specific_masks(self) -> None:
        env = build_environment("aki_rrt", "state_dependent_optimum", self.config)
        expected_transition = np.sum(
            env.transition * np.arange(env.n_states, dtype=np.float64)[None, None, None, :],
            axis=-1,
        )
        next_state = np.rint(expected_transition.mean(axis=0)).astype(int)
        reward = np.sum(env.transition * env.reward, axis=-1).mean(axis=0)
        uncertainty = np.zeros_like(reward)
        policy, audit = learned_planner(
            next_state,
            reward,
            uncertainty,
            env,
            4,
            False,
            3408,
            float(self.config["planner_contract"]["uncertainty_penalty"]),
        )
        self.assertEqual(audit["iterations"], 3)
        self.assertEqual(audit["candidates"], 64)
        self.assertEqual(audit["elite_count"], 8)
        self.assertGreater(audit["minimum_unique_sequences"], 1)
        self.assertFalse(audit["support_mask_bypass"])
        self.assertFalse(np.any(policy * (~env.support)[None]))


if __name__ == "__main__":
    unittest.main()
