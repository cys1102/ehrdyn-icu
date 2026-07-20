from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from kdd2027_benchmark.entrant_runtime import IsolatedEntrant, ResourceContract, validate_policy_result
from kdd2027_benchmark.errors import ReleaseContractError
from kdd2027_benchmark.full_direct_evaluator import evaluate_repaired_policy_batch
from kdd2027_benchmark.full_ope import bootstrap_policy_groups, collect_observed_history_dataset, target_probabilities
from kdd2027_benchmark.full_suite import (
    DATASETS_PER_ENVIRONMENT,
    ENVIRONMENT_SEEDS,
    EPISODES_PER_DATASET,
    PROFILES,
    EntrantPolicy,
    environments,
    generate_full_suite,
    validate_entrant_conformance,
    _exogenous_seed,
    fixed_policy,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/full_benchmark/kdd198_v2_generator_contract.json"
POLICY = ROOT / "policy_entrant_example/entrant.json"
COMPONENT = ROOT / "component_entrant_example/entrant.json"
INVALID = ROOT / "invalid_entrant_fixtures/invalid.py"


class KDD215FullSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.environments = environments(MANIFEST)

    def test_exact_environment_and_dataset_nesting(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = generate_full_suite(MANIFEST, Path(directory) / "manifest.csv")
        self.assertEqual(receipt["environment_count"], 40)
        self.assertEqual(receipt["logged_dataset_count"], 320)
        self.assertEqual(len(PROFILES), 5)
        self.assertEqual(len(ENVIRONMENT_SEEDS), 8)
        self.assertEqual(DATASETS_PER_ENVIRONMENT, 8)
        self.assertEqual(EPISODES_PER_DATASET, 256)
        self.assertEqual(_exogenous_seed(4, 171903, 1), 1_993_000_000 + 4_000_000 + 171_903 + 100_000_000)

    def test_public_train_validation_cache_excludes_latent_channels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = generate_full_suite(MANIFEST, root / "manifest.csv", root / "cache")
            files = sorted((root / "cache").rglob("*.npz"))
            self.assertEqual(len(files), 80)
            with np.load(files[0]) as data:
                self.assertEqual(set(data.files), {
                    "observations", "masks", "recency", "actions", "next_observations",
                    "behavior_probabilities", "rewards", "terminal", "valid",
                })
            self.assertFalse(receipt["cache_contains_latent_state_or_subtype"])

    def test_same_payload_reconstructs_and_seed_changes_mechanism(self):
        first = environments(MANIFEST)
        second = environments(MANIFEST)
        self.assertEqual([item.mechanism_hash for item in first], [item.mechanism_hash for item in second])
        for profile in PROFILES:
            hashes = {item.mechanism_hash for item in first if item.contract.profile == profile}
            self.assertEqual(len(hashes), 8)

    def test_supported_runtime_probe_hash_is_frozen(self):
        expected = ROOT / "fixtures/kdd215_expected/runtime_probe.json"
        self.assertEqual(hashlib.sha256(expected.read_bytes()).hexdigest(),
                         "3ba2debdd7c6d6b9c6ec0f681ea6d08685480f83e1fc56cc7abbfda77b179763")

    def test_policy_and_component_examples_pass_isolated_contract(self):
        self.assertTrue(validate_entrant_conformance(POLICY, MANIFEST)["pass"])
        self.assertTrue(validate_entrant_conformance(COMPONENT, MANIFEST)["pass"])

    def test_entrant_declaration_schema_rejects_unknown_and_false_acknowledgement(self):
        declaration = json.loads(POLICY.read_text())
        for mutation, message in (({"unknown": 1}, "additionalProperties"),
                                  ({"support_handling_acknowledged": False}, "const")):
            with tempfile.TemporaryDirectory() as directory:
                value = dict(declaration); value.update(mutation)
                path = Path(directory) / "entrant.json"
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaisesRegex(ReleaseContractError, message):
                    with IsolatedEntrant(path):
                        pass

    def test_latent_and_future_information_are_not_in_runtime_payload(self):
        env = self.environments[0]
        with IsolatedEntrant(POLICY) as process:
            adapter = EntrantPolicy(process, env, 1)
            result = adapter(
                np.zeros((2, env.contract.feature_dim)),
                np.ones((2, env.contract.feature_dim), dtype=bool),
                np.zeros((2, env.contract.feature_dim)),
                np.zeros(2, dtype=int), 0,
            )
        self.assertEqual(result.shape, (2, env.contract.action_count))
        self.assertNotIn("latent_state", POLICY.read_text())
        self.assertNotIn("subtype", POLICY.read_text())
        self.assertNotIn("future", POLICY.read_text())

    def test_small_direct_rerun_is_identical_and_terminal_once(self):
        env = self.environments[0]
        values = []
        for _ in range(2):
            with IsolatedEntrant(POLICY) as process:
                result = evaluate_repaired_policy_batch(env, EntrantPolicy(process, env, 7), 32, 1993171901, 2993000000)
            self.assertLessEqual(result["terminal_emission_max"], 1)
            values.append(result["returns"])
        self.assertTrue(np.array_equal(values[0], values[1]))

    def test_ope_bootstrap_is_worker_count_invariant(self):
        env = self.environments[0]
        data = collect_observed_history_dataset(env, 32, 2_022_100_000, "ehr_matched")
        target = target_probabilities(data, fixed_policy(env, "behavior"))
        arguments = (data, {"behavior": [target]}, 4, 77, "crossfit_stronger", None, 5, 0.5, 0.5)
        one = bootstrap_policy_groups(*arguments, workers=1)
        two = bootstrap_policy_groups(*arguments, workers=2)
        for estimator in one["behavior"]:
            self.assertTrue(np.array_equal(one["behavior"][estimator], two["behavior"][estimator]))

    def test_probability_failure_classes(self):
        with self.assertRaisesRegex(ReleaseContractError, "dimension"):
            validate_policy_result({"probabilities": [[0.5]]}, 1, 2, [0, 1])
        for values, message in (([float("nan"), 0.0], "nonfinite"), ([-0.1, 1.1], "negative"), ([0.2, 0.2], "normalization")):
            with self.assertRaisesRegex(ReleaseContractError, message):
                validate_policy_result({"probabilities": [values]}, 1, 2, [0, 1])
        with self.assertRaisesRegex(ReleaseContractError, "unsupported"):
            validate_policy_result({"probabilities": [[0.0, 1.0]]}, 1, 2, [0])

    def test_crashing_malformed_slow_and_nondeterministic_fixtures(self):
        expected = {
            "crash": "crashed", "malformed": "malformed", "slow": "timeout",
        }
        for mode, message in expected.items():
            with self.subTest(mode=mode), self._invalid(mode) as declaration:
                with self.assertRaisesRegex(ReleaseContractError, message):
                    with IsolatedEntrant(declaration, ResourceContract(timeout_seconds=0.05)) as process:
                        process.request("predict_policy", {"observations": [[0]], "action_count": 2}, 1)
        with self._invalid("nondeterministic") as declaration:
            with IsolatedEntrant(declaration) as process:
                payload = {"observations": [[0]], "action_count": 2}
                one = process.request("predict_policy", payload, 1)["result"]
                two = process.request("predict_policy", payload, 1)["result"]
            self.assertNotEqual(one, two)
        respiratory = next(env for env in self.environments if env.contract.profile == "respiratory")
        with self._invalid("unsupported") as declaration:
            with self.assertRaisesRegex(ReleaseContractError, "unsupported"):
                with IsolatedEntrant(declaration) as process:
                    EntrantPolicy(process, respiratory, 1)(
                        np.zeros((1, respiratory.contract.feature_dim)),
                        np.ones((1, respiratory.contract.feature_dim), dtype=bool),
                        np.zeros((1, respiratory.contract.feature_dim)),
                        np.asarray([respiratory.supported[0]]), 0,
                    )

    def _invalid(self, mode: str):
        class Fixture:
            def __init__(self):
                self.temp = tempfile.TemporaryDirectory()
                root = Path(self.temp.name)
                (root / "invalid.py").write_bytes(INVALID.read_bytes())
                declaration = {
                    "schema_version": "kdd215.entrant.v1", "entrant_id": f"invalid_{mode}",
                    "command": ["/usr/bin/env", f"INVALID_MODE={mode}", "/usr/bin/python3", "/entrant/invalid.py"],
                    "capabilities": ["policy_probability"], "deterministic": True,
                    "support_handling_acknowledged": True,
                }
                self.path = root / "entrant.json"
                self.path.write_text(json.dumps(declaration), encoding="utf-8")
            def __enter__(self): return self.path
            def __exit__(self, *_): self.temp.cleanup()
        return Fixture()


if __name__ == "__main__":
    unittest.main()
