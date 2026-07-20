from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from kdd2027_benchmark.errors import ReleaseContractError
from kdd2027_benchmark.public_bundle import rebuild_public_bundle
from kdd2027_benchmark.public_ope import ESTIMATORS, run_public_ope_smoke
from kdd2027_benchmark.public_pomdp import PublicPOMDP, PublicPOMDPConfig, run_public_pomdp_smoke
from kdd2027_benchmark.transition_entrant import validate_transition_submission


ROOT = Path(__file__).resolve().parents[1]
POMDP_CONFIG = ROOT / "configs/public_pomdp/kdd198_repaired_v2.json"


class KDD212ReleaseTests(unittest.TestCase):
    def test_environment_seed_reconstruction_and_separation(self):
        config = PublicPOMDPConfig.load(POMDP_CONFIG, "aki")
        first = PublicPOMDP(config, 21201)
        repeated = PublicPOMDP(config, 21201)
        different = PublicPOMDP(config, 21202)
        self.assertEqual(first.mechanism_hash, repeated.mechanism_hash)
        self.assertNotEqual(first.mechanism_hash, different.mechanism_hash)

    def test_public_pomdp_smoke_is_deterministic_and_multistep(self):
        first = run_public_pomdp_smoke(POMDP_CONFIG, "aki", 21201, 16, 3408)
        second = run_public_pomdp_smoke(POMDP_CONFIG, "aki", 21201, 16, 3408)
        self.assertEqual(first, second)
        self.assertTrue(first["reconstruction_hash_equal"])
        trace = first["planner_trace"]
        self.assertEqual((trace["horizon"], trace["cem_iterations"]), (4, 3))
        self.assertEqual((trace["candidates_per_iteration"], trace["elites"]), (64, 8))
        self.assertGreater(trace["sequences_evaluated"], trace["executed_actions"])

    def test_public_ope_smoke_refits_and_is_deterministic(self):
        first = run_public_ope_smoke(POMDP_CONFIG, "aki", 21201, 2, 16, 2, 3411)
        second = run_public_ope_smoke(POMDP_CONFIG, "aki", 21201, 2, 16, 2, 3411)
        self.assertEqual(first, second)
        self.assertEqual(tuple(first["estimators"]), ESTIMATORS)
        self.assertTrue(first["nuisance_refit_inside_each_bootstrap"])
        self.assertEqual(len(first["rows"]), 2 * len(ESTIMATORS))
        self.assertTrue(all(row["finite_fraction"] == 1.0 for row in first["rows"]))

    def test_frozen_smoke_files_match_documented_hashes(self):
        receipt = json.loads((ROOT / "fixtures/kdd212_public_smoke_hashes.json").read_text())
        for relative, expected in receipt["fixtures"].items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), expected)

    def test_transition_fixture_verifies_task_and_version_hashes(self):
        result = validate_transition_submission(
            ROOT / "fixtures/transition_submission_small.json", ROOT / "configs/tasks"
        )
        self.assertEqual(result["valid_rows"], 1)
        fixture = json.loads((ROOT / "fixtures/transition_submission_small.json").read_text())
        fixture["rows"][0]["task_config_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "bad.json"
            bad.write_text(json.dumps(fixture), encoding="utf-8")
            with self.assertRaises(ReleaseContractError):
                validate_transition_submission(bad, ROOT / "configs/tasks")

    def test_public_bundle_rebuilds_exact_bytes(self):
        bundle = ROOT / "public_bundle"
        with tempfile.TemporaryDirectory() as directory:
            result = rebuild_public_bundle(bundle, Path(directory))
            self.assertTrue(result["pass"])
            with (bundle / "public_manuscript_aggregate_bundle_manifest.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                manifest = list(csv.DictReader(handle))
            for row in manifest:
                generated = Path(directory) / row["output_path"]
                self.assertEqual(hashlib.sha256(generated.read_bytes()).hexdigest(), row["expected_output_sha256"])

    def test_public_bundle_contains_aggregate_sources_not_restricted_inputs(self):
        with (ROOT / "public_bundle/public_manuscript_aggregate_bundle_manifest.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 16)
        self.assertTrue(all(row["restricted_input_required"] == "False" for row in rows))
        self.assertTrue(any(row["role"] == "figure" for row in rows))


if __name__ == "__main__":
    unittest.main()
