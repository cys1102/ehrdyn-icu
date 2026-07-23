from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from kdd2027_benchmark.canonical import canonical_bytes
from kdd2027_benchmark.ehr_component_scorer import (
    EHR_COMPONENT_BENCHMARK_VERSION,
    EHR_COMPONENT_EVALUATOR_VERSION,
    TASK_CONTRACT,
    score_submission,
)
from kdd2027_benchmark.errors import ReleaseContractError
from kdd2027_benchmark.schema import schema_path, validate_schema_file


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "kdd245v2r"


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def score(payload: dict) -> dict:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "submission.json"
        path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")
        return score_submission(path)


class KDD245V2RScorerTests(unittest.TestCase):
    def test_released_schemas_are_draft_2020_12(self) -> None:
        for name in ("ehr_component_submission", "ehr_component_result"):
            self.assertTrue(validate_schema_file(schema_path(name))["valid"])
        for name in sorted(path.name for path in (ROOT / "schemas").glob("*.schema.json")):
            self.assertEqual(
                (ROOT / "schemas" / name).read_bytes(),
                (ROOT / "src" / "kdd2027_benchmark" / "package_schemas" / name).read_bytes(),
            )

    def test_point_fixture_and_structural_na(self) -> None:
        result = score_submission(FIXTURES / "point.json")
        self.assertEqual(result["benchmark_version"], EHR_COMPONENT_BENCHMARK_VERSION)
        self.assertEqual(result["evaluator_version"], EHR_COMPONENT_EVALUATOR_VERSION)
        self.assertEqual(result["one_step_rmse"], 1.0)
        self.assertEqual(result["probabilistic_metrics"]["status"], "structural_na_point_only")
        self.assertIsNone(result["probabilistic_metrics"]["crps"])
        self.assertEqual(result["capabilities"]["planning"], "No")
        self.assertEqual(result["capabilities"]["known_policy_value"], "Structural N/A")

    def test_gaussian_fixture_has_independently_known_scores(self) -> None:
        result = score_submission(FIXTURES / "gaussian.json")
        metrics = result["probabilistic_metrics"]
        self.assertAlmostEqual(metrics["crps"], 0.23369497725510913, places=12)
        self.assertEqual(metrics["coverage_50"], 1.0)
        self.assertAlmostEqual(metrics["interval_width_90"], 3.289707253902943, places=12)
        self.assertAlmostEqual(result["termination_metrics"]["brier"], 0.04, places=12)
        self.assertAlmostEqual(result["support_diagnostics"]["importance_weight_ess"], 1.8, places=12)
        self.assertAlmostEqual(result["support_diagnostics"]["unsupported_target_mass"], 0.5, places=12)

    def test_ensemble_within_between_uncertainty(self) -> None:
        result = score_submission(FIXTURES / "ensemble.json")
        self.assertAlmostEqual(
            result["probabilistic_metrics"]["crps"], 0.3304946062926473, places=12
        )
        self.assertAlmostEqual(
            result["probabilistic_metrics"]["interval_width_90"],
            4.6523486147066935,
            places=12,
        )

    def test_all_five_task_action_and_horizon_contracts(self) -> None:
        base = load("point.json")
        for task_id, contract in TASK_CONTRACT.items():
            payload = copy.deepcopy(base)
            payload["task_id"] = task_id
            payload["records"][0]["action"] = contract["action_count"] - 1
            payload["records"][1]["action"] = contract["action_count"] - 1
            payload["records"][1]["horizon_step"] = contract["max_horizon"]
            result = score(payload)
            self.assertEqual(result["task_id"], task_id)
            self.assertEqual(result["records_scored"], 2)

    def test_version_horizon_action_and_finiteness_fail_closed(self) -> None:
        cases = []
        payload = load("point.json")
        payload["benchmark_version"] = "v1.3.0"
        cases.append(payload)
        payload = load("point.json")
        payload["records"][0]["horizon_step"] = 12
        cases.append(payload)
        payload = load("point.json")
        payload["records"][0]["action"] = 4
        cases.append(payload)
        payload = load("point.json")
        payload["records"][0]["mean"][0] = float("nan")
        cases.append(payload)
        for payload in cases:
            with self.assertRaises(ReleaseContractError):
                score(payload)

    def test_normalization_identifier_leakage_and_row_output_fail_closed(self) -> None:
        cases = []
        for key, value in (
            ("subject_id", 1),
            ("future_outcome", 1),
            ("row_predictions", [[1.0]]),
        ):
            payload = load("point.json")
            payload[key] = value
            cases.append(payload)
        payload = load("point.json")
        payload["records"][0]["behavior_probability"] = 0
        cases.append(payload)
        payload = load("point.json")
        payload["records"][0]["target_probability"] = 1.1
        cases.append(payload)
        for payload in cases:
            with self.assertRaises(ReleaseContractError):
                score(payload)

    def test_prediction_form_contracts_fail_closed(self) -> None:
        point = load("point.json")
        point["records"][0]["scale"] = [1.0] * 33
        with self.assertRaisesRegex(ReleaseContractError, "fabrication"):
            score(point)
        gaussian = load("gaussian.json")
        del gaussian["records"][0]["scale"]
        with self.assertRaisesRegex(ReleaseContractError, "scale"):
            score(gaussian)
        ensemble = load("ensemble.json")
        ensemble["records"][0]["mean"][0] = 0.1
        with self.assertRaisesRegex(ReleaseContractError, "identity"):
            score(ensemble)

    def test_real_ehr_small_cells_are_suppressed(self) -> None:
        payload = load("gaussian.json")
        payload["synthetic"] = False
        payload["aggregate_context"] = {"distinct_subjects": 99, "episodes": 100}
        result = score(payload)
        self.assertTrue(result["suppression"]["applied"])
        self.assertIsNone(result["support_diagnostics"]["importance_weight_ess"])
        self.assertEqual(result["support_diagnostics"]["status"], "suppressed_minimum_cell")

    def test_canonical_serialization_is_deterministic(self) -> None:
        result = score_submission(FIXTURES / "gaussian.json")
        self.assertEqual(canonical_bytes(result), canonical_bytes(score_submission(FIXTURES / "gaussian.json")))

    def test_cli_writes_schema_valid_aggregate_only_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "score.json"
            run = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "kdd2027_benchmark.cli",
                    "score-ehr-components",
                    "--submission",
                    str(FIXTURES / "gaussian.json"),
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            text = output.read_text(encoding="utf-8")
            for forbidden in ("subject_id", "stay_id", "hadm_id", "patient_id", "timestamp"):
                self.assertNotIn(forbidden, text)
            self.assertEqual(json.loads(text)["records_scored"], 2)

    def test_public_constructor_source_is_exact_79deff45(self) -> None:
        expected = {
            "lineage_source_port.py": "08d65ba14a0ad7df9afd6805c4fa04ed3fadc888bb822c1f12ce77336fcbc020",
            "reconstruct.py": "914356315ba3a489b6d97f5b2770b3c87e1b770e3ff735a6198a9a064f69592f",
        }
        root = ROOT / "src" / "kdd2027_benchmark" / "current_five_task"
        for name, digest in expected.items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), digest)

    def test_canonical_manifest_and_capability_boundary(self) -> None:
        manifest = json.loads(
            (ROOT / "expected" / "canonical_v2_five_task_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["constructor_source_commit"], "79deff45c5d4e5349e4ce648e212e6fbce1a27bd")
        self.assertEqual(
            manifest["scientific_surface_sha256"],
            "848be7b3103f6272c020e4ba1a7c23fe278c51bfd4aaf2d7ad53a171569a5505",
        )
        self.assertEqual([row["task_id"] for row in manifest["tasks"]], list(TASK_CONTRACT))
        self.assertEqual([row["action_count"] for row in manifest["tasks"]], [25, 25, 25, 4, 2])
        for name in ("planning", "direct_return", "known_policy_value", "treatment_benefit", "clinical_utility"):
            self.assertIn(name, json.loads(
                (ROOT / "configs" / "ehr_component" / "canonical_v2_scorer.json").read_text()
            )["unsupported_real_ehr_surfaces"])

    def test_stopped_kdd245_receipts_are_byte_identical(self) -> None:
        expected = {
            "kdd245": "5734bf6a819d20655769b418ca683bd02645698571cfab3803a295bf7763084e",
            "kdd245v2": "fcd8251126de5aa42e4fd5fe4669da47ac1b1dbd60f2f4c26e8f00050295c621",
        }
        for name, digest in expected.items():
            root = ROOT / "release" / name
            value = hashlib.sha256()
            files = sorted(path for path in root.rglob("*") if path.is_file())
            for path in files:
                value.update(path.relative_to(root).as_posix().encode("utf-8"))
                value.update(path.read_bytes())
            self.assertEqual(value.hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
