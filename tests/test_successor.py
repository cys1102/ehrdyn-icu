from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import cast

from kdd2027_benchmark.errors import ReleaseContractError
from kdd2027_benchmark.rv import CLAIM_BOUNDARY, SUCCESSOR_BENCHMARK_VERSION
from kdd2027_benchmark.rv.contract import validate_config_directory, validate_contract_manifest
from kdd2027_benchmark.rv.evaluator import evaluate_predictions
from kdd2027_benchmark.rv.evidence import verify_evidence
from kdd2027_benchmark.rv.fixture import generate_fixture
from kdd2027_benchmark.rv.rollout import LoggedTransition, conditional_recursive_rollout
from kdd2027_benchmark.rv.split import subject_role
from kdd2027_benchmark.rv.submission import validate_submission


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "successor/configs/tasks"


class SuccessorContractTests(unittest.TestCase):
    def test_exact_five_task_contract_validates(self):
        configs = validate_config_directory(CONFIGS)
        self.assertEqual({str(config["task_id"]) for config in configs}, {"sepsis", "respiratory", "aki", "af_flutter", "heart_failure"})
        self.assertEqual({config["clinical_review_status"] for config in configs}, {"pending_not_simulated"})
        self.assertEqual({config["claim_boundary"] for config in configs}, {CLAIM_BOUNDARY})
        manifest = validate_contract_manifest(ROOT / "successor/contracts/contract_manifest.json")
        self.assertEqual(manifest["benchmark_version"], SUCCESSOR_BENCHMARK_VERSION)

    def test_sha256_role_hash_has_frozen_examples(self):
        self.assertEqual(subject_role("0"), "train")
        self.assertEqual(subject_role("2"), "sealed_test")
        self.assertEqual(subject_role("4"), "validation")

    def test_recursive_rollout_uses_prior_prediction_and_resets_after_gap(self):
        seen: list[tuple[tuple[tuple[float, ...], ...], tuple[tuple[int, ...], ...], tuple[float, ...]]] = []

        def predictor(history, masks, recencies, action):
            del recencies
            seen.append((history, masks, action))
            return ([history[-1][0] + 1.0], [0.5])

        transitions = [
            LoggedTransition("a", 0, ((1.0,), (2.0,)), ((1,), (1,)), ((0.0,), (0.0,)), (0.0,)),
            LoggedTransition("a", 1, ((10.0,), (20.0,)), ((0,), (1,)), ((4.0,), (0.0,)), (1.0,)),
            LoggedTransition("a", 3, ((30.0,), (40.0,)), ((1,), (1,)), ((0.0,), (0.0,)), (2.0,)),
        ]
        output = conditional_recursive_rollout(transitions, predictor)
        self.assertEqual([row.segment_horizon for row in output], [1, 2, 1])
        self.assertEqual([row.mean for row in output], [(3.0,), (4.0,), (41.0,)])
        self.assertEqual(seen[1][0], ((2.0,), (3.0,)))
        self.assertEqual(seen[1][1], ((0,), (1,)))
        self.assertEqual(seen[1][2], (1.0,))

    def test_fixture_evaluation_is_aggregate_and_subject_clustered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.csv"
            normalization = root / "normalization.json"
            contract = root / "evaluation_contract.json"
            generate_fixture(predictions, normalization, contract, subjects=4, transitions=3, seed=7)
            result = evaluate_predictions(
                predictions,
                normalization,
                contract,
                cluster_key_column="synthetic_subject_key",
                sequence_key_column="synthetic_sequence_key",
                synthetic=True,
                bootstrap_replicates=40,
            )
        self.assertEqual(result["benchmark_version"], SUCCESSOR_BENCHMARK_VERSION)
        self.assertEqual(result["bootstrap_unit"], "subject_cluster")
        self.assertEqual(len(cast(list[object], result["metrics"])), 20)
        self.assertEqual(len(cast(list[object], result["paired_subject_cluster_delta_vs_point_minimum"])), 20)
        self.assertIn("evaluation_receipt", result)
        self.assertTrue(
            all(
                row["unique_winner_claim_allowed"] is False
                for row in cast(list[dict[str, object]], result["practical_leader_sets"])
            )
        )
        serialized = json.dumps(result, sort_keys=True)
        self.assertNotIn("syn-subject", serialized)
        self.assertNotIn("synthetic_subject_key", serialized)
        self.assertNotIn("synthetic_sequence_key", serialized)

    def test_evaluator_rejects_unpaired_method_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.csv"
            broken = root / "broken.csv"
            normalization = root / "normalization.json"
            contract = root / "evaluation_contract.json"
            generate_fixture(predictions, normalization, contract, subjects=2, transitions=2, seed=9)
            with predictions.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
                fields = list(rows[0])
            rows.pop()
            with broken.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ReleaseContractError, "identical successor cells"):
                evaluate_predictions(
                    broken,
                    normalization,
                    contract,
                    cluster_key_column="synthetic_subject_key",
                    sequence_key_column="synthetic_sequence_key",
                    synthetic=True,
                    bootstrap_replicates=20,
                )

    def test_self_reported_submission_template_is_rejected(self):
        path = ROOT / "successor/submission/aggregate_submission_template.json"
        with self.assertRaisesRegex(ReleaseContractError, "Self-reported aggregate rows are rejected"):
            validate_submission(path, validate_config_directory(CONFIGS))

    def test_evaluator_receipt_validates_and_tampering_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.csv"
            normalization = root / "normalization.json"
            contract = root / "evaluation_contract.json"
            output = root / "evaluation.json"
            generate_fixture(predictions, normalization, contract, subjects=3, transitions=2, seed=11)
            result = evaluate_predictions(
                predictions,
                normalization,
                contract,
                cluster_key_column="synthetic_subject_key",
                sequence_key_column="synthetic_sequence_key",
                synthetic=True,
                bootstrap_replicates=20,
            )
            output.write_text(json.dumps(result), encoding="utf-8")
            receipt = validate_submission(output, validate_config_directory(CONFIGS))
            self.assertTrue(receipt["evaluator_receipt_verified"])
            metrics = cast(list[dict[str, object]], result["metrics"])
            metrics[0]["normalized_rmse"] = 999.0
            output.write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(ReleaseContractError, "payload hash mismatch"):
                validate_submission(output, validate_config_directory(CONFIGS))

    def test_packaged_evidence_matches_source_receipts(self):
        manifest = ROOT / "successor/evidence/evidence_manifest.csv"
        with manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 16)
        for row in rows:
            path = ROOT / row["packaged_path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row["sha256"])
            self.assertEqual(path.stat().st_size, int(row["bytes"]))
            self.assertEqual(row["aggregate_or_contract_only"], "True")
        receipt = verify_evidence(ROOT, manifest)
        self.assertEqual(receipt["evidence_files_verified"], 16)
        self.assertEqual(receipt["privacy_receipts_verified"], 2)

    def test_evaluation_contract_rejects_truncated_common_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.csv"
            truncated = root / "truncated.csv"
            normalization = root / "normalization.json"
            contract = root / "evaluation_contract.json"
            generate_fixture(predictions, normalization, contract, subjects=2, transitions=2, seed=13)
            with predictions.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
                fields = list(rows[0])
            rows = [row for row in rows if row["feature_name"] != "lactate"]
            with truncated.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ReleaseContractError, "exact 33-feature contract"):
                evaluate_predictions(
                    truncated,
                    normalization,
                    contract,
                    cluster_key_column="synthetic_subject_key",
                    sequence_key_column="synthetic_sequence_key",
                    synthetic=True,
                    bootstrap_replicates=20,
                )

    def test_clinical_review_rows_remain_pending(self):
        with (ROOT / "successor/clinical_review/status.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 5)
        self.assertEqual({row["status"] for row in rows}, {"pending"})
        self.assertEqual({row["decision"] for row in rows}, {""})


if __name__ == "__main__":
    unittest.main()
