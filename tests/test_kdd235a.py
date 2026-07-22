from __future__ import annotations

import copy
import unittest
from pathlib import Path

import numpy as np

from kdd2027_benchmark.errors import ReleaseContractError
from kdd2027_benchmark.entrant_runtime import validate_policy_result
from kdd2027_benchmark.schema import schema_path, validate_instance, validate_schema_file
from kdd2027_benchmark.world_model_entrant import (
    frozen_h4_contract, structural_probability_metrics, support_only_h4_probabilities,
    validate_history, validate_policy_output, validate_prediction, validate_recursive_request,
)


ROOT = Path(__file__).resolve().parents[1]


def declaration(kind: str) -> dict:
    return {"prediction_type": kind, "component_sources": {"reward": "benchmark", "termination": "benchmark"}}


def prediction(kind: str, horizon: int = 2) -> dict:
    means = [[1.0, 2.0] for _ in range(horizon)]
    result = {"schema_version": "kdd235a.prediction.v1", "prediction_type": kind,
              "horizon": horizon, "mean": means}
    if kind == "independent_gaussian":
        result["scale"] = [[0.5, 0.75] for _ in range(horizon)]
    elif kind == "gaussian_ensemble":
        member_means = [[[0.9, 1.9] for _ in range(horizon)], means,
                        [[1.1, 2.1] for _ in range(horizon)]]
        member_scales = [[[0.4, 0.4] for _ in range(horizon)],
                         [[0.5, 0.5] for _ in range(horizon)],
                         [[0.6, 0.6] for _ in range(horizon)]]
        result["members"] = [{"mean": member_means[i], "scale": member_scales[i]} for i in range(3)]
        within = (0.4 ** 2 + 0.5 ** 2 + 0.6 ** 2) / 3
        between = (0.1 ** 2 + 0 + 0.1 ** 2) / 3
        result["within_variance"] = [[within, within] for _ in range(horizon)]
        result["between_variance"] = [[between, between] for _ in range(horizon)]
        result["total_variance"] = [[within + between, within + between] for _ in range(horizon)]
    return result


class KDD235ATest(unittest.TestCase):
    def test_released_schemas_are_valid(self) -> None:
        for name in ("world_model_entrant", "world_model_request", "world_model_prediction", "world_model_policy"):
            self.assertTrue(validate_schema_file(schema_path(name))["valid"])

    def test_point_gaussian_and_ensemble(self) -> None:
        for kind in ("point", "independent_gaussian", "gaussian_ensemble"):
            checked = validate_prediction(prediction(kind), declaration(kind), 1, 2, 2)
            self.assertEqual(checked["mean"].shape, (1, 2, 2))
        self.assertEqual(structural_probability_metrics("point")["probabilistic_status"], "structural_na_point_only")

    def test_point_cannot_claim_scale(self) -> None:
        value = prediction("point"); value["scale"] = [[1.0, 1.0], [1.0, 1.0]]
        with self.assertRaisesRegex(ReleaseContractError, "fabricate"):
            validate_prediction(value, declaration("point"), 1, 2, 2)

    def test_ensemble_variance_and_component_source_failures(self) -> None:
        value = prediction("gaussian_ensemble")
        value["total_variance"][0][0] += 0.1
        with self.assertRaisesRegex(ReleaseContractError, "identity"):
            validate_prediction(value, declaration("gaussian_ensemble"), 1, 2, 2)
        value = prediction("point"); value["reward"] = [0.0, 0.0]
        with self.assertRaisesRegex(ReleaseContractError, "must_not_be_supplied"):
            validate_prediction(value, declaration("point"), 1, 2, 2)

    def test_nonfinite_prediction_rejected(self) -> None:
        value = prediction("point"); value["mean"][0][0] = float("nan")
        with self.assertRaisesRegex(ReleaseContractError, "finite"):
            validate_prediction(value, declaration("point"), 1, 2, 2)

    def test_recursive_validator_catches_shape_after_one_step(self) -> None:
        validate_prediction(prediction("point", 1), declaration("point"), 1, 1, 2)
        bad = prediction("point", 2); bad["mean"] = bad["mean"][:1]
        with self.assertRaisesRegex(ReleaseContractError, "shape"):
            validate_prediction(bad, declaration("point"), 1, 2, 2)

    def test_history_action_mask_horizon_failures(self) -> None:
        value = {"schema_version": "kdd235a.request.v1", "observations": [[1.0, 2.0]], "masks": [[1, 0]], "recency": [[0.0, 1.0]],
                 "previous_actions": [0], "action_sequences": [[0, 1]], "horizon": 2}
        self.assertEqual(validate_recursive_request(value, 2, 2, [0, 1]), (1, 2))
        for key, replacement in (("masks", [[1, 2]]), ("horizon", 3), ("action_sequences", [[0, 2]])):
            invalid = copy.deepcopy(value); invalid[key] = replacement
            with self.assertRaises(ReleaseContractError):
                validate_recursive_request(invalid, 2, 2, [0, 1])

    def test_policy_probability_and_planner_identity(self) -> None:
        value = {"schema_version": "kdd235a.policy.v1", "planner": frozen_h4_contract(),
                 "probabilities": [[0.25, 0.75]]}
        output = validate_policy_output(value, 1, 2, [0, 1])
        self.assertAlmostEqual(float(output.sum()), 1.0)
        np.testing.assert_array_equal(output, validate_policy_result({"probabilities": [[0.25, 0.75]]}, 1, 2, [0, 1]))
        value["probabilities"] = [[0.25, 0.70]]
        with self.assertRaisesRegex(ReleaseContractError, "sum"):
            validate_policy_output(value, 1, 2, [0, 1])

    def test_h4_adapter_deterministic_reference(self) -> None:
        def score(_row: int, candidates: np.ndarray) -> np.ndarray:
            return -(candidates != np.asarray([1, 1, 1, 1])).sum(axis=1)
        first = support_only_h4_probabilities(score, 2, 3, [0, 1, 2], 171901)
        second = support_only_h4_probabilities(score, 2, 3, [0, 1, 2], 171901)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(np.argmax(first, axis=1), [1, 1])

    def test_schema_rejects_malformed_nested_probability(self) -> None:
        invalid = {"schema_version": "kdd235a.policy.v1", "planner": frozen_h4_contract(),
                   "probabilities": [[0.5, "bad"]]}
        with self.assertRaises(ReleaseContractError):
            validate_instance(invalid, schema_path("world_model_policy"))

    def test_declarations_validate(self) -> None:
        import json
        for name in ("point", "gaussian", "ensemble"):
            value = json.loads((ROOT / "world_model_entrant_example" / f"{name}.json").read_text())
            validate_instance(value, schema_path("world_model_entrant"))


if __name__ == "__main__":
    unittest.main()
