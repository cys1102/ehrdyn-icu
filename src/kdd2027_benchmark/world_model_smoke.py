"""Bounded two-profile KDD235A end-to-end constructed-workflow smoke."""
from __future__ import annotations

import math
import resource
import time
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from .canonical import write_canonical_json
from .entrant_runtime import IsolatedEntrant
from .full_direct_evaluator import collect_repaired_dataset, evaluate_repaired_policy_batch
from .full_suite import environments
from .world_model_entrant import (
    COVERAGE_LEVELS, PROTOCOL_VERSION, RETENTION_GRID, frozen_h4_contract,
    structural_probability_metrics, support_only_h4_probabilities,
    validate_history, validate_policy_output, validate_prediction, validate_recursive_request,
)


def run_world_model_smoke(manifest: Path, declarations: list[Path], output: Path,
                          episodes: int = 8) -> dict[str, Any]:
    started = time.monotonic()
    selected = [env for env in environments(manifest)
                if env.seed == 171901 and env.contract.profile in ("sepsis", "aki")]
    rows: list[dict[str, Any]] = []
    policies: list[dict[str, Any]] = []
    output.mkdir(parents=True, exist_ok=True)
    for declaration_path in declarations:
        with IsolatedEntrant(declaration_path, declaration_schema="world_model_entrant",
                             protocol_version=PROTOCOL_VERSION) as entrant:
            declaration = entrant.declaration
            for profile_index, env in enumerate(selected):
                data = collect_repaired_dataset(env, episodes, 2_350_000 + profile_index, "ehr_matched")
                feature_count, action_count = env.contract.feature_dim, env.contract.action_count
                history = {
                    "observations": data.observed[:, 0].tolist(), "masks": data.masks[:, 0].astype(int).tolist(),
                    "recency": data.deltas[:, 0].tolist(), "previous_actions": data.actions[:, 0].astype(int).tolist(),
                }
                metadata = {"profile": env.contract.profile, "environment_seed": env.seed,
                            "feature_count": feature_count, "action_count": action_count,
                            "supported_actions": env.supported.astype(int).tolist()}
                entrant.request("initialize", metadata, 23501)
                entrant.request("fit_or_load", {"roles": ["train", "validation"], "sealed_final_opened": False}, 23502)
                horizon = min(4, env.horizon)
                request = history | {"schema_version": "kdd235a.request.v1",
                                     "action_sequences": data.actions[:, :horizon].astype(int).tolist(), "horizon": horizon}
                batch, checked_horizon = validate_recursive_request(
                    request, feature_count, action_count, metadata["supported_actions"])
                one_request = history | {"schema_version": "kdd235a.request.v1",
                                         "action_sequences": [[row[0]] for row in request["action_sequences"]], "horizon": 1}
                one = entrant.request("predict_one_step", one_request, 23503)["result"]
                one_validated = validate_prediction(one, declaration, batch, 1, feature_count)
                rollout = entrant.request("predict_rollout", request, 23504)["result"]
                validated = validate_prediction(rollout, declaration, batch, checked_horizon, feature_count)
                truth = data.next_observed[:, :horizon]
                valid = data.valid[:, :horizon, None] & data.masks[:, :horizon]
                one_mask = valid[:, :1]
                metrics = _metrics(validated["mean"], validated["scale"], truth, valid)
                metrics["one_step_rmse"] = _rmse(one_validated["mean"], truth[:, :1], one_mask)
                if declaration["prediction_type"] == "point":
                    metrics.update(structural_probability_metrics("point"))

                def score_sequences(row_index: int, candidates: np.ndarray) -> np.ndarray:
                    repeated = {key: [value[row_index]] * len(candidates) for key, value in history.items()}
                    candidate_request = repeated | {"schema_version": "kdd235a.request.v1",
                                                    "action_sequences": candidates.astype(int).tolist(), "horizon": 4}
                    validate_recursive_request(candidate_request, feature_count, action_count, metadata["supported_actions"])
                    result = entrant.request("predict_rollout", candidate_request, 23510 + row_index)["result"]
                    prediction = validate_prediction(result, declaration, len(candidates), 4, feature_count)["mean"]
                    return -np.mean(prediction ** 2, axis=(1, 2))

                planned = support_only_h4_probabilities(
                    score_sequences, min(episodes, 2), action_count, metadata["supported_actions"], env.seed)
                policy_payload = history | {"profile": env.contract.profile, "action_count": action_count,
                                            "supported_actions": metadata["supported_actions"], "step": 0}
                policy_result = entrant.request("predict_policy", policy_payload, 23505)["result"]
                probability = validate_policy_output(
                    policy_result, episodes, action_count, metadata["supported_actions"])

                def policy(observation: np.ndarray, mask: np.ndarray, recency: np.ndarray,
                           previous: np.ndarray, step: int) -> np.ndarray:
                    dynamic_history = {"observations": observation.tolist(), "masks": mask.astype(int).tolist(),
                                       "recency": recency.tolist(), "previous_actions": previous.astype(int).tolist()}
                    validate_history(dynamic_history, feature_count, action_count, metadata["supported_actions"])

                    def dynamic_score(row_index: int, candidates: np.ndarray) -> np.ndarray:
                        repeated = {key: [value[row_index]] * len(candidates) for key, value in dynamic_history.items()}
                        candidate_request = repeated | {"schema_version": "kdd235a.request.v1",
                                                        "action_sequences": candidates.astype(int).tolist(), "horizon": 4}
                        result = entrant.request("predict_rollout", candidate_request, 23600 + step * 10 + row_index)["result"]
                        prediction = validate_prediction(result, declaration, len(candidates), 4, feature_count)["mean"]
                        return -np.mean(prediction ** 2, axis=(1, 2))

                    return support_only_h4_probabilities(
                        dynamic_score, len(observation), action_count, metadata["supported_actions"], env.seed)
                direct = evaluate_repaired_policy_batch(env, policy, 2, 2_351_000 + profile_index, 2_352_000)
                direct_return = float(np.mean(direct["returns"]))
                learned_return = float(-np.mean(validated["mean"] ** 2))
                rows.append({"entrant_id": declaration["entrant_id"], "prediction_type": declaration["prediction_type"],
                             "profile": env.contract.profile, "environment_seed": env.seed,
                             "action_count": action_count, **metrics, "planner": frozen_h4_contract(),
                             "planner_rows": len(planned), "learned_model_return": learned_return,
                             "direct_return": direct_return, "absolute_return_gap": abs(learned_return - direct_return),
                             "direct_return_policy": "frozen_support_only_h4_recursive_entrant",
                             "counterfactual_truth": "constructed_known"})
                policies.append({"entrant_id": declaration["entrant_id"], "profile": env.contract.profile,
                                 "environment_seed": env.seed, "probabilities": probability.tolist(),
                                 "ope_ready": True, "schema_version": "kdd235a.policy.v1"})
    write_canonical_json(output / "metrics.json", rows)
    write_canonical_json(output / "ope_ready_policy_probabilities.json", policies)
    receipt = {"schema_version": "kdd235a.smoke.v1", "profiles": ["sepsis", "aki"],
               "environment_seed": 171901, "entrant_count": len(declarations), "rows": len(rows),
               "wall_time_seconds": time.monotonic() - started,
               "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               "temporary_disk_bytes": sum(path.stat().st_size for path in output.rglob("*") if path.is_file()),
               "claim_boundary": "constructed_workflow_only"}
    write_canonical_json(output / "receipt.json", receipt)
    return receipt


def _rmse(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction[mask] - truth[mask]) ** 2)))


def _metrics(mean: np.ndarray, scale: np.ndarray | None, truth: np.ndarray,
             mask: np.ndarray) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for horizon in range(mean.shape[1]):
        output[f"recursive_rmse_h{horizon + 1}"] = _rmse(mean[:, :horizon + 1], truth[:, :horizon + 1], mask[:, :horizon + 1])
    if scale is None:
        return output
    residual = truth - mean
    z = residual / scale
    normal = NormalDist()
    phi = np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    cdf = np.vectorize(normal.cdf)(z)
    crps = scale * (z * (2 * cdf - 1) + 2 * phi - 1 / math.sqrt(math.pi))
    output["crps"] = float(np.mean(crps[mask]))
    errors = []
    for level in COVERAGE_LEVELS:
        quantile = normal.inv_cdf((1 + level) / 2)
        covered = np.abs(residual) <= quantile * scale
        empirical = float(np.mean(covered[mask]))
        label = str(int(level * 100))
        output[f"coverage_{label}"] = empirical
        output[f"interval_width_{label}"] = float(np.mean((2 * quantile * scale)[mask]))
        errors.append(abs(empirical - level))
    output["mace"] = float(np.mean(errors))
    uncertainty = np.mean(scale, axis=2).reshape(-1)
    squared = np.mean(residual ** 2, axis=2).reshape(-1)
    valid_rows = mask.any(axis=2).reshape(-1)
    order = np.argsort(uncertainty[valid_rows], kind="stable")
    losses = squared[valid_rows][order]
    risks = [float(np.mean(losses[:max(1, int(len(losses) * fraction))])) for fraction in RETENTION_GRID]
    output["risk_coverage_area"] = float(np.trapz(risks, RETENTION_GRID))
    output["probabilistic_status"] = "computed"
    return output
