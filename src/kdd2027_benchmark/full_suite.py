from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .entrant_runtime import IsolatedEntrant, validate_policy_result
from .errors import ReleaseContractError
from .full_direct_evaluator import collect_repaired_dataset, evaluate_repaired_policy_batch, generator_return_range
from .full_ope import ESTIMATORS, bootstrap_policy_groups, collect_observed_history_dataset, point_policy_groups, target_probabilities
from .full_pomdp_core import EnvironmentConstruction
from .full_pomdp_types import BehaviorCalibration, DenseRewardTarget, ProfileContract
from .full_pomdp_v2 import KDD198EnvironmentV2


PROFILES = ("sepsis", "respiratory", "shock", "aki", "heart_failure")
ENVIRONMENT_SEEDS = tuple(range(171901, 171909))
DATASETS_PER_ENVIRONMENT = 8
EPISODES_PER_DATASET = 256
BOOTSTRAP_REPLICATES = 500
DATASET_SEED_BASE = 2_022_100_000
BOOTSTRAP_SEED_BASE = 2_022_200_000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    payloads = value.get("accepted_old_environments")
    if not isinstance(payloads, list) or len(payloads) != 40:
        raise ReleaseContractError("full generator contract must contain exactly 40 environments")
    identities = {(row["profile_contract"]["profile"], int(row["environment_seed"])) for row in payloads}
    expected = {(profile, seed) for profile in PROFILES for seed in ENVIRONMENT_SEEDS}
    if identities != expected:
        raise ReleaseContractError("profile/environment inventory mismatch")
    return value


def deserialize_environment(payload: dict[str, Any]) -> KDD198EnvironmentV2:
    contract_data = dict(payload["profile_contract"])
    contract_data["supported_actions"] = tuple(contract_data["supported_actions"])
    contract_data["target_action_frequency"] = tuple(contract_data["target_action_frequency"])
    contract_data["termination_hazards"] = tuple(contract_data["termination_hazards"])
    contract_data["dense_targets"] = tuple(DenseRewardTarget(**row) for row in contract_data["dense_targets"])
    behavior_data = dict(payload["behavior_calibration"])
    behavior_data["supported_actions"] = tuple(behavior_data["supported_actions"])
    behavior_data["transition_matrix"] = tuple(tuple(row) for row in behavior_data["transition_matrix"])
    environment = KDD198EnvironmentV2(
        ProfileContract(**contract_data),
        BehaviorCalibration(**behavior_data),
        int(payload["environment_seed"]),
        payload["generator"],
        EnvironmentConstruction(**payload["construction"]),
    )
    if not np.array_equal(environment.subtype_prevalence, np.asarray(payload["subtype_prevalence"])):
        raise ReleaseContractError("subtype prevalence reconstruction mismatch")
    if not np.array_equal(environment.feature_offsets, np.asarray(payload["feature_offsets"])):
        raise ReleaseContractError("feature offset reconstruction mismatch")
    return environment


def environments(manifest_path: Path) -> list[KDD198EnvironmentV2]:
    manifest = load_manifest(manifest_path)
    return [deserialize_environment(row) for row in manifest["accepted_old_environments"]]


def _payload(observation: np.ndarray, mask: np.ndarray, recency: np.ndarray,
             previous: np.ndarray, step: int, environment: KDD198EnvironmentV2) -> dict[str, Any]:
    return {
        "profile": environment.contract.profile,
        "action_count": environment.contract.action_count,
        "supported_actions": environment.supported.tolist(),
        "step": int(step),
        "observations": observation.tolist(),
        "masks": mask.astype(int).tolist(),
        "recency": recency.tolist(),
        "previous_actions": previous.astype(int).tolist(),
    }


class EntrantPolicy:
    def __init__(self, process: IsolatedEntrant, environment: KDD198EnvironmentV2, seed: int):
        self.process = process
        self.environment = environment
        self.seed = int(seed)
        self.calls = 0
        self.maximum_latency = 0.0

    def __call__(self, observation: np.ndarray, mask: np.ndarray, recency: np.ndarray,
                 previous: np.ndarray, step: int) -> np.ndarray:
        response = self.process.request(
            "predict_policy", _payload(observation, mask, recency, previous, step, self.environment),
            self.seed + step,
        )
        self.calls += 1
        self.maximum_latency = max(self.maximum_latency, float(response["latency_seconds"]))
        return np.asarray(validate_policy_result(
            response["result"], len(observation), self.environment.contract.action_count,
            self.environment.supported.tolist(),
        ))


def fixed_policy(environment: KDD198EnvironmentV2, name: str) -> Callable[..., np.ndarray]:
    supported = environment.supported
    action_count = environment.contract.action_count
    marginal = np.asarray(environment.contract.target_action_frequency, dtype=float)[supported]
    marginal /= marginal.sum()

    def policy(observation: np.ndarray, mask: np.ndarray, recency: np.ndarray,
               previous: np.ndarray, step: int) -> np.ndarray:
        output = np.zeros((len(observation), action_count), dtype=float)
        if name == "supported_random":
            output[:, supported] = 1.0 / len(supported)
        elif name == "minimum":
            output[:, supported[0]] = 1.0
        elif name == "maximum":
            output[:, supported[-1]] = 1.0
        elif name == "behavior":
            bins = environment.behavior.context_bin(observation, mask, recency, previous, step)
            for index in range(len(output)):
                output[index] = environment.behavior.distribution(int(previous[index]), int(bins[index]), action_count)
        elif name == "severity":
            signs = np.where(np.arange(observation.shape[1]) % 2 == 0, 1.0, -1.0)
            count = np.maximum(mask.sum(axis=1), 1)
            severity = np.clip(np.rint((np.sum(observation * signs * mask, axis=1) / count / environment.observation_loading + 1) * 2), 0, 4).astype(int)
            if action_count == 25:
                proposed = severity * 5 + severity
            elif action_count == 4:
                proposed = np.where(severity >= 2, 3, 0)
            else:
                proposed = (severity >= 2).astype(int)
            actions = np.asarray([supported[np.argmin(np.abs(supported - value))] for value in proposed])
            output[np.arange(len(output)), actions] = 1.0
        else:
            raise ValueError(name)
        return output
    return policy


def _exogenous_seed(profile_index: int, environment_seed: int, batch_index: int) -> int:
    return 1_993_000_000 + profile_index * 1_000_000 + environment_seed + batch_index * 100_000_000


def evaluate_policy_full(environment: KDD198EnvironmentV2, profile_index: int,
                         policy: Callable[..., np.ndarray], policy_seed_base: int,
                         levels: tuple[int, ...] = (4096, 8192, 16384, 32768, 65536),
                         se_target: float = 0.0025) -> tuple[dict[str, Any], np.ndarray]:
    parts: list[np.ndarray] = []
    previous = 0
    scale = generator_return_range(environment)
    for batch_index, level in enumerate(levels):
        increment = level - previous
        result = evaluate_repaired_policy_batch(
            environment, policy, increment,
            _exogenous_seed(profile_index, environment.seed, batch_index),
            policy_seed_base + batch_index,
        )
        parts.append(result["returns"])
        combined = np.concatenate(parts)
        normalized_se = float(combined.std(ddof=1) / math.sqrt(len(combined)) / scale)
        previous = level
        if normalized_se <= se_target:
            break
    values = np.concatenate(parts)
    se = float(values.std(ddof=1) / math.sqrt(len(values)))
    return ({
        "profile": environment.contract.profile,
        "environment_seed": environment.seed,
        "episode_count": len(values),
        "mean_return": float(values.mean()),
        "return_standard_error": se,
        "normalized_standard_error": se / scale,
        "generator_return_range": scale,
        "precision_pass": se / scale <= se_target,
    }, values)


def dataset_seed(profile_index: int, environment_index: int, dataset_index: int) -> int:
    return DATASET_SEED_BASE + profile_index * 1_000_000 + environment_index * 10_000 + dataset_index


def bootstrap_seed(profile_index: int, environment_index: int, dataset_index: int, contract_index: int) -> int:
    return BOOTSTRAP_SEED_BASE + profile_index * 10_000_000 + environment_index * 100_000 + dataset_index * 1_000 + contract_index


OPE_CONTRACTS = (
    ("primary_crossfit_no_clip", "crossfit_stronger", None),
    ("sensitivity_exact_behavior_no_clip", "exact_behavior", None),
    ("sensitivity_misspecified_power055_no_clip", "misspecified_behavior", None),
    ("sensitivity_crossfit_clip10", "crossfit_stronger", 10.0),
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(values, 0.05)), float(np.quantile(values, 0.95))


def generate_full_suite(manifest_path: Path, output: Path, cache_dir: Path | None = None) -> dict[str, Any]:
    envs = environments(manifest_path)
    rows = []
    for profile_index, profile in enumerate(PROFILES):
        for environment_index, environment_seed in enumerate(ENVIRONMENT_SEEDS):
            env = next(value for value in envs if value.contract.profile == profile and value.seed == environment_seed)
            train_seed = 1_991_000_000 + profile_index * 1_000_000 + environment_seed
            validation_seed = 1_992_000_000 + profile_index * 1_000_000 + environment_seed
            if cache_dir is not None:
                for role, episodes, seed in (("train", 128, train_seed), ("validation", 64, validation_seed)):
                    data = collect_repaired_dataset(env, episodes, seed, "ehr_matched")
                    destination = cache_dir / profile / str(environment_seed)
                    destination.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        destination / f"{role}.npz", observations=data.observed,
                        masks=data.masks, recency=data.deltas, actions=data.actions,
                        next_observations=data.next_observed, behavior_probabilities=data.behavior_probability,
                        rewards=data.rewards, terminal=data.done, valid=data.valid,
                    )
            rows.append({
                "profile": profile, "profile_index": profile_index,
                "environment_seed": environment_seed, "environment_index": environment_index,
                "action_count": env.contract.action_count, "feature_count": env.contract.feature_dim,
                "horizon": env.horizon, "mechanism_sha256": env.mechanism_hash,
                "train_seed_namespace": train_seed,
                "validation_seed_namespace": validation_seed,
                "final_exogenous_seed_namespace": _exogenous_seed(profile_index, environment_seed, 0),
                "logged_dataset_count": DATASETS_PER_ENVIRONMENT,
                "episodes_per_logged_dataset": EPISODES_PER_DATASET,
                "bootstrap_refits_per_dataset_contract": BOOTSTRAP_REPLICATES,
                "entrant_visible_latent_state": False, "entrant_visible_subtype": False,
                "final_role_open_to_training": False,
            })
    write_csv(output, rows)
    return {
        "environment_count": len(rows),
        "profile_count": len(PROFILES),
        "logged_dataset_count": len(rows) * DATASETS_PER_ENVIRONMENT,
        "episodes_per_logged_dataset": EPISODES_PER_DATASET,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "manifest_sha256": sha256(manifest_path),
        "synthetic_train_validation_cache": str(cache_dir) if cache_dir is not None else "not_materialized",
        "cache_contains_latent_state_or_subtype": False,
        "pass": len(rows) == 40 and len(rows) * DATASETS_PER_ENVIRONMENT == 320,
    }


def validate_entrant_conformance(declaration: Path, manifest_path: Path) -> dict[str, Any]:
    representative = []
    all_environments = environments(manifest_path)
    for profile in PROFILES:
        representative.append(next(env for env in all_environments if env.contract.profile == profile))
    rows = 3
    with IsolatedEntrant(declaration) as process:
        capabilities = set(process.declaration["capabilities"])
        result: dict[str, Any] = {"entrant_id": process.declaration["entrant_id"], "capabilities": sorted(capabilities)}
        reproducible = []
        distributions = []
        for profile_index, env in enumerate(representative):
            observation = np.zeros((rows, env.contract.feature_dim), dtype=float)
            mask = np.ones_like(observation, dtype=bool)
            recency = np.zeros_like(observation)
            previous = np.full(rows, env.supported[0])
            if "policy_probability" in capabilities:
                adapter = EntrantPolicy(process, env, 215_000 + profile_index * 100)
                first = adapter(observation, mask, recency, previous, 0)
                second = adapter(observation, mask, recency, previous, 0)
                reproducible.append(bool(np.array_equal(first, second)))
            if capabilities & {"transition_point", "transition_distribution", "complete_prt"}:
                response = process.request(
                    "predict_component", _payload(observation, mask, recency, previous, 0, env),
                    215_001 + profile_index * 100,
                )["result"]
                mean = np.asarray(response.get("mean"), dtype=float)
                if mean.shape != observation.shape or not np.isfinite(mean).all():
                    raise ReleaseContractError("component_mean_contract_failure")
                distributions.append("scale" in response)
        if reproducible:
            result["policy_reproducible"] = all(reproducible)
            result["policy_rows"] = rows * len(representative)
            if not result["policy_reproducible"]:
                raise ReleaseContractError("entrant_nondeterministic")
        if distributions:
            result["component_rows"] = rows * len(representative)
            result["distribution_available"] = all(distributions)
        result["profiles_probed"] = [env.contract.profile for env in representative]
        result["pass"] = True
        return result


def run_direct_returns(declaration: Path, manifest_path: Path, output: Path,
                       contrast_output: Path) -> list[dict[str, Any]]:
    envs = environments(manifest_path)
    rows: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    controls = ("behavior", "supported_random", "minimum", "maximum", "severity")
    levels = (4096, 8192, 16384, 32768, 65536)
    for profile_index, profile in enumerate(PROFILES):
        for environment in [item for item in envs if item.contract.profile == profile]:
            with IsolatedEntrant(declaration) as process:
                entrant_id = str(process.declaration["entrant_id"])
                adapter = EntrantPolicy(process, environment, 3_993_000_000 + environment.seed)
                oracle_actions = environment.exact_values("oracle")[1]
                method_names = (*controls, entrant_id, "privileged_latent_state_oracle")
                policies = {name: fixed_policy(environment, name) for name in controls}
                policies[entrant_id] = adapter
                parts: dict[str, list[np.ndarray]] = {name: [] for name in method_names}
                previous = 0
                scale = generator_return_range(environment)
                for batch_index, level in enumerate(levels):
                    increment = level - previous
                    env_seed = _exogenous_seed(profile_index, environment.seed, batch_index)
                    for method_index, method in enumerate(method_names):
                        if method == "privileged_latent_state_oracle":
                            result = environment.simulate(
                                increment, env_seed, "oracle", exact_actions=oracle_actions
                            )
                            result["unsupported_mass"] = 0.0
                        else:
                            result = evaluate_repaired_policy_batch(
                                environment, policies[method], increment, env_seed,
                                2_993_000_000 + profile_index * 1_000_000 + method_index * 1_000 + batch_index,
                            )
                        if result["terminal_emission_max"] > 1 or result["unsupported_mass"] > 1e-12:
                            raise ReleaseContractError("direct evaluator terminal or support failure")
                        parts[method].append(result["returns"])
                    returned = {name: np.concatenate(values) for name, values in parts.items()}
                    return_precision = all(
                        values.std(ddof=1) / math.sqrt(len(values)) / scale <= 0.0025
                        for values in returned.values()
                    )
                    paired_precision = all(
                        (returned[entrant_id] - returned[name]).std(ddof=1)
                        / math.sqrt(len(returned[entrant_id])) / scale <= 0.0035
                        for name in controls
                    )
                    previous = level
                    if return_precision and paired_precision:
                        break
            for method in method_names:
                values = returned[method]
                se = float(values.std(ddof=1) / math.sqrt(len(values)))
                row = {
                    "profile": profile, "environment_seed": environment.seed,
                    "episode_count": len(values), "mean_return": float(values.mean()),
                    "return_standard_error": se, "normalized_standard_error": se / scale,
                    "generator_return_range": scale, "precision_pass": se / scale <= 0.0025,
                    "entrant_id": entrant_id if method == entrant_id else "public_reference",
                    "method": method,
                    "role": ("policy_entrant" if method == entrant_id else
                             "privileged_reference" if method == "privileged_latent_state_oracle"
                             else "preregistered_control"),
                    "paired_precision_pass": all(
                        (returned[entrant_id] - returned[name]).std(ddof=1)
                        / math.sqrt(len(returned[entrant_id])) / scale <= 0.0035
                        for name in controls
                    ),
                }
                if method == entrant_id:
                    row.update({"jsonl_calls": adapter.calls})
                rows.append(row)
            for comparator in controls:
                difference = returned[entrant_id] - returned[comparator]
                n = len(difference)
                se = float(difference.std(ddof=1) / math.sqrt(n))
                contrasts.append({
                    "profile": profile, "environment_seed": environment.seed,
                    "entrant_id": entrant_id, "comparator": comparator,
                    "paired_episode_count": n, "paired_mean_difference": float(difference.mean()),
                    "paired_standard_error": se, "paired_ci_lower": float(difference.mean() - 1.96 * se),
                    "paired_ci_upper": float(difference.mean() + 1.96 * se),
                    "normalized_paired_standard_error": se / scale,
                    "paired_precision_threshold": 0.0035,
                    "paired_precision_pass": se / scale <= 0.0035,
                    "inferential_unit_for_summary": "profile_x_environment_seed",
                })
    write_csv(output, rows)
    write_csv(contrast_output, contrasts)
    return rows


def run_component_forecasting(declaration: Path, manifest_path: Path, output: Path) -> list[dict[str, Any]]:
    envs = environments(manifest_path)
    rows: list[dict[str, Any]] = []
    normal_crps_constant = 1.0 / math.sqrt(math.pi)
    for profile_index, profile in enumerate(PROFILES):
        for environment_index, environment in enumerate([item for item in envs if item.contract.profile == profile]):
            data = collect_repaired_dataset(
                environment, 256, 1_992_500_000 + profile_index * 1_000_000 + environment.seed,
                "ehr_matched",
            )
            with IsolatedEntrant(declaration) as process:
                means, scales, targets, observed_masks = [], [], [], []
                distribution_available = False
                for step in range(environment.horizon):
                    response = process.request(
                        "predict_component",
                        _payload(data.observed[:, step], data.masks[:, step], data.deltas[:, step],
                                 data.actions[:, max(step - 1, 0)], step, environment),
                        5_215_000_000 + profile_index * 100_000 + environment_index * 1_000 + step,
                    )["result"]
                    mean = np.asarray(response.get("mean"), dtype=float)
                    scale_value = response.get("scale")
                    scale = np.asarray(scale_value, dtype=float) if scale_value is not None else None
                    if mean.shape != data.next_observed[:, step].shape or not np.isfinite(mean).all():
                        raise ReleaseContractError("component_mean_contract_failure")
                    if scale is not None and (scale.shape != mean.shape or not np.isfinite(scale).all() or np.any(scale <= 0)):
                        raise ReleaseContractError("component_scale_contract_failure")
                    means.append(mean); scales.append(scale); targets.append(data.next_observed[:, step])
                    observed_masks.append(data.masks[:, step] & data.valid[:, step, None])
                    distribution_available |= scale is not None
                mean_all = np.stack(means, axis=1)
                target_all = np.stack(targets, axis=1)
                observed = np.stack(observed_masks, axis=1)
                error = mean_all[observed] - target_all[observed]
                row = {
                    "profile": profile, "environment_seed": environment.seed, "forecast_horizon": 1,
                    "entrant_id": process.declaration["entrant_id"], "observed_target_count": int(observed.sum()),
                    "rmse": float(np.sqrt(np.mean(np.square(error)))), "mae": float(np.mean(np.abs(error))),
                    "distribution_available": distribution_available, "status": "numeric_one_step",
                }
                if distribution_available:
                    scale_all = np.stack([value for value in scales if value is not None], axis=1)
                    if scale_all.shape != mean_all.shape:
                        raise ReleaseContractError("component_distribution_missing_for_some_steps")
                    local_scale = scale_all[observed]
                    z = error / local_scale
                    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
                    pdf = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
                    crps = local_scale * (z * (2 * cdf - 1) + 2 * pdf - normal_crps_constant)
                    coverage = float(np.mean(np.abs(z) <= 1.6448536269514722))
                    row.update({"crps": float(np.mean(crps)), "coverage90": coverage,
                                "coverage_error90": abs(coverage - 0.9),
                                "width90": float(np.mean(2 * 1.6448536269514722 * local_scale)),
                                "mace": abs(coverage - 0.9)})
                rows.append(row)
                for horizon in range(2, environment.horizon + 1):
                    rows.append({"profile": profile, "environment_seed": environment.seed,
                                 "forecast_horizon": horizon, "entrant_id": process.declaration["entrant_id"],
                                 "observed_target_count": "na", "distribution_available": distribution_available,
                                 "status": "structural_na_transition_only_no_observation_process_rollout"})
    write_csv(output, rows)
    return rows


def run_full_ope(declaration: Path, manifest_path: Path, direct_rows_path: Path,
                 output: Path, workers: int = 1) -> list[dict[str, Any]]:
    direct_rows = list(csv.DictReader(direct_rows_path.open(newline="", encoding="utf-8")))
    entrant_id = json.loads(declaration.read_text(encoding="utf-8"))["entrant_id"]
    truth = {(row["profile"], int(row["environment_seed"])): float(row["mean_return"])
             for row in direct_rows if row["method"] == entrant_id}
    behavior_truth = {(row["profile"], int(row["environment_seed"])): float(row["mean_return"])
                      for row in direct_rows if row["method"] == "behavior"}
    if len(truth) != 40:
        raise ReleaseContractError("entrant direct-return truth inventory is not 40")
    manifest = load_manifest(manifest_path)
    tasks = []
    for profile_index, profile in enumerate(PROFILES):
        payloads = [row for row in manifest["accepted_old_environments"] if row["profile_contract"]["profile"] == profile]
        for environment_index, payload in enumerate(payloads):
            tasks.append((str(declaration.resolve()), profile_index, environment_index, payload,
                          truth[(profile, int(payload["environment_seed"]))],
                          behavior_truth[(profile, int(payload["environment_seed"]))]))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_run_environment_ope, tasks))
    else:
        parts = [_run_environment_ope(task) for task in tasks]
    rows = [row for part in parts for row in part]
    write_csv(output, rows)
    return rows


def _run_environment_ope(arguments: tuple[str, int, int, dict[str, Any], float, float]) -> list[dict[str, Any]]:
    declaration_text, profile_index, environment_index, payload, true_value, behavior_value = arguments
    declaration = Path(declaration_text)
    entrant_id = json.loads(declaration.read_text(encoding="utf-8"))["entrant_id"]
    environment = deserialize_environment(payload)
    profile = environment.contract.profile
    rows: list[dict[str, Any]] = []
    for dataset_index in range(DATASETS_PER_ENVIRONMENT):
        dseed = dataset_seed(profile_index, environment_index, dataset_index)
        data = collect_observed_history_dataset(environment, EPISODES_PER_DATASET, dseed, "ehr_matched")
        with IsolatedEntrant(declaration) as process:
            adapter = EntrantPolicy(process, environment, 6_215_000_000 + dseed)
            entrant_target = target_probabilities(data, adapter)
        behavior_target = target_probabilities(data, fixed_policy(environment, "behavior"))
        policy_groups = {entrant_id: [entrant_target], "behavior": [behavior_target]}
        true_values = {entrant_id: true_value, "behavior": behavior_value}
        for contract_index, (contract_name, denominator, clip) in enumerate(OPE_CONTRACTS):
            bseed = bootstrap_seed(profile_index, environment_index, dataset_index, contract_index)
            points, diagnostics = point_policy_groups(
                data, policy_groups, denominator, clip, 5, 0.5, 0.5, bseed
            )
            bootstrap = bootstrap_policy_groups(
                data, policy_groups, BOOTSTRAP_REPLICATES, bseed,
                denominator, clip, 5, 0.5, 0.5, workers=1,
            )
            for method in (entrant_id, "behavior"):
                for estimator in ESTIMATORS:
                    lower, upper = percentile_interval(bootstrap[method][estimator])
                    estimate = points[method][estimator]
                    rows.append({
                        "profile": profile, "environment_seed": environment.seed,
                        "dataset_index": dataset_index, "dataset_seed": dseed,
                        "episodes": EPISODES_PER_DATASET, "entrant_id": entrant_id,
                        "policy": method, "contract": contract_name, "denominator": denominator,
                        "clip": "none" if clip is None else clip, "support": "frozen_environment_support",
                        "horizon": "full_episode", "estimator": estimator,
                        "estimate": estimate, "ci90_lower": lower, "ci90_upper": upper,
                        "direct_true_return": true_values[method], "absolute_error": abs(estimate - true_values[method]),
                        "behavior_direct_return": behavior_value,
                        "behavior_relative_estimate": estimate - points["behavior"][estimator],
                        "behavior_relative_true_difference": true_values[method] - behavior_value,
                        "sign_recovered": np.sign(estimate - points["behavior"][estimator]) == np.sign(true_values[method] - behavior_value),
                        "pairwise_order_recovered": np.sign(points[entrant_id][estimator] - points["behavior"][estimator]) == np.sign(true_value - behavior_value),
                        "false_improvement": method != "behavior" and estimate > points["behavior"][estimator] and true_value <= behavior_value,
                        "generator_return_range": generator_return_range(environment),
                        "covered90": lower <= true_values[method] <= upper, "ess": diagnostics[method]["ess"],
                        "unsupported_mass": diagnostics[method]["unsupported_mass"],
                        "finite": math.isfinite(estimate), "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                        "nuisance_refit_inside_each_bootstrap": True,
                    })
    return rows


def summarize_ope(input_path: Path, output: Path) -> list[dict[str, Any]]:
    source = list(csv.DictReader(input_path.open(newline="", encoding="utf-8")))
    disagreement_groups: dict[tuple[str, str, str, str, str], list[float]] = {}
    for row in source:
        key = (row["policy"], row["contract"], row["profile"], row["environment_seed"], row["dataset_index"])
        disagreement_groups.setdefault(key, []).append(float(row["estimate"]))
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in source:
        groups.setdefault((row["policy"], row["contract"], row["estimator"]), []).append(row)
    rows = []
    for (policy, contract, estimator), items in sorted(groups.items()):
        errors = np.asarray([float(row["absolute_error"]) for row in items])
        normalized = np.asarray([
            float(row["absolute_error"]) / float(row["generator_return_range"])
            for row in items
        ])
        rows.append({
            "policy": policy, "contract": contract, "estimator": estimator, "dataset_cell_count": len(items),
            "environment_count": len({(row["profile"], row["environment_seed"]) for row in items}),
            "mean_absolute_error": float(errors.mean()), "normalized_mean_absolute_error": float(normalized.mean()),
            "repeated_dataset_coverage90": float(np.mean([row["covered90"] == "True" for row in items])),
            "sign_recovery": float(np.mean([row["sign_recovered"] == "True" for row in items])),
            "pairwise_order_recovery": float(np.mean([row["pairwise_order_recovered"] == "True" for row in items])),
            "false_improvement_rate": float(np.mean([row["false_improvement"] == "True" for row in items])),
            "mean_ess": float(np.mean([float(row["ess"]) for row in items])),
            "nonfinite_rate": float(np.mean([row["finite"] != "True" for row in items])),
            "mean_within_dataset_estimator_standard_deviation": float(np.mean([
                np.std(disagreement_groups[key])
                for key in sorted({
                    (row["policy"], row["contract"], row["profile"], row["environment_seed"], row["dataset_index"])
                    for row in items
                })
            ])),
            "counterfactual_truth": "known_constructed_simulator_direct_return",
        })
    write_csv(output, rows)
    return rows
