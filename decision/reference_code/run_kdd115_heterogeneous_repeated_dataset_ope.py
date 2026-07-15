from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from kdd_benchmark_discovery import kdd_e01_evaluator as e01
from kdd_benchmark_discovery import run_kdd107_heterogeneous_known_value as k107
from kdd_benchmark_discovery import run_kdd_adapt01_adaptive_known_value as adapt
from kdd_benchmark_discovery import run_kdd_e02_known_value_full as e02
from kdd_benchmark_discovery import run_kdd_ope_rd01_repeated_dataset as rd01


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/kdd115_heterogeneous_repeated_dataset_ope_v1.json"
KDD107_CONFIG = ROOT / "configs/kdd107_heterogeneous_known_value_v1.json"
KDD107_RESULT = ROOT / "kdd_benchmark_discovery/results/kdd107_heterogeneous_known_value_20260715_1010"
RD01_CONFIG = ROOT / "configs/kdd_ope_rd01_repeated_dataset_v1.json"
RD01_RESULT = ROOT / "kdd_benchmark_discovery/results/kdd_ope_rd01_repeated_dataset_20260715_094928"
REQUIRED_OUTPUTS = (
    "repeated_dataset_ope_rows.csv",
    "repeated_dataset_coverage.csv",
    "coverage_precision_intervals.csv",
    "ope_rank_and_sign_recovery.csv",
    "ope_authorization_heterogeneous.csv",
    "mechanism_level_failure_summary.csv",
    "estimator_task_mechanism_map.csv",
    "seed_namespace_manifest.csv",
    "failure_ledger.csv",
    "result_audit.md",
    "summary.md",
)
CLAIM_BOUNDARY = (
    "Exact finite heterogeneous known-value OPE calibration only; no retrospective EHR policy-value "
    "authorization, treatment benefit, causal effect, clinical utility, deployment, policy superiority, "
    "or cross-disease generalization claim."
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_paths() -> dict[str, Path]:
    return {
        "kdd107_config": KDD107_CONFIG,
        "kdd107_runner": ROOT / "kdd_benchmark_discovery/run_kdd107_heterogeneous_known_value.py",
        "kdd107_frozen_environment_contracts": KDD107_RESULT / "known_value_environment_contracts.json",
        "kdd107_policy_level_results": KDD107_RESULT / "policy_level_results.csv",
        "kdd107_policy_implementation": ROOT / "kdd_benchmark_discovery/run_kdd_x02_cross_cohort_policy_benchmark.py",
        "completed_repeated_dataset_config": RD01_CONFIG,
        "completed_repeated_dataset_runner": ROOT / "kdd_benchmark_discovery/run_kdd_ope_rd01_repeated_dataset.py",
        "completed_repeated_dataset_authorization": RD01_RESULT / "ope_authorization_revised.csv",
        "e01_evaluator": ROOT / "kdd_benchmark_discovery/kdd_e01_evaluator.py",
        "e02_runner": ROOT / "kdd_benchmark_discovery/run_kdd_e02_known_value_full.py",
    }


def _table_policy(probability: np.ndarray, env: e01.FiniteMDP) -> np.ndarray:
    return adapt.table_policy(probability, env)


def target_policies(
    env: e01.FiniteMDP, task: str, k107_config: dict[str, Any]
) -> dict[str, np.ndarray]:
    profile = k107_config["task_profiles"][task]
    supported = np.asarray(profile["supported_action_indices"], dtype=int)
    random = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
    random[:, supported] = 1.0 / len(supported)
    minimum = np.eye(env.n_actions)[np.full(env.n_states, supported[0])]
    maximum = np.eye(env.n_actions)[np.full(env.n_states, supported[-1])]
    severity_index = np.clip(
        np.floor(np.arange(env.n_states) / max(env.n_states / len(supported), 1)).astype(int),
        0,
        len(supported) - 1,
    )
    severity = np.eye(env.n_actions)[supported[severity_index]]
    _, oracle, _ = e01.backward_induction(env)
    policies = {
        "empirical_behavior": env.behavior.copy(),
        "random_supported": _table_policy(random, env),
        "minimum_supported_action": _table_policy(minimum, env),
        "maximum_supported_action": _table_policy(maximum, env),
        "severity_rule": _table_policy(severity, env),
        "exact_dynamic_programming_oracle": oracle,
    }
    for name, policy in policies.items():
        if policy.shape != env.behavior.shape:
            raise RuntimeError(f"policy shape drift: {name}: {policy.shape}")
        unsupported = float(np.sum(policy * (~env.support)[None]))
        if unsupported != 0.0:
            raise RuntimeError(f"support-mask bypass: {name}: {unsupported}")
    return policies


def logged_seed(config: dict[str, Any], task_index: int, mechanism_index: int, dataset_index: int) -> int:
    return (
        int(config["seed_namespaces"]["logged_dataset_base"])
        + task_index * 1_000_000
        + mechanism_index * 100_000
        + dataset_index
    )


def verify_frozen_contract(config: dict[str, Any]) -> dict[str, Any]:
    actual = {name: sha256(path) for name, path in source_paths().items()}
    if actual != config["immutable_source_hashes"]:
        raise RuntimeError(f"immutable source drift: {actual}")
    repeated = json.loads(RD01_CONFIG.read_text(encoding="utf-8"))
    for key in (
        "target_policies",
        "task_horizons",
        "estimators",
        "denominators",
        "clipping",
        "support_contract",
        "interval",
        "coverage_precision_interval",
        "authorization_gates",
    ):
        if config[key] != repeated[key]:
            raise RuntimeError(f"completed repeated-dataset contract drift: {key}")
    k107_config = json.loads(KDD107_CONFIG.read_text(encoding="utf-8"))
    frozen_contracts = {
        (row["task_profile"], row["mechanism"]): row
        for row in json.loads((KDD107_RESULT / "known_value_environment_contracts.json").read_text(encoding="utf-8"))
    }
    frozen_values = pd.read_csv(KDD107_RESULT / "policy_level_results.csv")
    control_names = [
        "empirical_behavior",
        "random_supported",
        "minimum_supported_action",
        "maximum_supported_action",
        "severity_rule",
    ]
    parity_rows = []
    for task in config["logged_dataset_replicates"]:
        for mechanism in config["mechanisms"]:
            env = k107.build_environment(task, mechanism, k107_config)
            generated_contract = k107._contract_record(task, mechanism, env, k107_config)
            if generated_contract != frozen_contracts[(task, mechanism)]:
                raise RuntimeError(f"frozen environment contract parity failure: {task}/{mechanism}")
            policies = target_policies(env, task, k107_config)
            local = frozen_values[
                (frozen_values["task"] == task)
                & (frozen_values["mechanism"] == mechanism)
                & (frozen_values["policy"].isin(control_names))
            ]
            for policy_name in control_names:
                expected = local.loc[local["policy"] == policy_name, "exact_return"].drop_duplicates()
                if len(expected) != 1:
                    raise RuntimeError(f"ambiguous KDD107 control truth: {task}/{mechanism}/{policy_name}")
                observed = e01.evaluate_policy_exact(env, policies[policy_name])
                difference = abs(observed - float(expected.iloc[0]))
                if difference > 1e-12:
                    raise RuntimeError(
                        f"KDD107 policy truth parity failure: {task}/{mechanism}/{policy_name}: {difference}"
                    )
                parity_rows.append({
                    "task": task,
                    "mechanism": mechanism,
                    "policy": policy_name,
                    "exact_return_absolute_difference": difference,
                })
    prior_max = int(config["seed_namespaces"]["prior_repeated_dataset_seed_max"])
    new_min = logged_seed(config, 0, 0, 0)
    if new_min <= prior_max:
        raise RuntimeError("logged-dataset seed namespace overlaps prior audit")
    return {"source_hashes": actual, "policy_parity_rows": parity_rows}


def _cell_worker(
    task: str,
    mechanism: str,
    task_index: int,
    mechanism_index: int,
    replicates: int,
    config: dict[str, Any],
    k107_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import torch

    torch.set_num_threads(1)
    env = k107.build_environment(task, mechanism, k107_config)
    policies = target_policies(env, task, k107_config)
    policy_names = list(policies)
    horizons = [int(value) for value in config["task_horizons"][task]]
    truths = {
        (horizon, policy_name): e01.evaluate_policy_exact(env, policy, horizon)
        for horizon in horizons
        for policy_name, policy in policies.items()
    }
    unsupported_mass = max(
        float(np.sum(policy * (~env.support)[None]) / max(float(policy.sum()), 1.0))
        for policy in policies.values()
    )
    dataset_rows: list[dict[str, Any]] = []
    policy_accumulator: dict[tuple[Any, ...], dict[str, Any]] = {}
    bootstrap = e02._bootstrap_counts(int(config["logged_episodes_per_dataset"]))
    for dataset_index in range(replicates):
        seed = logged_seed(config, task_index, mechanism_index, dataset_index)
        data = e01.generate_logged_data(env, n=int(config["logged_episodes_per_dataset"]), seed=seed)
        learned, _ = e02.fit_tabular_model(env, data)
        denominators = {
            name: e01.denominator_full_probabilities(env, data, name)
            for name in config["denominators"]
        }
        nuisance = {
            (horizon, policy_name): e01.exact_qv(learned, policy, horizon)
            for horizon in horizons
            for policy_name, policy in policies.items()
        }
        for denominator_name, denominator in denominators.items():
            for clip_label in config["clipping"]:
                clip = None if clip_label == "none" else float(clip_label)
                for horizon in horizons:
                    estimates = {}
                    for policy_name, policy in policies.items():
                        q, v = nuisance[(horizon, policy_name)]
                        estimates[policy_name] = rd01._ope_with_cached_nuisance(
                            env, data, policy, denominator, horizon, clip, bootstrap, q, v
                        )
                    for estimator in config["estimators"]:
                        points = np.asarray([estimates[name][estimator][0] for name in policy_names])
                        lows = np.asarray([estimates[name][estimator][1] for name in policy_names])
                        highs = np.asarray([estimates[name][estimator][2] for name in policy_names])
                        ess = np.asarray([estimates[name][estimator][3] for name in policy_names])
                        true = np.asarray([truths[(horizon, name)] for name in policy_names])
                        covered = np.isfinite(lows) & (lows <= true) & (true <= highs)
                        spearman, pairwise = rd01._rank_metrics(true, points)
                        behavior_index = policy_names.index("empirical_behavior")
                        nonbehavior = np.arange(len(policy_names)) != behavior_index
                        nonzero = nonbehavior & (np.abs(true - true[behavior_index]) > 1e-12)
                        sign = float(np.mean(
                            np.sign(points[nonzero] - points[behavior_index])
                            == np.sign(true[nonzero] - true[behavior_index])
                        )) if np.any(nonzero) else np.nan
                        declared = nonbehavior & (lows > highs[behavior_index])
                        false = declared & (true <= true[behavior_index] + 1e-12)
                        dataset_rows.append({
                            "environment": env.name,
                            "task": task,
                            "mechanism": mechanism,
                            "dataset_index": dataset_index,
                            "logged_dataset_seed": seed,
                            "logged_episodes": len(data["actions"]),
                            "horizon": horizon,
                            "estimator": estimator,
                            "denominator_contract": denominator_name,
                            "clipping_contract": "none" if clip is None else "20",
                            "support_contract": "masked",
                            "policy_set_interval_inclusion_rate": float(covered.mean()),
                            "policy_set_interval_inclusion_status": "diagnostic_only_not_frequentist_coverage",
                            "spearman_rank_recovery": spearman,
                            "pairwise_order_recovery": pairwise,
                            "behavior_relative_sign_recovery": sign,
                            "false_improvement_count": int(false.sum()),
                            "false_improvement_opportunities": int(nonbehavior.sum()),
                            "median_ess": float(np.nanmedian(ess)),
                            "minimum_ess": float(np.nanmin(ess)),
                            "maximum_unsupported_action_mass": unsupported_mass,
                            "nonfinite_estimate_count": int((~np.isfinite(points)).sum()),
                        })
                        for index, policy_name in enumerate(policy_names):
                            key = (
                                env.name,
                                task,
                                mechanism,
                                horizon,
                                estimator,
                                denominator_name,
                                "none" if clip is None else "20",
                                "masked",
                                policy_name,
                            )
                            record = policy_accumulator.setdefault(key, {
                                "trials": 0,
                                "covered": 0,
                                "true_value_min": float(true[index]),
                                "true_value_max": float(true[index]),
                            })
                            record["trials"] += 1
                            record["covered"] += int(covered[index])
                            record["true_value_min"] = min(record["true_value_min"], float(true[index]))
                            record["true_value_max"] = max(record["true_value_max"], float(true[index]))
    fields = (
        "environment",
        "task",
        "mechanism",
        "horizon",
        "estimator",
        "denominator_contract",
        "clipping_contract",
        "support_contract",
        "target_policy",
    )
    policy_rows = [{**dict(zip(fields, key)), **record} for key, record in policy_accumulator.items()]
    return dataset_rows, policy_rows


def summarize(
    policy_summary: pd.DataFrame, dataset_frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    policy_group = [
        "environment",
        "task",
        "mechanism",
        "horizon",
        "estimator",
        "denominator_contract",
        "clipping_contract",
        "support_contract",
        "target_policy",
    ]
    coverage_rows = []
    for key, group in policy_summary.groupby(policy_group, sort=False, dropna=False):
        successes = int(group["covered"].sum())
        trials = int(group["trials"].sum())
        low, high = rd01.wilson_interval(successes, trials)
        coverage_rows.append({
            **dict(zip(policy_group, key)),
            "independent_logged_datasets": trials,
            "intervals_containing_exact_true_value": successes,
            "repeated_dataset_empirical_coverage": successes / trials,
            "binomial_ci_low": low,
            "binomial_ci_high": high,
            "binomial_ci_method": "wilson_95",
            "true_value_invariant": bool(
                float(group["true_value_max"].max()) - float(group["true_value_min"].min()) <= 1e-12
            ),
        })
    coverage = pd.DataFrame(coverage_rows)
    precision = coverage[policy_group + [
        "independent_logged_datasets",
        "intervals_containing_exact_true_value",
        "repeated_dataset_empirical_coverage",
        "binomial_ci_low",
        "binomial_ci_high",
        "binomial_ci_method",
    ]].copy()
    tuple_group = policy_group[:-1]
    rank_rows = []
    authorization_rows = []
    gates = config["authorization_gates"]
    for key, group in dataset_frame.groupby(tuple_group, sort=False, dropna=False):
        selector = np.ones(len(coverage), dtype=bool)
        for column, value in zip(tuple_group, key):
            selector &= coverage[column].to_numpy() == value
        matching = coverage.loc[selector]
        false_opportunities = int(group["false_improvement_opportunities"].sum())
        false_count = int(group["false_improvement_count"].sum())
        false_probability = false_count / false_opportunities if false_opportunities else 0.0
        rank_mean = float(group["spearman_rank_recovery"].mean())
        pairwise_mean = float(group["pairwise_order_recovery"].mean())
        sign_mean = float(group["behavior_relative_sign_recovery"].mean())
        repeated_count = int(group["dataset_index"].nunique())
        rank_rows.append({
            **dict(zip(tuple_group, key)),
            "independent_logged_datasets": repeated_count,
            "mean_spearman_rank_recovery": rank_mean,
            "mean_pairwise_order_recovery": pairwise_mean,
            "mean_behavior_relative_sign_recovery": sign_mean,
            "false_improvement_probability": false_probability,
        })
        coverage_gate = bool(len(matching) and (
            matching["repeated_dataset_empirical_coverage"] >= gates["minimum_empirical_coverage"]
        ).all())
        rank_gate = rank_mean >= gates["minimum_spearman_rank_recovery"]
        pairwise_gate = pairwise_mean >= gates["minimum_pairwise_order_recovery"]
        sign_gate = sign_mean >= gates["minimum_behavior_relative_sign_recovery"]
        median_ess = float(group["median_ess"].median())
        ess_gate = median_ess >= gates["minimum_median_ess"] and median_ess >= (
            gates["minimum_median_ess_fraction"] * int(group["logged_episodes"].iloc[0])
        )
        support_mass = float(group["maximum_unsupported_action_mass"].max())
        support_gate = support_mass <= gates["maximum_unsupported_action_mass"]
        finite_gate = bool((group["nonfinite_estimate_count"] == 0).all())
        false_gate = false_probability <= gates["maximum_false_improvement_probability"]
        authorization_rows.append({
            **dict(zip(tuple_group, key)),
            "independent_logged_datasets": repeated_count,
            "minimum_policy_empirical_coverage": float(matching["repeated_dataset_empirical_coverage"].min()),
            "minimum_policy_coverage_ci_low": float(matching["binomial_ci_low"].min()),
            "coverage_gate": coverage_gate,
            "rank_gate": rank_gate,
            "pairwise_gate": pairwise_gate,
            "sign_gate": sign_gate,
            "false_improvement_probability": false_probability,
            "false_improvement_gate": false_gate,
            "median_ess": median_ess,
            "ess_gate": ess_gate,
            "maximum_unsupported_action_mass": support_mass,
            "support_gate": support_gate,
            "finite_gate": finite_gate,
            "approved_known_value_tuple": bool(
                coverage_gate and rank_gate and pairwise_gate and sign_gate and false_gate
                and ess_gate and support_gate and finite_gate
            ),
            "retrospective_ehr_ope_authorized": False,
            "authorization_boundary": config["authorization_boundary"],
        })
    coverage = coverage.sort_values(policy_group).reset_index(drop=True)
    precision = precision.sort_values(policy_group).reset_index(drop=True)
    rank = pd.DataFrame(rank_rows).sort_values(tuple_group).reset_index(drop=True)
    authorization = pd.DataFrame(authorization_rows).sort_values(tuple_group).reset_index(drop=True)
    return coverage, precision, rank, authorization


def seed_manifest(config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    prior_max = int(config["seed_namespaces"]["prior_repeated_dataset_seed_max"])
    for task_index, (task, count) in enumerate(config["logged_dataset_replicates"].items()):
        for mechanism_index, mechanism in enumerate(config["mechanisms"]):
            start = logged_seed(config, task_index, mechanism_index, 0)
            end = logged_seed(config, task_index, mechanism_index, int(count) - 1)
            rows.append({
                "cell_id": task_index * len(config["mechanisms"]) + mechanism_index,
                "task": task,
                "mechanism": mechanism,
                "replicates": count,
                "logged_dataset_seed_start": start,
                "logged_dataset_seed_end": end,
                "prior_logged_dataset_seed_max": prior_max,
                "prior_logged_dataset_overlap": bool(start <= prior_max),
                "optimization_seed_overlap": bool(set(range(start, end + 1)) & set(config["seed_namespaces"]["optimization_training"])),
                "evaluation_seed_overlap": bool(set(range(start, end + 1)) & set(config["seed_namespaces"]["evaluation_noise"])),
                "bootstrap_seed_overlap": bool(set(range(start, end + 1)) & set(config["seed_namespaces"]["bootstrap_fixed"])),
                "seed_rule": config["seed_namespaces"]["logged_dataset_rule"],
            })
    return pd.DataFrame(rows)


def aggregate_maps(
    authorization: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    task_ids = {task: index for index, task in enumerate(config["logged_dataset_replicates"])}
    mechanism_ids = {name: index for index, name in enumerate(config["mechanisms"])}
    estimator_ids = {name: index for index, name in enumerate(config["estimators"])}
    mechanism_rows = []
    for (task, mechanism), group in authorization.groupby(["task", "mechanism"], sort=False):
        mechanism_rows.append({
            "task": task,
            "mechanism": mechanism,
            "tuple_count": len(group),
            "approved_tuple_count": int(group["approved_known_value_tuple"].sum()),
            "coverage_gate_failures": int((~group["coverage_gate"]).sum()),
            "rank_gate_failures": int((~group["rank_gate"]).sum()),
            "pairwise_gate_failures": int((~group["pairwise_gate"]).sum()),
            "sign_gate_failures": int((~group["sign_gate"]).sum()),
            "false_improvement_gate_failures": int((~group["false_improvement_gate"]).sum()),
            "ess_gate_failures": int((~group["ess_gate"]).sum()),
            "support_gate_failures": int((~group["support_gate"]).sum()),
            "finite_gate_failures": int((~group["finite_gate"]).sum()),
            "minimum_policy_empirical_coverage": float(group["minimum_policy_empirical_coverage"].min()),
            "median_tuple_ess": float(group["median_ess"].median()),
            "retrospective_ehr_ope_authorized": False,
        })
    estimator_rows = []
    for (task, mechanism, estimator), group in authorization.groupby(
        ["task", "mechanism", "estimator"], sort=False
    ):
        estimator_rows.append({
            "cell_id": task_ids[task] * len(mechanism_ids) + mechanism_ids[mechanism],
            "estimator_id": estimator_ids[estimator],
            "task": task,
            "mechanism": mechanism,
            "estimator": estimator,
            "tuple_count": len(group),
            "approved_tuple_count": int(group["approved_known_value_tuple"].sum()),
            "minimum_policy_empirical_coverage": float(group["minimum_policy_empirical_coverage"].min()),
            "mean_false_improvement_probability": float(group["false_improvement_probability"].mean()),
            "median_ess": float(group["median_ess"].median()),
            "all_support_pass": bool(group["support_gate"].all()),
            "all_finite_pass": bool(group["finite_gate"].all()),
            "retrospective_ehr_ope_authorized": False,
        })
    mechanism_frame = pd.DataFrame(mechanism_rows).sort_values(["task", "mechanism"]).reset_index(drop=True)
    estimator_frame = pd.DataFrame(estimator_rows).sort_values(
        ["task", "mechanism", "estimator_id"]
    ).reset_index(drop=True)
    return mechanism_frame, estimator_frame


def interval_parity_valid(coverage: pd.DataFrame) -> bool:
    """Validate Wilson bounds with only IEEE-754 roundoff allowance."""
    columns = ["repeated_dataset_empirical_coverage", "binomial_ci_low", "binomial_ci_high"]
    if not np.isfinite(coverage[columns]).all().all():
        return False
    tolerance = 16.0 * np.finfo(np.float64).eps
    point = coverage["repeated_dataset_empirical_coverage"]
    low = coverage["binomial_ci_low"]
    high = coverage["binomial_ci_high"]
    return bool(
        (low >= -tolerance).all()
        and (high <= 1.0 + tolerance).all()
        and (low <= high + tolerance).all()
        and (low <= point + tolerance).all()
        and (point <= high + tolerance).all()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KDD115 heterogeneous repeated-dataset OPE calibration")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke-replicates", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    parity = verify_frozen_contract(config)
    k107_config = json.loads(KDD107_CONFIG.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    jobs = []
    for task_index, (task, configured_count) in enumerate(config["logged_dataset_replicates"].items()):
        for mechanism_index, mechanism in enumerate(config["mechanisms"]):
            count = int(args.smoke_replicates if args.smoke_replicates is not None else configured_count)
            jobs.append((task, mechanism, task_index, mechanism_index, count, config, k107_config))
    results = []
    failures = []
    if args.workers == 1:
        for job in jobs:
            try:
                results.append(_cell_worker(*job))
            except Exception as exc:
                failures.append({
                    "task": job[0], "mechanism": job[1], "stage": "cell_worker",
                    "failure_type": type(exc).__name__, "detail": str(exc)[:300], "retained": True,
                })
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_cell_worker, *job): job[:2] for job in jobs}
            for future in as_completed(futures):
                task, mechanism = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    failures.append({
                        "task": task, "mechanism": mechanism, "stage": "cell_worker",
                        "failure_type": type(exc).__name__, "detail": str(exc)[:300], "retained": True,
                    })
    dataset_rows = [row for result in results for row in result[0]]
    policy_rows = [row for result in results for row in result[1]]
    if not dataset_rows or not policy_rows:
        raise RuntimeError(f"no completed KDD115 cells: {failures}")
    dataset_frame = pd.DataFrame(dataset_rows).sort_values([
        "task", "mechanism", "dataset_index", "horizon", "estimator",
        "denominator_contract", "clipping_contract",
    ])
    policy_summary = pd.DataFrame(policy_rows)
    coverage, precision, rank, authorization = summarize(policy_summary, dataset_frame, config)
    mechanism_summary, estimator_map = aggregate_maps(authorization, config)
    seeds = seed_manifest(config)
    expected = {
        (task, mechanism): int(args.smoke_replicates if args.smoke_replicates is not None else count)
        for task, count in config["logged_dataset_replicates"].items()
        for mechanism in config["mechanisms"]
    }
    observed = dataset_frame.groupby(["task", "mechanism"])["dataset_index"].nunique().to_dict()
    count_gate = not failures and all(observed.get(cell, 0) == count for cell, count in expected.items())
    truth_gate = bool(coverage["true_value_invariant"].all())
    interval_gate = bool(
        (coverage["independent_logged_datasets"] > 0).all()
        and interval_parity_valid(coverage)
    )
    seed_gate = bool(
        (~seeds[["prior_logged_dataset_overlap", "optimization_seed_overlap", "evaluation_seed_overlap", "bootstrap_seed_overlap"]]).all().all()
        and len(set(dataset_frame["logged_dataset_seed"])) == sum(expected.values())
    )
    if not count_gate or not seed_gate:
        decision = "stop_repeat_count_or_source_drift"
    elif not truth_gate or not interval_gate:
        decision = "stop_interval_or_truth_parity_failure"
    elif int(authorization["approved_known_value_tuple"].sum()) == 0:
        decision = "complete_negative_no_heterogeneous_tuple_approved"
    else:
        decision = "complete_heterogeneous_repeated_dataset_ope_map"
    if decision not in config["allowed_decisions"]:
        raise RuntimeError(decision)
    if not failures:
        failures = [{
            "task": "all", "mechanism": "all", "stage": "all",
            "failure_type": "none", "detail": "No execution failures.", "retained": True,
        }]
    task_id = {task: index for index, task in enumerate(config["logged_dataset_replicates"])}
    mechanism_id = {name: index for index, name in enumerate(config["mechanisms"])}
    estimator_id = {name: index for index, name in enumerate(config["estimators"])}
    denominator_id = {name: index for index, name in enumerate(config["denominators"])}
    clip_id = {"none": 0, "20": 1}
    compact_rows = pd.DataFrame({
        "cell_id": [task_id[task] * len(mechanism_id) + mechanism_id[mechanism]
                    for task, mechanism in zip(dataset_frame["task"], dataset_frame["mechanism"])],
        "dataset_index": dataset_frame["dataset_index"],
        "horizon": dataset_frame["horizon"],
        "estimator_id": dataset_frame["estimator"].map(estimator_id),
        "denominator_id": dataset_frame["denominator_contract"].map(denominator_id),
        "clipping_id": dataset_frame["clipping_contract"].map(clip_id),
        "policy_set_interval_inclusion_rate": dataset_frame["policy_set_interval_inclusion_rate"],
        "spearman_rank_recovery": dataset_frame["spearman_rank_recovery"],
        "pairwise_order_recovery": dataset_frame["pairwise_order_recovery"],
        "behavior_relative_sign_recovery": dataset_frame["behavior_relative_sign_recovery"],
        "false_improvement_count": dataset_frame["false_improvement_count"],
        "false_improvement_opportunities": dataset_frame["false_improvement_opportunities"],
        "median_ess": dataset_frame["median_ess"],
        "minimum_ess": dataset_frame["minimum_ess"],
        "maximum_unsupported_action_mass": dataset_frame["maximum_unsupported_action_mass"],
        "nonfinite_estimate_count": dataset_frame["nonfinite_estimate_count"],
    })
    outputs = {
        "repeated_dataset_ope_rows.csv": compact_rows,
        "repeated_dataset_coverage.csv": coverage,
        "coverage_precision_intervals.csv": precision,
        "ope_rank_and_sign_recovery.csv": rank,
        "ope_authorization_heterogeneous.csv": authorization,
        "mechanism_level_failure_summary.csv": mechanism_summary,
        "estimator_task_mechanism_map.csv": estimator_map,
        "seed_namespace_manifest.csv": seeds,
        "failure_ledger.csv": pd.DataFrame(failures),
    }
    for name, frame in outputs.items():
        precision_format = "%.6g" if name == "repeated_dataset_ope_rows.csv" else "%.10g"
        frame.to_csv(args.output / name, index=False, lineterminator="\n", float_format=precision_format)
    total_datasets = sum(expected.values())
    approved = int(authorization["approved_known_value_tuple"].sum())
    summary = [
        "# KDD115 heterogeneous repeated-dataset OPE extension",
        "",
        f"Decision: `{decision}`",
        "",
        f"- Independent logged datasets: {total_datasets:,}",
        f"- Dataset-level estimator tuple rows: {len(dataset_frame):,}",
        f"- Policy-specific coverage rows: {len(coverage):,}",
        f"- Exact heterogeneous tuple authorizations: {approved:,}/{len(authorization):,}",
        f"- Frozen KDD107 control-policy exact-return parity rows: {len(parity['policy_parity_rows']):,}",
        f"- Runtime seconds: {time.time() - started:.1f}",
        "- Policy-set interval inclusion is diagnostic-only and cannot satisfy empirical coverage.",
        "- The full per-dataset tuple table uses numeric cell/estimator/denominator/clipping IDs; the seed manifest, estimator map, and immutable config provide the lossless contract dictionary.",
        "- Every estimated denominator and nuisance model was refit within its logged dataset; intervals were not pooled across replicates.",
        "- Passing tuples are estimator--denominator--clip--support--horizon--mechanism specific and do not authorize retrospective EHR scoring.",
        f"- {CLAIM_BOUNDARY}",
    ]
    summary_text = "\n".join(summary) + "\n"
    (args.output / "summary.md").write_text(summary_text, encoding="utf-8")
    audit = [
        "# KDD115 result audit",
        "",
        f"Decision: `{decision}`.",
        "",
        f"All {total_datasets:,} planned independent logged datasets were required by the count gate. "
        f"The run produced {len(dataset_frame):,} dataset-level estimator tuples, {len(coverage):,} "
        f"policy-specific empirical-coverage rows, and {len(authorization):,} exact authorization tuples.",
        "",
        "Frozen KDD107 environment contracts and five shared control-policy exact returns were checked before execution. "
        "The completed repeated-dataset estimator, denominator, clipping, support, interval, horizon, policy-set, and authorization gates were unchanged.",
        "",
        "The result is exact-finite known-construction evaluator calibration only. It does not authorize retrospective EHR OPE or support treatment, causal, clinical-utility, deployment, policy-superiority, or cross-disease generalization claims.",
    ]
    (args.output / "result_audit.md").write_text("\n".join(audit) + "\n", encoding="utf-8")
    missing = [name for name in REQUIRED_OUTPUTS if not (args.output / name).is_file()]
    if missing:
        raise RuntimeError(f"missing required outputs: {missing}")
    manifest = {path.name: sha256(path) for path in sorted(args.output.iterdir()) if path.is_file()}
    (args.output / "artifact_hashes.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"decision={decision} datasets={total_datasets} rows={len(dataset_frame)} "
        f"coverage={len(coverage)} approved={approved}/{len(authorization)}"
    )


if __name__ == "__main__":
    main()
