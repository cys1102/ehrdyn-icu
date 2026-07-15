from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from kdd_benchmark_discovery import kdd_e01_evaluator as e01
from kdd_benchmark_discovery import run_kdd_adapt01_adaptive_known_value as adapt
from kdd_benchmark_discovery import run_kdd_e02_known_value_full as e02


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/kdd_ope_rd01_repeated_dataset_v1.json"
ADAPT_CONFIG = ROOT / "configs/kdd_adapt01_adaptive_known_value_v1.json"
HISTORICAL_DIAGNOSTIC = (
    ROOT
    / "kdd_benchmark_discovery/results/kdd_e02_known_value_full_20260714_190217"
    / "ope_coverage_false_improvement.csv"
)
REQUIRED = (
    "repeated_dataset_ope_rows.csv",
    "repeated_dataset_coverage.csv",
    "coverage_precision_intervals.csv",
    "policy_set_interval_inclusion_diagnostic.csv",
    "ope_rank_and_sign_recovery.csv",
    "ope_authorization_revised.csv",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources(config: dict[str, Any]) -> None:
    paths = {
        "adapt01_config": ADAPT_CONFIG,
        "adapt01_runner": ROOT / "kdd_benchmark_discovery/run_kdd_adapt01_adaptive_known_value.py",
        "adapt01_environment_contract": (
            ROOT
            / "kdd_benchmark_discovery/results/kdd_adapt01_adaptive_known_value_20260715_092634"
            / "adaptive_environment_contract.csv"
        ),
        "e01_evaluator": ROOT / "kdd_benchmark_discovery/kdd_e01_evaluator.py",
        "e02_runner": ROOT / "kdd_benchmark_discovery/run_kdd_e02_known_value_full.py",
        "historical_fixed_dataset_coverage": HISTORICAL_DIAGNOSTIC,
    }
    actual = {name: _sha256(path) for name, path in paths.items()}
    if actual != config["immutable_source_hashes"]:
        raise RuntimeError(f"immutable source drift: {actual}")


def fixed_policy(env: e01.FiniteMDP, action: int) -> np.ndarray:
    return adapt.fixed_policy(env, action)


def target_policies(env: e01.FiniteMDP, layout: adapt.Layout, task: str) -> dict[str, np.ndarray]:
    supported = np.flatnonzero(env.support[:-1].all(axis=0))
    _, oracle, _ = e01.backward_induction(env)
    policies = {
        "empirical_behavior": env.behavior.copy(),
        "random_supported": e01.support_aware_stochastic_policy(env),
        "minimum_supported_action": fixed_policy(env, int(supported[0])),
        "maximum_supported_action": fixed_policy(env, int(supported[-1])),
        "severity_rule": adapt.severity_rule(env, layout, adapt.action_intensity(task, env.n_actions)),
        "exact_dynamic_programming_oracle": oracle,
    }
    for name, policy in policies.items():
        if policy.shape != env.behavior.shape:
            raise RuntimeError(f"policy shape mismatch: {name}: {policy.shape}")
        if float(np.sum(policy * (~env.support)[None])) != 0.0:
            raise RuntimeError(f"support-mask bypass: {name}")
    return policies


def _bootstrap_counts(n: int) -> np.ndarray:
    return e02._bootstrap_counts(n)


def _ope_with_cached_nuisance(
    env: e01.FiniteMDP,
    data: dict[str, np.ndarray],
    target: np.ndarray,
    denominator_full: np.ndarray,
    horizon: int,
    clip: float | None,
    bootstrap_counts: np.ndarray,
    q: np.ndarray,
    v: np.ndarray,
) -> dict[str, tuple[float, float, float, float, float]]:
    h = horizon
    states = data["states"][:, : h + 1]
    actions = data["actions"][:, :h]
    rewards = data["rewards"][:, :h]
    target_prob = np.empty(actions.shape, dtype=np.float64)
    for t in range(h):
        target_prob[:, t] = target[t, states[:, t], actions[:, t]]
    denominator = np.take_along_axis(
        denominator_full[:, :h], actions[:, :, None], axis=2
    )[:, :, 0]
    ratio = np.divide(target_prob, denominator, out=np.zeros_like(target_prob), where=denominator > 0)
    if clip is not None:
        ratio = np.minimum(ratio, clip)
    cumulative = np.cumprod(ratio, axis=1)
    discount = env.discount ** np.arange(h)
    returns = rewards @ discount
    final = cumulative[:, -1]
    ess = float(final.sum() ** 2 / max(np.square(final).sum(), 1e-12))
    concentration = float(final.max() / max(final.sum(), 1e-12))
    b = bootstrap_counts
    estimates: dict[str, tuple[float, np.ndarray]] = {}
    is_contrib = final * returns
    estimates["IS"] = (float(is_contrib.mean()), b @ is_contrib / b.sum(axis=1))
    estimates["WIS"] = (
        float(np.sum(final * returns) / max(final.sum(), 1e-12)),
        (b @ (final * returns)) / np.maximum(b @ final, 1e-12),
    )
    pdis_contrib = np.sum(cumulative * rewards * discount, axis=1)
    estimates["PDIS"] = (float(pdis_contrib.mean()), b @ pdis_contrib / b.sum(axis=1))
    step_num = cumulative * rewards
    step_den = cumulative
    wp_point = float(
        np.sum(
            np.divide(
                step_num.sum(axis=0),
                step_den.sum(axis=0),
                out=np.zeros(h),
                where=step_den.sum(axis=0) > 0,
            )
            * discount
        )
    )
    wp_boot = np.zeros(len(b), dtype=np.float64)
    for t in range(h):
        wp_boot += discount[t] * (b @ step_num[:, t]) / np.maximum(b @ step_den[:, t], 1e-12)
    estimates["WPDIS"] = (wp_point, wp_boot)
    at_risk = states[:, :-1] != env.n_states - 1
    cw_num = cumulative * rewards * at_risk
    cw_den = cumulative * at_risk
    cw_point = float(
        np.sum(
            np.divide(
                cw_num.sum(axis=0),
                cw_den.sum(axis=0),
                out=np.zeros(h),
                where=cw_den.sum(axis=0) > 0,
            )
            * discount
        )
    )
    cw_boot = np.zeros(len(b), dtype=np.float64)
    for t in range(h):
        cw_boot += discount[t] * (b @ cw_num[:, t]) / np.maximum(b @ cw_den[:, t], 1e-12)
    estimates["CWPDIS"] = (cw_point, cw_boot)
    dr_contrib = v[0, states[:, 0]].copy()
    for t in range(h):
        residual = rewards[:, t] + env.discount * v[t + 1, states[:, t + 1]] - q[t, states[:, t], actions[:, t]]
        dr_contrib += discount[t] * cumulative[:, t] * residual
    estimates["DR"] = (float(dr_contrib.mean()), b @ dr_contrib / b.sum(axis=1))
    wdr_point = float(v[0, states[:, 0]].mean())
    wdr_boot = b @ v[0, states[:, 0]] / b.sum(axis=1)
    for t in range(h):
        residual = rewards[:, t] + env.discount * v[t + 1, states[:, t + 1]] - q[t, states[:, t], actions[:, t]]
        weighted = cumulative[:, t] * residual
        wdr_point += discount[t] * float(weighted.sum() / max(cumulative[:, t].sum(), 1e-12))
        wdr_boot += discount[t] * (b @ weighted) / np.maximum(b @ cumulative[:, t], 1e-12)
    estimates["WDR"] = (wdr_point, wdr_boot)
    fqe_episode = v[0, states[:, 0]]
    estimates["FQE"] = (float(fqe_episode.mean()), b @ fqe_episode / b.sum(axis=1))
    estimates["support_restricted_WPDIS"] = estimates["WPDIS"]
    output: dict[str, tuple[float, float, float, float, float]] = {}
    for name, (point, samples) in estimates.items():
        finite = samples[np.isfinite(samples)]
        if len(finite):
            low, high = (float(value) for value in np.quantile(finite, (0.025, 0.975)))
        else:
            low, high = np.nan, np.nan
        output[name] = (point, low, high, ess, concentration)
    return output


def _rank_metrics(true: np.ndarray, estimate: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(estimate)
    if finite.sum() < 2 or np.ptp(true[finite]) <= 1e-12:
        return np.nan, np.nan
    true_rank = rankdata(true[finite])
    estimate_rank = rankdata(estimate[finite])
    spearman = float(np.corrcoef(true_rank, estimate_rank)[0, 1])
    pairs = [
        (i, j)
        for i in range(len(true))
        for j in range(i + 1, len(true))
        if finite[i] and finite[j] and abs(true[i] - true[j]) > 1e-12
    ]
    pairwise = float(
        np.mean([np.sign(true[i] - true[j]) == np.sign(estimate[i] - estimate[j]) for i, j in pairs])
    )
    return spearman, pairwise


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    if trials <= 0:
        return np.nan, np.nan
    z = 1.959963984540054 if confidence == 0.95 else 1.959963984540054
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _cell_worker(
    task: str,
    regime: str,
    task_index: int,
    regime_index: int,
    replicates: int,
    config: dict[str, Any],
    adapt_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import torch

    torch.set_num_threads(1)
    profile = adapt_config["tasks"][task]
    env, layout, _ = adapt.build_environment(task, profile, regime, adapt_config["mechanisms"])
    policies = target_policies(env, layout, task)
    policy_names = list(policies)
    horizons = [int(value) for value in config["task_horizons"][task]]
    truths = {
        (horizon, policy_name): e01.evaluate_policy_exact(env, policy, horizon)
        for horizon in horizons
        for policy_name, policy in policies.items()
    }
    dataset_rows: list[dict[str, Any]] = []
    policy_accumulator: dict[tuple[Any, ...], dict[str, Any]] = {}
    bootstrap = _bootstrap_counts(int(config["logged_episodes_per_dataset"]))
    for dataset_index in range(replicates):
        seed = 88_010_000 + task_index * 1_000_000 + regime_index * 100_000 + dataset_index
        data = e01.generate_logged_data(
            env, n=int(config["logged_episodes_per_dataset"]), seed=seed
        )
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
                        estimates[policy_name] = _ope_with_cached_nuisance(
                            env, data, policy, denominator, horizon, clip, bootstrap, q, v
                        )
                    for estimator in config["estimators"]:
                        points = np.asarray([estimates[name][estimator][0] for name in policy_names])
                        lows = np.asarray([estimates[name][estimator][1] for name in policy_names])
                        highs = np.asarray([estimates[name][estimator][2] for name in policy_names])
                        ess = np.asarray([estimates[name][estimator][3] for name in policy_names])
                        true = np.asarray([truths[(horizon, name)] for name in policy_names])
                        covered = np.isfinite(lows) & (lows <= true) & (true <= highs)
                        spearman, pairwise = _rank_metrics(true, points)
                        behavior_index = policy_names.index("empirical_behavior")
                        nonbehavior = np.arange(len(policy_names)) != behavior_index
                        nonzero = nonbehavior & (np.abs(true - true[behavior_index]) > 1e-12)
                        sign = (
                            float(
                                np.mean(
                                    np.sign(points[nonzero] - points[behavior_index])
                                    == np.sign(true[nonzero] - true[behavior_index])
                                )
                            )
                            if np.any(nonzero)
                            else np.nan
                        )
                        declared = nonbehavior & (lows > highs[behavior_index])
                        false = declared & (true <= true[behavior_index] + 1e-12)
                        dataset_rows.append(
                            {
                                "environment": env.name,
                                "task": task,
                                "response_regime": regime,
                                "dataset_index": dataset_index,
                                "logged_dataset_seed": seed,
                                "logged_episodes": len(data["actions"]),
                                "horizon": horizon,
                                "estimator": estimator,
                                "denominator_contract": denominator_name,
                                "clipping_contract": "none" if clip is None else "20",
                                "support_contract": "masked",
                                "policy_set_interval_inclusion_rate": float(covered.mean()),
                                "spearman_rank_recovery": spearman,
                                "pairwise_order_recovery": pairwise,
                                "behavior_relative_sign_recovery": sign,
                                "false_improvement_count": int(false.sum()),
                                "false_improvement_opportunities": int(nonbehavior.sum()),
                                "median_ess": float(np.nanmedian(ess)),
                                "minimum_ess": float(np.nanmin(ess)),
                                "nonfinite_estimate_count": int((~np.isfinite(points)).sum()),
                            }
                        )
                        for index, policy_name in enumerate(policy_names):
                            key = (
                                env.name,
                                task,
                                regime,
                                horizon,
                                estimator,
                                denominator_name,
                                "none" if clip is None else "20",
                                "masked",
                                policy_name,
                            )
                            record = policy_accumulator.setdefault(
                                key,
                                {
                                    "trials": 0,
                                    "covered": 0,
                                    "true_value_min": float(true[index]),
                                    "true_value_max": float(true[index]),
                                },
                            )
                            record["trials"] += 1
                            record["covered"] += int(covered[index])
                            record["true_value_min"] = min(record["true_value_min"], float(true[index]))
                            record["true_value_max"] = max(record["true_value_max"], float(true[index]))
    policy_summary_rows = []
    fields = (
        "environment",
        "task",
        "response_regime",
        "horizon",
        "estimator",
        "denominator_contract",
        "clipping_contract",
        "support_contract",
        "target_policy",
    )
    for key, record in policy_accumulator.items():
        policy_summary_rows.append({**dict(zip(fields, key)), **record})
    return dataset_rows, policy_summary_rows


def _historical_diagnostic() -> pd.DataFrame:
    frame = pd.read_csv(HISTORICAL_DIAGNOSTIC)
    frame = frame.rename(columns={"coverage_rate": "policy_set_interval_inclusion_rate"})
    frame.insert(0, "metric_status", "diagnostic_only_not_frequentist_coverage")
    frame["authorization_use"] = "prohibited"
    frame["estimand"] = "within_one_fixed_logged_dataset_fraction_of_prespecified_policy_intervals_containing_policy_truth"
    return frame


def summarize(policy_summary: pd.DataFrame, dataset_frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    policy_group = [
        "environment",
        "task",
        "response_regime",
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
        low, high = wilson_interval(successes, trials)
        coverage_rows.append(
            {
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
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    precision = coverage[
        policy_group
        + [
            "independent_logged_datasets",
            "intervals_containing_exact_true_value",
            "repeated_dataset_empirical_coverage",
            "binomial_ci_low",
            "binomial_ci_high",
            "binomial_ci_method",
        ]
    ].copy()

    tuple_group = policy_group[:-1]
    rank_rows = []
    authorization_rows = []
    gates = config["authorization_gates"]
    for key, group in dataset_frame.groupby(tuple_group, sort=False, dropna=False):
        selector = np.ones(len(coverage), dtype=bool)
        for column, value in zip(tuple_group, key):
            selector &= coverage[column].to_numpy() == value
        matching_coverage = coverage.loc[selector]
        regime = str(key[2])
        false_opportunities = int(group["false_improvement_opportunities"].sum())
        false_count = int(group["false_improvement_count"].sum())
        false_probability = false_count / false_opportunities if false_opportunities else 0.0
        rank_mean = float(group["spearman_rank_recovery"].mean())
        pairwise_mean = float(group["pairwise_order_recovery"].mean())
        sign_mean = float(group["behavior_relative_sign_recovery"].mean())
        rank_applicable = regime != "null_response"
        repeated_count = int(group["dataset_index"].nunique())
        rank_rows.append(
            {
                **dict(zip(tuple_group, key)),
                "independent_logged_datasets": repeated_count,
                "mean_spearman_rank_recovery": rank_mean,
                "mean_pairwise_order_recovery": pairwise_mean,
                "mean_behavior_relative_sign_recovery": sign_mean,
                "false_improvement_probability": false_probability,
                "rank_and_sign_applicable": rank_applicable,
                "null_response_note": "no_nonzero_true_order_or_behavior_relative_sign" if not rank_applicable else "",
            }
        )
        coverage_gate = bool(
            len(matching_coverage)
            and (matching_coverage["repeated_dataset_empirical_coverage"] >= gates["minimum_empirical_coverage"]).all()
        )
        rank_gate = (not rank_applicable) or rank_mean >= gates["minimum_spearman_rank_recovery"]
        pairwise_gate = (not rank_applicable) or pairwise_mean >= gates["minimum_pairwise_order_recovery"]
        sign_gate = (not rank_applicable) or sign_mean >= gates["minimum_behavior_relative_sign_recovery"]
        median_ess = float(group["median_ess"].median())
        ess_gate = median_ess >= gates["minimum_median_ess"] and median_ess >= gates["minimum_median_ess_fraction"] * int(group["logged_episodes"].iloc[0])
        finite_gate = bool((group["nonfinite_estimate_count"] == 0).all())
        authorization_rows.append(
            {
                **dict(zip(tuple_group, key)),
                "independent_logged_datasets": repeated_count,
                "minimum_policy_empirical_coverage": float(matching_coverage["repeated_dataset_empirical_coverage"].min()),
                "minimum_policy_coverage_ci_low": float(matching_coverage["binomial_ci_low"].min()),
                "coverage_gate": coverage_gate,
                "rank_gate": rank_gate,
                "pairwise_gate": pairwise_gate,
                "sign_gate": sign_gate,
                "false_improvement_probability": false_probability,
                "false_improvement_gate": false_probability <= gates["maximum_false_improvement_probability"],
                "median_ess": median_ess,
                "ess_gate": ess_gate,
                "support_gate": True,
                "finite_gate": finite_gate,
                "approved_known_value_tuple": bool(
                    coverage_gate
                    and rank_gate
                    and pairwise_gate
                    and sign_gate
                    and false_probability <= gates["maximum_false_improvement_probability"]
                    and ess_gate
                    and finite_gate
                ),
                "retrospective_ehr_ope_authorized": False,
                "authorization_boundary": config["authorization_boundary"],
            }
        )
    return coverage, precision, pd.DataFrame(rank_rows), pd.DataFrame(authorization_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke-replicates", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    adapt_config = json.loads(ADAPT_CONFIG.read_text(encoding="utf-8"))
    verify_sources(config)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    jobs = []
    for task_index, task in enumerate(config["logged_dataset_replicates"]):
        for regime_index, regime in enumerate(config["response_regimes"]):
            replicates = int(config["logged_dataset_replicates"][task])
            if args.smoke_replicates is not None:
                replicates = int(args.smoke_replicates)
            jobs.append((task, regime, task_index, regime_index, replicates, config, adapt_config))
    all_dataset_rows: list[dict[str, Any]] = []
    all_policy_summary_rows: list[dict[str, Any]] = []
    if args.workers == 1:
        results = [_cell_worker(*job) for job in jobs]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_cell_worker, *job): job[:2] for job in jobs}
            for future in as_completed(futures):
                results.append(future.result())
    for dataset_rows, policy_rows in results:
        all_dataset_rows.extend(dataset_rows)
        all_policy_summary_rows.extend(policy_rows)
    dataset_frame = pd.DataFrame(all_dataset_rows).sort_values(
        ["task", "response_regime", "dataset_index", "horizon", "estimator", "denominator_contract", "clipping_contract"]
    )
    policy_summary = pd.DataFrame(all_policy_summary_rows)
    coverage, precision, rank, authorization = summarize(policy_summary, dataset_frame, config)
    dataset_frame.to_csv(args.output / REQUIRED[0], index=False, lineterminator="\n", float_format="%.10g")
    coverage.to_csv(args.output / REQUIRED[1], index=False, lineterminator="\n", float_format="%.10g")
    precision.to_csv(args.output / REQUIRED[2], index=False, lineterminator="\n", float_format="%.10g")
    _historical_diagnostic().to_csv(args.output / REQUIRED[3], index=False, lineterminator="\n", float_format="%.10g")
    rank.to_csv(args.output / REQUIRED[4], index=False, lineterminator="\n", float_format="%.10g")
    authorization.to_csv(args.output / REQUIRED[5], index=False, lineterminator="\n", float_format="%.10g")
    expected_datasets = {
        task: int(args.smoke_replicates if args.smoke_replicates is not None else count)
        for task, count in config["logged_dataset_replicates"].items()
    }
    observed = dataset_frame.groupby(["task", "response_regime"])["dataset_index"].nunique().to_dict()
    count_gate = all(
        observed.get((task, regime), 0) == count
        for task, count in expected_datasets.items()
        for regime in config["response_regimes"]
    )
    truth_gate = bool(coverage["true_value_invariant"].all())
    decision = "complete_repeated_dataset_ope_validation" if count_gate and truth_gate else "stop_truth_or_repeat_count_invalid"
    summary = [
        "# KDD-OPE-RD01 repeated-dataset OPE validation",
        "",
        f"Decision: `{decision}`",
        "",
        f"- Independent logged datasets: {sum(expected_datasets.values()) * len(config['response_regimes']):,}",
        f"- Per-dataset full-tuple rows: {len(dataset_frame):,}",
        f"- Coverage rows: {len(coverage):,}",
        f"- Revised known-value tuple authorizations: {int(authorization['approved_known_value_tuple'].sum()):,}/{len(authorization):,}",
        f"- Runtime seconds: {time.time() - started:.1f}",
        "- `policy_set_interval_inclusion_rate` is retained only as a diagnostic and is not interpreted as frequentist coverage.",
        "- Every empirical-coverage interval is constructed independently within one logged dataset and compared with exact dynamic-programming truth; the reported precision interval is Wilson-binomial across datasets.",
        "- No tuple authorizes retrospective EHR OPE; separate trained-policy, probability, overlap, and task-contract gates remain required.",
        "- No EHR rows, test roles, patient identifiers, timestamps, trajectories, tensors, or checkpoints were accessed or exported.",
    ]
    (args.output / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    hashes = {path.name: _sha256(path) for path in sorted(args.output.iterdir()) if path.is_file()}
    (args.output / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
