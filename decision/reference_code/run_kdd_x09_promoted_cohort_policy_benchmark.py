from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from kdd_benchmark_discovery import kdd_e01_evaluator as e01
from kdd_benchmark_discovery import run_kdd100_complete_known_value as kv
from kdd_benchmark_discovery import run_kdd100r_task_matched_known_value as k100r
from kdd_benchmark_discovery import run_kdd_x02_cross_cohort_policy_benchmark as x02
from kdd_benchmark_discovery.run_kdd_x08_task_matched_evaluator import make_aki_env, make_hf_env


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/kdd_x09_promoted_cohort_policy_benchmark_v1.json"
X08 = ROOT / "kdd_benchmark_discovery/results/kdd_x08_task_matched_evaluator_20260715_061859"
S03 = ROOT / "kdd_benchmark_discovery/results/kdd_s03_canonical_sepsis_baselines_20260714_194100"
X05 = ROOT / "kdd_benchmark_discovery/results/kdd_x05_aki_rrt_policy_task_20260715_050038"
X06 = ROOT / "kdd_benchmark_discovery/results/kdd_x06_hf_policy_feasibility_20260715_062500"
X02_CONFIG = ROOT / "configs/kdd_x02_cross_cohort_policy_benchmark_v1.json"
CLAIM = (
    "Known-value results use injected mechanisms. Retrospective rows are development-only aggregate "
    "support and observability diagnostics. No treatment benefit, causal effect, real-EHR policy "
    "superiority, clinical utility, deployment, or autonomous-decision claim is supported."
)
REQUIRED = (
    "promoted_task_roles.csv",
    "model_free_results.csv",
    "world_model_results.csv",
    "world_model_planner_matrix.csv",
    "known_value_policy_results.csv",
    "retrospective_policy_diagnostics.csv",
    "real_ehr_ope_or_nonexecution.csv",
    "support_collapse_seed_stability.csv",
    "cross_task_rank_stability.csv",
    "failure_rate_summary.csv",
    "decision.md",
    "summary.md",
)
BUILDERS = {"aki_rrt": make_aki_env, "heart_failure": make_hf_env}
REGIMES = ("null", "weak", "moderate", "delayed")
COLLAPSE_SHARE = 0.95
MATERIAL_SEED_TV = 0.25
CONTROL_METHODS = {
    "empirical_behavior",
    "random_supported",
    "minimum_supported_action",
    "maximum_supported_action",
    "severity_rule",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"empty required artifact: {path.name}")
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")


def preflight(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected = config["immutable_input_hashes"]
    actual = {
        "x08_decision": sha256(X08 / "decision.md"),
        "x08_authorization": sha256(X08 / "real_ehr_scoring_authorization.csv"),
        "x08_estimator_disposition": sha256(X08 / "estimator_contract_disposition.csv"),
        "s03_fidelity_registry": sha256(S03 / "baseline_implementation_fidelity.csv"),
        "x02_config": sha256(X02_CONFIG),
        "x05_decision": sha256(X05 / "decision.md"),
        "x06_decision": sha256(X06 / "decision.md"),
    }
    if actual != expected:
        raise RuntimeError(f"immutable input hash drift: {actual}")
    if "`task_matched_evaluation_complete`" not in (X08 / "decision.md").read_text(encoding="utf-8"):
        raise RuntimeError("KDD-X08 evaluator defect or decision drift")
    authorization = pd.read_csv(X08 / "real_ehr_scoring_authorization.csv")
    if authorization.task.tolist() != config["authorized_tasks"]:
        raise RuntimeError("KDD-X08 task authorization drift")
    if int(authorization.loc[authorization.task.eq("aki_rrt"), "tier2_approved_exact_tuple_count"].iloc[0]) != 236:
        raise RuntimeError("AKI exact tuple count drift")
    if int(authorization.loc[authorization.task.eq("heart_failure"), "tier2_approved_exact_tuple_count"].iloc[0]) != 0:
        raise RuntimeError("unexpected HF exact tuple authorization")
    fidelity = pd.read_csv(S03 / "baseline_implementation_fidelity.csv")
    expected_labels = config["fidelity_labels"]
    aliases = {"decision_transformer_adapter": "decision_transformer", "dreamer_v3_categorical_rssm": "categorical_rssm"}
    for method, label in expected_labels.items():
        if method in {"gaussian_recurrent_ensemble", "integrated_rssm_planner"}:
            continue
        source = aliases.get(method, method)
        row = fidelity[fidelity.method.eq(source)]
        if row.empty or str(row.fidelity_label.iloc[0]) != label:
            raise RuntimeError(f"fidelity drift for {method}")
    x02_config = json.loads(X02_CONFIG.read_text(encoding="utf-8"))
    if config["model_free_core"] != x02_config["model_free_core"] or config["world_model_core"] != x02_config["world_model_core"]:
        raise RuntimeError("baseline inventory expanded or changed")
    if config["diagnostic_thresholds"] != {
        "single_action_collapse_share": COLLAPSE_SHARE,
        "material_seed_total_variation": MATERIAL_SEED_TV,
    }:
        raise RuntimeError("diagnostic threshold drift")
    return authorization, fidelity


def task_environment(task: str, regime: str) -> e01.FiniteMDP:
    return BUILDERS[task]("tier2_ehr_calibrated_semisynthetic", regime)


def global_support(env: e01.FiniteMDP) -> np.ndarray:
    support = env.support[:-1].all(axis=0)
    if not support.any():
        raise RuntimeError(f"no globally supported action for {env.name}")
    return support


def output_regime_label(regime: str) -> str:
    return "null_response" if regime == "null" else regime


def canonicalize_policy(policy: np.ndarray, env: e01.FiniteMDP) -> np.ndarray:
    result = np.asarray(policy, dtype=np.float64).copy()
    if result.shape != env.behavior.shape:
        raise ValueError(f"policy shape {result.shape} != {env.behavior.shape}")
    for t in range(env.horizon):
        for state in range(env.n_states):
            allowed = np.flatnonzero(env.support[state])
            result[t, state, ~env.support[state]] = 0.0
            total = float(result[t, state, allowed].sum())
            if not np.isfinite(total) or total <= 0:
                result[t, state, allowed[0]] = 1.0
                result[t, state, allowed[1:]] = 0.0
            else:
                result[t, state, allowed] /= total
                last = int(allowed[-1])
                result[t, state, last] = 1.0 - float(result[t, state, allowed[:-1]].sum())
    if not np.allclose(result.sum(axis=-1), 1.0, atol=1e-15, rtol=0.0):
        raise RuntimeError("policy simplex normalization failed")
    return result


def policy_diagnostic(policy: np.ndarray, env: e01.FiniteMDP, behavior: np.ndarray) -> dict[str, float | bool]:
    supported_mass = float((policy * (~env.support)[None]).sum() / max(policy.sum(), 1e-12))
    marginal = policy[:, :-1].mean(axis=(0, 1))
    marginal /= max(marginal.sum(), 1e-12)
    positive = marginal[marginal > 0]
    entropy = float(-np.sum(positive * np.log(positive)) / max(np.log(env.n_actions), 1e-12))
    b = behavior[:, :-1].mean(axis=(0, 1))
    b /= max(b.sum(), 1e-12)
    kl = float(np.sum(marginal * np.log(np.clip(marginal, 1e-12, 1.0) / np.clip(b, 1e-12, 1.0))))
    argmax = np.argmax(policy[:, :-1], axis=-1)
    top_share = float(pd.Series(argmax.reshape(-1)).value_counts(normalize=True).iloc[0])
    return {
        "unsupported_action_mass": supported_mass,
        "normalized_entropy": entropy,
        "behavior_divergence_kl": kl,
        "top_argmax_action_share": top_share,
        "single_action_collapse": top_share >= COLLAPSE_SHARE,
        "policy_probability_complete": bool(np.allclose(policy.sum(axis=-1), 1.0)),
    }


def seed_stability_rows(policies: dict[tuple[str, int, str], np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    keys = sorted({(task, method) for task, _, method in policies})
    for task, method in keys:
        members = [(seed, policies[(task, seed, method)]) for seed in (3408, 3411, 3414) if (task, seed, method) in policies]
        tv, agreement = [], []
        for i, (_, left) in enumerate(members):
            for _, right in members[i + 1:]:
                tv.append(float(0.5 * np.abs(left - right).sum(axis=-1).mean()))
                agreement.append(float((np.argmax(left, axis=-1) == np.argmax(right, axis=-1)).mean()))
        rows.append({
            "row_type": "cross_seed_stability",
            "task": task,
            "method": method,
            "seed": "3408|3411|3414",
            "mean_pairwise_total_variation": float(np.mean(tv)) if tv else math.nan,
            "mean_pairwise_argmax_agreement": float(np.mean(agreement)) if agreement else math.nan,
            "material_seed_instability": bool(tv and np.mean(tv) > MATERIAL_SEED_TV),
            "claim_boundary": CLAIM,
        })
    return rows


def retrospective_rows() -> list[dict[str, Any]]:
    aki_support = pd.read_csv(X05 / "aki_rrt_action_support.csv")
    aki_reward = pd.read_csv(X05 / "aki_reward_candidates.csv")
    aki_axes = pd.read_csv(X05 / "aki_policy_evaluability_axes.csv")
    hf_support = pd.read_csv(X06 / "hf_repeated_decision_support.csv")
    hf_reward = pd.read_csv(X06 / "hf_reward_candidates.csv")
    hf_axes = pd.read_csv(X06 / "hf_policy_evaluability_axes.csv")
    aki_train = aki_support[aki_support.role.eq("train")]
    hf_train = hf_support[
        hf_support.role.eq("train")
        & hf_support.action_contract.eq("loop_diuretic_4_x_vaso_inotrope_2")
        & hf_support.row_type.eq("action_class_support")
    ]
    aki_reward_row = aki_reward[
        aki_reward.role.eq("train") & aki_reward.reward_candidate.eq("in_hospital_survival_discharge")
    ].iloc[0]
    hf_reward_row = hf_reward[
        hf_reward.role.eq("train")
        & hf_reward.action_contract.eq("loop_diuretic_4_x_vaso_inotrope_2")
        & hf_reward.interval_hours.eq(6)
        & hf_reward.reward_candidate.eq("urine_output_response")
    ].iloc[0]
    return [
        {
            "task": "aki_rrt",
            "evidence_surface": "retrospective_development_materialization_aggregate",
            "action_classes": 2,
            "train_decisions": int(aki_train.transitions.sum()),
            "minimum_train_subjects_per_action": int(aki_train.subjects.min()),
            "primary_reward_observed_fraction": float(aki_reward_row.observed_fraction),
            "behavior_denominator_gate": bool(aki_axes.loc[aki_axes.axis.eq("behavior_denominator"), "pass"].dropna().astype(bool).all()),
            "real_ehr_target_policy_trained": False,
            "retrospective_policy_value_computed": False,
            "test_or_lockbox_accessed": False,
            "claim_boundary": CLAIM,
        },
        {
            "task": "heart_failure",
            "evidence_surface": "retrospective_development_materialization_aggregate",
            "action_classes": 8,
            "train_decisions": int(hf_train.decisions.sum()),
            "minimum_train_subjects_per_action": int(hf_train.subjects.min()),
            "primary_reward_observed_fraction": float(hf_reward_row.observed_fraction),
            "behavior_denominator_gate": bool(hf_axes.loc[hf_axes.axis.eq("behavior_support_pre_gate"), "pass"].astype(bool).all()),
            "real_ehr_target_policy_trained": False,
            "retrospective_policy_value_computed": False,
            "test_or_lockbox_accessed": False,
            "claim_boundary": CLAIM,
        },
    ]


def make_ensemble(members: list[kv.WorldModelFit], seed: int) -> kv.WorldModelFit:
    fingerprints = ";".join(member.fingerprint for member in members)
    return kv.WorldModelFit(
        "gaussian_recurrent_ensemble",
        seed,
        tuple(member.model for member in members),
        float(np.mean([member.validation_rmse for member in members])),
        float(np.mean([member.validation_mae for member in members])),
        float(np.mean([member.nll for member in members])),
        float(np.mean([member.coverage90 for member in members])),
        float(np.mean([member.rollout_rmse for member in members])),
        float(np.mean([member.reward_rmse for member in members])),
        float(np.nanmean([member.termination_auc for member in members])),
        float(np.mean([member.uncertainty_ece for member in members])),
        int(sum(member.parameter_count for member in members)),
        float(sum(member.training_seconds for member in members)),
        float(max(member.peak_memory_mb for member in members)),
        "derived_three_member_recurrent_ensemble_seeds_3408_3411_3414",
        hashlib.sha256(fingerprints.encode()).hexdigest(),
    )


def run(output: Path, config_path: Path) -> str:
    if output.exists():
        raise FileExistsError(output)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    authorization, _ = preflight(config)
    output.mkdir(parents=True)
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failures: list[dict[str, Any]] = []
    policy_store: dict[tuple[str, int, str], np.ndarray] = {}
    planner_truth: dict[tuple[str, int, str], tuple[float, float]] = {}
    k100r.TRAINING = dict(config["world_model_training"])

    for _, item in authorization.iterrows():
        task = str(item.task)
        rows["promoted_task_roles.csv"].append({
            "task": task,
            "materialization_gate_passed": True,
            "x08_tier1_approved_exact_tuple_count": int(item.tier1_approved_exact_tuple_count),
            "x08_tier2_approved_exact_tuple_count": int(item.tier2_approved_exact_tuple_count),
            "x08_real_ehr_policy_value_status": item.real_ehr_policy_value_status,
            "known_value_policy_comparison_authorized": True,
            "real_ehr_policy_value_executed": False,
            "confirmatory_holdout_claimed": False,
            "claim_boundary": CLAIM,
        })

    for task in config["authorized_tasks"]:
        train_env = task_environment(task, config["known_value_contract"]["training_regime"])
        support = global_support(train_env)
        for seed in config["seeds"]:
            data, raw = x02.logged_offline(
                train_env,
                config["known_value_contract"]["training_episodes"] + config["known_value_contract"]["validation_episodes"],
                seed,
                0.0,
            )
            policies, diagnostics = x02.train_model_free(data, train_env, support, seed)
            policies = {method: canonicalize_policy(policy, train_env) for method, policy in policies.items()}
            for diagnostic in diagnostics:
                method = str(diagnostic["method"])
                rows["model_free_results.csv"].append({
                    "task": task,
                    "method": method,
                    "seed": seed,
                    "trained": diagnostic.get("epochs_run", diagnostic.get("epochs", 0)) > 0,
                    "fidelity_label": diagnostic["fidelity"],
                    "validation_objective": diagnostic.get("validation_objective", diagnostic.get("validation_nll", math.nan)),
                    "training_role": "known_value_training",
                    "checkpoint_selection_role": "known_value_validation_only",
                    "real_ehr_policy_value": "not_computed",
                    "claim_boundary": CLAIM,
                })
            for method, policy in policies.items():
                policy_store[(task, seed, method)] = policy
                rows["support_collapse_seed_stability.csv"].append({
                    "row_type": "policy_seed_diagnostic",
                    "task": task,
                    "method": method,
                    "seed": seed,
                    **policy_diagnostic(policy, train_env, train_env.behavior),
                    "claim_boundary": CLAIM,
                })

            spec = kv.EnvSpec(
                environment_id=train_env.name,
                family="task_matched_known_value",
                episodes=len(data.states),
                horizon=train_env.horizon,
                reward_sparsity="task_proxy_dense_plus_terminal",
                support="train_frozen_mask",
                state_dim=train_env.n_states,
                missingness=float(1.0 - 0.020748576078112285 if task == "aki_rrt" else 1.0 - 0.8487328295269508),
                behavior_concentration=0.8,
                dynamics_misspecification=0.0,
                action_count=train_env.n_actions,
            )
            fits: dict[str, kv.WorldModelFit] = {}
            for method in ("grud_world_model", "transformer_world_model", "dreamer_v3_categorical_rssm"):
                try:
                    fit = k100r.fit_converged(method, data, spec, seed)
                    fits[method] = fit
                    rows["world_model_results.csv"].append({
                        "task": task,
                        "method": method,
                        "seed": seed,
                        "fidelity_label": config["fidelity_labels"][method],
                        "state_one_step_rmse": fit.validation_rmse,
                        "state_mae": fit.validation_mae,
                        "recursive_rollout_rmse": fit.rollout_rmse,
                        "reward_rmse": fit.reward_rmse,
                        "termination_auroc": fit.termination_auc,
                        "nll": fit.nll,
                        "coverage90": fit.coverage90,
                        "uncertainty_ece": fit.uncertainty_ece,
                        "parameter_count": fit.parameter_count,
                        "training_seconds": fit.training_seconds,
                        "status": fit.status,
                        "claim_boundary": CLAIM,
                    })
                except Exception as exc:
                    failures.append({"task": task, "component": "world_model", "method": method, "seed": seed,
                                     "status": "not_run_with_reason", "reason": type(exc).__name__, "detail": str(exc)[:160]})
            if "grud_world_model" in fits:
                try:
                    members = [fits["grud_world_model"]]
                    for member_seed in config["seeds"]:
                        if member_seed != seed:
                            members.append(k100r.fit_converged("grud_world_model", data, spec, member_seed))
                    ensemble = make_ensemble(members, seed)
                    fits["gaussian_recurrent_ensemble"] = ensemble
                    rows["world_model_results.csv"].append({
                        "task": task,
                        "method": "gaussian_recurrent_ensemble",
                        "seed": seed,
                        "fidelity_label": config["fidelity_labels"]["gaussian_recurrent_ensemble"],
                        "ensemble_member_count": len(members),
                        "ensemble_member_seeds": "3408|3411|3414",
                        "state_one_step_rmse": ensemble.validation_rmse,
                        "state_mae": ensemble.validation_mae,
                        "recursive_rollout_rmse": ensemble.rollout_rmse,
                        "reward_rmse": ensemble.reward_rmse,
                        "termination_auroc": ensemble.termination_auc,
                        "nll": ensemble.nll,
                        "coverage90": ensemble.coverage90,
                        "uncertainty_ece": ensemble.uncertainty_ece,
                        "parameter_count": ensemble.parameter_count,
                        "training_seconds": ensemble.training_seconds,
                        "status": ensemble.status,
                        "claim_boundary": CLAIM,
                    })
                except Exception as exc:
                    failures.append({"task": task, "component": "world_model", "method": "gaussian_recurrent_ensemble", "seed": seed,
                                     "status": "not_run_with_reason", "reason": type(exc).__name__, "detail": str(exc)[:160]})

            reward = x02.reward_table(raw, train_env)
            for method, fit in fits.items():
                next_state, uncertainty = x02.transition_tables(fit, train_env)
                for planning_horizon in config["planner_horizons"]:
                    planner = "H1_exhaustive" if planning_horizon == 1 else f"H{planning_horizon}_categorical_CEM"
                    for penalized in (False, True):
                        variant = "support_constrained" if not penalized else "support_and_uncertainty_penalized"
                        policy, audit = x02.learned_planner(next_state, reward, uncertainty, train_env, planning_horizon, penalized, seed)
                        policy = canonicalize_policy(policy, train_env)
                        name = f"{method}__{planner}__{variant}"
                        policies[name] = policy
                        policy_store[(task, seed, name)] = policy
                        predicted = x02.learned_model_value(next_state, reward, train_env, policy)
                        exact = e01.evaluate_policy_exact(train_env, policy)
                        planner_truth[(task, seed, name)] = (predicted, exact)
                        diagnostics = policy_diagnostic(policy, train_env, train_env.behavior)
                        rows["world_model_planner_matrix.csv"].append({
                            "task": task,
                            "world_model": method,
                            "planner": planner,
                            "planner_variant": variant,
                            "seed": seed,
                            "fidelity_label": "conceptual_adapter" if method == "dreamer_v3_categorical_rssm" else config["fidelity_labels"].get(method, "official_contract_adapter"),
                            "effective_horizon": min(planning_horizon, train_env.horizon),
                            "iterations": audit["iterations"],
                            "candidate_sequences": audit["candidates"],
                            "elite_count": audit["elite_count"],
                            "minimum_unique_sequences": audit["minimum_unique_sequences"],
                            "support_mask_bypass": audit["support_mask_bypass"],
                            "known_value_true_return": exact,
                            "learned_model_predicted_return": predicted,
                            "model_exploitation_gap": predicted - exact,
                            **diagnostics,
                            "claim_boundary": CLAIM,
                        })

            for regime in REGIMES:
                env = task_environment(task, regime)
                behavior_value = e01.evaluate_policy_exact(env, env.behavior)
                for method, policy in policies.items():
                    true_value = e01.evaluate_policy_exact(env, policy)
                    _, paired_delta, paired_se = x02.evaluate_policy(env, policy, seed)
                    diagnostic = policy_diagnostic(policy, env, env.behavior)
                    group = "world_model_planner" if "__H" in method else ("control" if method in CONTROL_METHODS else "model_free")
                    predicted = planner_truth.get((task, seed, method), (math.nan, math.nan))[0]
                    rows["known_value_policy_results.csv"].append({
                        "task": task,
                        "response_regime": regime,
                        "method": method,
                        "method_group": group,
                        "seed": seed,
                        "true_return": true_value,
                        "behavior_true_return": behavior_value,
                        "behavior_relative_true_value_difference": true_value - behavior_value,
                        "paired_crn_difference": paired_delta,
                        "paired_standard_error": paired_se,
                        "learned_model_predicted_return": predicted,
                        "model_exploitation_gap": predicted - true_value if np.isfinite(predicted) else math.nan,
                        **diagnostic,
                        "raw_reward_compared_across_tasks": False,
                        "claim_boundary": CLAIM,
                    })

    rows["support_collapse_seed_stability.csv"].extend(seed_stability_rows(policy_store))
    known = pd.DataFrame(rows["known_value_policy_results.csv"])
    known["rank_within_task_regime_seed"] = known.groupby(["task", "response_regime", "seed"])["true_return"].rank(ascending=False, method="average")
    rows["known_value_policy_results.csv"] = known.to_dict("records")
    common_methods = sorted(set.intersection(*(set(known.loc[known.task.eq(task), "method"]) for task in config["authorized_tasks"])))
    for method in common_methods:
        task_means = known[known.method.eq(method)].groupby("task").rank_within_task_regime_seed.mean()
        rows["cross_task_rank_stability.csv"].append({
            "method": method,
            "aki_rrt_mean_rank": float(task_means.get("aki_rrt", math.nan)),
            "heart_failure_mean_rank": float(task_means.get("heart_failure", math.nan)),
            "cross_task_mean_rank": float(task_means.mean()),
            "cross_task_rank_range": float(task_means.max() - task_means.min()),
            "known_value_rows": int(len(known[known.method.eq(method)])),
            "clinical_generalization_claimed": False,
            "claim_boundary": CLAIM,
        })

    rows["retrospective_policy_diagnostics.csv"] = retrospective_rows()
    rows["real_ehr_ope_or_nonexecution.csv"] = [
        {
            "task": "aki_rrt",
            "x08_tier2_exact_tuple_count": 236,
            "real_ehr_ope_executed": False,
            "status": "not_run_missing_real_ehr_trained_probability_complete_policy_and_policy_specific_post_training_gates",
            "approved_tuple_transferred_by_family_name_only": False,
            "real_ehr_policy_winner_available": False,
            "claim_boundary": CLAIM,
        },
        {
            "task": "heart_failure",
            "x08_tier2_exact_tuple_count": 0,
            "real_ehr_ope_executed": False,
            "status": "not_run_prohibited_zero_kdd_x08_tier2_exact_tuples",
            "approved_tuple_transferred_by_family_name_only": False,
            "real_ehr_policy_winner_available": False,
            "claim_boundary": CLAIM,
        },
    ]

    expected_model_free = len(config["authorized_tasks"]) * len(config["seeds"]) * (len(config["model_free_core"]) + len(config["supported_controls"]))
    expected_world = len(config["authorized_tasks"]) * len(config["seeds"]) * len(config["world_model_core"])
    expected_planner = expected_world * len(config["planner_horizons"]) * 2
    actual = {
        "model_free": len(rows["model_free_results.csv"]),
        "world_model": len(rows["world_model_results.csv"]),
        "world_model_planner": len(rows["world_model_planner_matrix.csv"]),
    }
    expected = {"model_free": expected_model_free, "world_model": expected_world, "world_model_planner": expected_planner}
    for component in ("model_free", "world_model", "world_model_planner"):
        component_failures = sum(x["component"] == component for x in failures)
        rows["failure_rate_summary.csv"].append({
            "component": component,
            "expected_rows": expected[component],
            "successful_rows": actual[component],
            "failure_rows": component_failures,
            "accounted_rows": actual[component] + component_failures,
            "failure_rate": component_failures / max(expected[component], 1),
            "inventory_accounted": actual[component] + component_failures == expected[component],
            "claim_boundary": CLAIM,
        })
    for failure in failures:
        rows["failure_rate_summary.csv"].append({**failure, "claim_boundary": CLAIM})

    planner_frame = pd.DataFrame(rows["world_model_planner_matrix.csv"])
    null_frame = known[known.response_regime.eq("null")]
    null_absolute = null_frame.behavior_relative_true_value_difference.abs()
    max_null_delta = float(null_absolute.max())
    max_null_row = null_frame.loc[null_absolute.idxmax(), ["task", "method", "seed", "behavior_relative_true_value_difference"]].to_dict()
    hard_checks = {
        "planner_support": not bool(planner_frame.support_mask_bypass.any()),
        "cem_iterations": bool((planner_frame.loc[planner_frame.planner.ne("H1_exhaustive"), "iterations"] == 3).all()),
        "cem_multisequence": bool((planner_frame.loc[planner_frame.planner.ne("H1_exhaustive"), "minimum_unique_sequences"] > 1).all()),
        "known_value_support": bool((known.unsupported_action_mass == 0).all()),
        "null_invariance": bool(np.isfinite(null_absolute).all() and (null_absolute <= 1e-12).all()),
    }
    hard_gate = all(hard_checks.values())
    inventories_accounted = all(row["inventory_accounted"] for row in rows["failure_rate_summary.csv"] if "inventory_accounted" in row)
    if not config["authorized_tasks"]:
        decision = "no_new_task_passed_promotion"
    elif hard_gate and inventories_accounted:
        decision = "complete_no_real_ehr_policy_value"
    else:
        (output / "failure_receipt.md").write_text(
            "# KDD-X09 failed execution receipt\n\n"
            f"Hard checks: `{json.dumps(hard_checks, sort_keys=True)}`.\n\n"
            f"Expected inventory: `{json.dumps(expected, sort_keys=True)}`.\n\n"
            f"Actual inventory: `{json.dumps(actual, sort_keys=True)}`.\n\n"
            f"Maximum null delta: `{max_null_delta}` at `{json.dumps(max_null_row, sort_keys=True)}`.\n\n"
            f"Failure rows: `{len(failures)}`. No retrospective EHR OPE value was computed.\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            f"core policy benchmark hard gate or inventory accounting failed: hard={hard_checks} "
            f"expected={expected} actual={actual} failures={len(failures)} max_null={max_null_delta} row={max_null_row}"
        )

    # "null" is a pandas default NA token. Preserve the scientific regime while
    # making the aggregate CSV unambiguous for default downstream parsers.
    known["response_regime"] = known.response_regime.map(output_regime_label)
    rows["known_value_policy_results.csv"] = known.to_dict("records")
    for name in REQUIRED[:10]:
        write_csv(output / name, rows[name])
    (output / "decision.md").write_text(
        f"# KDD-X09 decision\n\n`{decision}`\n\n"
        "Eligible promoted tasks: AKI-RRT and heart failure. The frozen baseline inventory is fully accounted. "
        f"Planner/support/null hard gate: {hard_gate}. Retrospective real-EHR OPE rows executed: 0.\n\n{CLAIM}\n",
        encoding="utf-8",
    )
    best = known[known.response_regime.eq("moderate")].groupby(["task", "method"], as_index=False).true_return.mean()
    leaders = best.loc[best.groupby("task").true_return.idxmax(), ["task", "method"]]
    leader_text = "; ".join(f"{row.task}={row.method}" for row in leaders.itertuples())
    (output / "summary.md").write_text(
        "# KDD-X09 promoted-cohort policy benchmark\n\n"
        f"Decision: `{decision}`. The run produced {len(known)} known-value policy rows, "
        f"{len(rows['world_model_planner_matrix.csv'])} world-model/planner rows, and "
        f"{len(rows['model_free_results.csv'])} model-free training receipts.\n\n"
        f"Moderate-regime task-specific point leaders were {leader_text}. These are injected-mechanism results, "
        "not cross-disease reward comparisons or clinical recommendations.\n\n"
        "AKI had KDD-X08 exact tuple candidates but no real-EHR-trained target-policy probability surface or "
        "policy-specific post-training overlap/ESS receipt in this run. HF had zero approved Tier-2 tuples. "
        "Accordingly no retrospective EHR OPE value or EHR policy winner is reported.\n",
        encoding="utf-8",
    )
    missing = [name for name in REQUIRED if not (output / name).is_file() or not (output / name).stat().st_size]
    if missing:
        raise RuntimeError(f"missing outputs: {missing}")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / f"kdd_benchmark_discovery/results/kdd_x09_promoted_cohort_policy_benchmark_{time.strftime('%Y%m%d_%H%M%S')}"
    started = time.perf_counter()
    decision = run(output, args.config)
    display = output.resolve().relative_to(ROOT)
    print(f"decision={decision} output={display} elapsed_seconds={time.perf_counter() - started:.3f}")


if __name__ == "__main__":
    main()
