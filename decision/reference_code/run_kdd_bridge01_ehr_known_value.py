from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
import torch
from torch.utils.data import DataLoader, TensorDataset

from kdd_benchmark_discovery import kdd098_training as k98
from kdd_benchmark_discovery import kdd_e01_evaluator as e01
from kdd_benchmark_discovery import run_kdd100_complete_known_value as kv
from kdd_benchmark_discovery import run_kdd_adapt01_adaptive_known_value as adapt
from kdd_benchmark_discovery import run_kdd_x02_cross_cohort_policy_benchmark as x02


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/kdd_bridge01_ehr_known_value_v1.json"
EHR_ROOT = ROOT / "results/kdd098r_convergence_planning_components_20260714_v2"
KV_ROOT = ROOT / "kdd_benchmark_discovery/results/kdd_adapt01_adaptive_known_value_20260715_092634"
REQUIRED = (
    "ehr_to_known_value_contract_matrix.csv",
    "cross_surface_model_family_rows.csv",
    "prediction_policy_consistency.csv",
    "uncertainty_planning_consistency.csv",
    "action_information_policy_discrimination.csv",
    "cross_surface_rank_stability.csv",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_paths() -> dict[str, Path]:
    return {
        "ehr_recursive_rollout": EHR_ROOT / "recursive_rollout_metrics.csv",
        "ehr_uncertainty": EHR_ROOT / "uncertainty_calibration.csv",
        "ehr_action_information": EHR_ROOT / "targeted_action_information.csv",
        "ehr_world_model_registry": EHR_ROOT / "world_model_registry.csv",
        "adaptive_environment_contract": KV_ROOT / "adaptive_environment_contract.csv",
        "adapt01_config": ROOT / "configs/kdd_adapt01_adaptive_known_value_v1.json",
        "adapt01_runner": ROOT / "kdd_benchmark_discovery/run_kdd_adapt01_adaptive_known_value.py",
        "kdd098r_config": ROOT / "configs/kdd098r_world_model.json",
        "kdd098_model_factory": ROOT / "kdd_benchmark_discovery/kdd098_training.py",
        "kdd098r_selection_implementation": ROOT / "kdd_benchmark_discovery/kdd098r_training.py",
    }


def verify_sources(config: dict[str, Any]) -> None:
    actual = {name: _sha256(path) for name, path in source_paths().items()}
    if actual != config["immutable_source_hashes"]:
        raise RuntimeError(f"immutable source drift: {actual}")


def _correlation_rows(
    frame: pd.DataFrame,
    x: str,
    y: str,
    relationship: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for scope, subset in (
        ("all_four_task_profiles", frame),
        ("aligned_respiratory_and_shock_only", frame[frame["primary_bridge_eligible"]]),
    ):
        clean = subset[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) >= 3 and clean[x].nunique() > 1 and clean[y].nunique() > 1:
            pearson = float(np.corrcoef(clean[x], clean[y])[0, 1])
            spearman = float(spearmanr(clean[x], clean[y]).statistic)
        else:
            pearson = np.nan
            spearman = np.nan
        rows.append(
            {
                "relationship": relationship,
                "analysis_scope": scope,
                "sample_unit": "task_model_family",
                "n": len(clean),
                "x_metric": x,
                "y_metric": y,
                "pearson_r": pearson,
                "spearman_rho": spearman,
                "analysis_label": config["analysis_label"],
                "confirmatory_inference": False,
                "claim_boundary": config["claim_boundary"],
            }
        )
    return rows


def contract_matrix(config: dict[str, Any], environment: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    env_by_task = environment.set_index("task")
    properties = (
        ("action_count", "ehr_aggregate", "Frozen retrospective action contract cardinality"),
        ("supported_action_count", "ehr_aggregate", "Train-frozen supported action classes"),
        ("decision_horizon", "ehr_aggregate_contract", "Nominal repeated-decision horizon"),
        ("missingness_rate", "ehr_aggregate", "Aggregate observation missingness rate only"),
        ("termination_prevalence", "ehr_aggregate", "Aggregate terminal prevalence only"),
        ("behavior_top_action_share", "ehr_aggregate", "Aggregate action-count concentration"),
        ("reward_scale", "synthetic_normalization", "Unit synthetic reward-scale convention"),
        ("reward_sparsity", "synthetic_contract_specification", "Mixed dense-terminal synthetic reward design"),
        ("state_count", "synthetic_finite_design", "Finite severity/subtype/momentum/observation layout"),
        ("response_components", "synthetic_known_mechanism", "Action-response and toxicity mechanisms known by construction"),
    )
    synthetic_extensions = (
        ("state_transition_kernel", "known finite transition mechanism", "synthetic_known_mechanism"),
        ("state_correlation_structure", "finite severity/subtype/momentum layout", "synthetic_finite_design"),
        ("latent_response_subtype", "two synthetic subtypes with partial observation", "synthetic_known_mechanism"),
        ("missingness_process", "Bernoulli process calibrated to EHR aggregate rate", "hybrid_ehr_rate_synthetic_process"),
        ("termination_process", "constant synthetic hazard calibrated to EHR aggregate prevalence", "hybrid_ehr_rate_synthetic_process"),
        ("behavior_state_dependence", "synthetic severity tilt over EHR aggregate action counts", "hybrid_ehr_counts_synthetic_process"),
        ("support_mask_state_dependence", "EHR-supported actions broadcast over synthetic states", "hybrid_ehr_support_synthetic_process"),
        ("initial_state_distribution", "synthetic severity/subtype/missingness distribution", "synthetic_finite_design"),
        ("world_model_weights", "retrained separately per environment and seed", "independent_synthetic_retraining"),
        ("exact_policy_truth", "backward induction in known finite mechanism", "synthetic_exact_truth"),
    )
    reverse_task = {value["known_value_task"]: key for key, value in config["task_map"].items()}
    for known_task, row in env_by_task.iterrows():
        ehr_task = reverse_task[known_task]
        mapping = config["task_map"][ehr_task]
        for name, provenance, meaning in properties:
            rows.append(
                {
                    "ehr_task": ehr_task,
                    "known_value_task": known_task,
                    "property": name,
                    "value": row[name],
                    "provenance_class": provenance,
                    "interpretation": meaning,
                    "contract_alignment": mapping["contract_alignment"],
                    "primary_bridge_eligible": mapping["primary_bridge_eligible"],
                    "source_artifact": "KDD-ADAPT01 adaptive_environment_contract.csv",
                    "claim_boundary": config["claim_boundary"],
                }
            )
        for name, value, provenance in synthetic_extensions:
            rows.append(
                {
                    "ehr_task": ehr_task,
                    "known_value_task": known_task,
                    "property": name,
                    "value": value,
                    "provenance_class": provenance,
                    "interpretation": "Known-value construction detail; not estimated as an EHR action response",
                    "contract_alignment": mapping["contract_alignment"],
                    "primary_bridge_eligible": mapping["primary_bridge_eligible"],
                    "source_artifact": "KDD-ADAPT01 config and runner",
                    "claim_boundary": config["claim_boundary"],
                }
            )
    return pd.DataFrame(rows)


def _model_fingerprint(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def fit_known_value_world_model(
    method: str,
    train_data: kv.OfflineData,
    validation_data: kv.OfflineData,
    state_dim: int,
    action_dim: int,
    seed: int,
    budget: dict[str, Any],
    epoch_cap: int,
) -> tuple[kv.WorldModelFit, dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    model = k98._make_model(method, state_dim, action_dim, budget)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(budget["learning_rate"]),
        weight_decay=float(budget["weight_decay"]),
    )
    train_action = np.eye(action_dim, dtype=np.float32)[train_data.actions]
    val_action = np.eye(action_dim, dtype=np.float32)[validation_data.actions]
    dataset = TensorDataset(
        torch.from_numpy(train_data.observed),
        torch.from_numpy(train_data.masks.astype(np.float32)),
        torch.from_numpy(train_data.deltas),
        torch.from_numpy(train_action),
        torch.from_numpy(train_data.next_states),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(budget["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    best_state = None
    best_score = math.inf
    best_epoch = 0
    stale = 0
    composites: list[float] = []
    started = time.perf_counter()
    for epoch in range(1, epoch_cap + 1):
        model.train()
        for values, masks, deltas, actions, target in loader:
            prediction = model(values, masks, deltas, actions)
            scale = torch.exp(prediction.log_scale)
            loss = (prediction.log_scale + 0.5 * torch.square((target - prediction.mean) / scale)).mean()
            loss = loss + 0.01 * prediction.auxiliary_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(budget["gradient_clip_norm"]))
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            prediction = model(
                torch.from_numpy(validation_data.observed),
                torch.from_numpy(validation_data.masks.astype(np.float32)),
                torch.from_numpy(validation_data.deltas),
                torch.from_numpy(val_action),
            )
            error = prediction.mean.numpy() - validation_data.next_states
            rmse = float(np.sqrt(np.mean(np.square(error))))
            mae = float(np.mean(np.abs(error)))
            composite = 0.7 * rmse + 0.3 * mae
        composites.append(composite)
        relative = (best_score - composite) / max(abs(best_score), 1e-12) if np.isfinite(best_score) else np.inf
        if relative >= float(budget["minimum_relative_validation_improvement"]):
            best_score = composite
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        elif epoch >= int(budget["min_epochs"]):
            stale += 1
        if epoch >= int(budget["min_epochs"]) and stale >= int(budget["patience"]):
            break
    if best_state is None:
        raise RuntimeError(f"no finite bridge checkpoint: {method}/{seed}")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        prediction = model(
            torch.from_numpy(validation_data.observed),
            torch.from_numpy(validation_data.masks.astype(np.float32)),
            torch.from_numpy(validation_data.deltas),
            torch.from_numpy(val_action),
        )
        mean = prediction.mean.numpy()
        scale = np.exp(prediction.log_scale.numpy())
    error = mean - validation_data.next_states
    validation_rmse = float(np.sqrt(np.mean(np.square(error))))
    validation_mae = float(np.mean(np.abs(error)))
    nll = float(np.mean(np.log(np.maximum(scale, 1e-8)) + 0.5 * np.square(error / np.maximum(scale, 1e-8))))
    coverage90 = float(np.mean(np.abs(error) <= 1.6448536269514722 * scale))
    trained_epochs = len(composites)
    cap_hit = trained_epochs == epoch_cap
    improving_at_cap = bool(
        cap_hit
        and trained_epochs > 1
        and (composites[-2] - composites[-1]) / max(abs(composites[-2]), 1e-12)
        >= float(budget["minimum_relative_validation_improvement"])
    )
    fit = kv.WorldModelFit(
        method,
        seed,
        model,
        validation_rmse,
        validation_mae,
        nll,
        coverage90,
        np.nan,
        np.nan,
        np.nan,
        abs(coverage90 - 0.9),
        int(sum(parameter.numel() for parameter in model.parameters())),
        time.perf_counter() - started,
        0.0,
        "cap_hit_still_improving" if improving_at_cap else ("cap_hit_plateau" if cap_hit else "early_stopped"),
        _model_fingerprint(model),
    )
    receipt = {
        "method": method,
        "seed": seed,
        "selected_epoch": best_epoch,
        "trained_epochs": trained_epochs,
        "epoch_cap": epoch_cap,
        "cap_hit": cap_hit,
        "improving_at_cap": improving_at_cap,
        "validation_composite": best_score,
        "checkpoint_fingerprint": fit.fingerprint,
        "ehr_weights_reused": False,
    }
    return fit, receipt


def retrain_known_value_planners(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    adapt_config = json.loads((ROOT / "configs/kdd_adapt01_adaptive_known_value_v1.json").read_text(encoding="utf-8"))
    budget = config["known_value_training"]
    seeds = [int(seed) for seed in config["known_value_training_seeds"]]
    methods = ("grud_world_model", "transformer_world_model", "dreamer_v3_categorical_rssm")
    planner_rows: list[dict[str, Any]] = []
    receipt_rows: list[dict[str, Any]] = []
    for task_index, (task, profile) in enumerate(adapt_config["tasks"].items()):
        env, _, features = adapt.build_environment(task, profile, "adaptive_composite", adapt_config["mechanisms"])
        oracle_value, _, _ = e01.backward_induction(env)
        data_by_seed: dict[int, tuple[kv.OfflineData, kv.OfflineData, dict[str, np.ndarray]]] = {}
        for seed in seeds:
            train_data, train_raw = adapt.logged_offline(
                env,
                features,
                int(budget["training_episodes"]),
                9_501_000 + task_index * 10_000 + seed,
                float(profile["missingness"]),
            )
            validation_data, _ = adapt.logged_offline(
                env,
                features,
                int(budget["validation_episodes"]),
                9_601_000 + task_index * 10_000 + seed,
                float(profile["missingness"]),
            )
            data_by_seed[seed] = (train_data, validation_data, train_raw)
        fits_by_seed: dict[int, dict[str, kv.WorldModelFit]] = {seed: {} for seed in seeds}
        receipts_by_method: dict[str, list[dict[str, Any]]] = {}
        for method in methods:
            initial = []
            for seed in seeds:
                train_data, validation_data, _ = data_by_seed[seed]
                fit, receipt = fit_known_value_world_model(
                    method, train_data, validation_data, env.n_states, env.n_actions, seed, budget, int(budget["max_epochs"])
                )
                initial.append((fit, receipt))
            extend = sum(receipt["cap_hit"] and receipt["improving_at_cap"] for _, receipt in initial) > len(initial) / 2
            final = initial
            if extend:
                final = []
                for seed in seeds:
                    train_data, validation_data, _ = data_by_seed[seed]
                    final.append(
                        fit_known_value_world_model(
                            method,
                            train_data,
                            validation_data,
                            env.n_states,
                            env.n_actions,
                            seed,
                            budget,
                            int(budget["single_uniform_extension_epochs"]),
                        )
                    )
            receipts_by_method[method] = []
            for fit, receipt in final:
                receipt.update(task=task, uniform_extension_applied=extend, architecture_capacity="hidden48_latent16")
                receipt_rows.append(receipt)
                receipts_by_method[method].append(receipt)
                fits_by_seed[int(fit.seed)][method] = fit
        ensemble = adapt.make_ensemble([fits_by_seed[seed]["grud_world_model"] for seed in seeds], seeds[0])
        ensemble = kv.WorldModelFit(
            "gaussian_recurrent_ensemble",
            config["ehr_derived_ensemble_seed_label"],
            ensemble.model,
            ensemble.validation_rmse,
            ensemble.validation_mae,
            ensemble.nll,
            ensemble.coverage90,
            ensemble.rollout_rmse,
            ensemble.reward_rmse,
            ensemble.termination_auc,
            ensemble.uncertainty_ece,
            ensemble.parameter_count,
            ensemble.training_seconds,
            ensemble.peak_memory_mb,
            "derived_from_three_independently_retrained_grud_members",
            ensemble.fingerprint,
        )
        evaluation_fits = [
            (task, method, seed, fits_by_seed[seed][method], data_by_seed[seed][2])
            for seed in seeds
            for method in methods
        ]
        evaluation_fits.append((task, "gaussian_recurrent_ensemble", config["ehr_derived_ensemble_seed_label"], ensemble, data_by_seed[seeds[0]][2]))
        for _, method, seed_label, fit, raw in evaluation_fits:
            next_state, uncertainty = adapt.transition_tables(fit, env, features)
            reward_table = x02.reward_table(raw, env)
            for horizon in (1, 4, 8):
                for penalized in (False, True):
                    policy, audit = x02.learned_planner(next_state, reward_table, uncertainty, env, horizon, penalized, seeds[0] if isinstance(seed_label, str) else int(seed_label))
                    policy = adapt.table_policy(policy, env)
                    true_value = e01.evaluate_policy_exact(env, policy)
                    predicted_value = x02.learned_model_value(next_state, reward_table, env, policy)
                    planner_rows.append(
                        {
                            "task": task,
                            "world_model": method,
                            "seed": seed_label,
                            "planner": "H1_exhaustive" if horizon == 1 else f"H{horizon}_categorical_CEM",
                            "planner_variant": "support_and_uncertainty_penalized" if penalized else "support_constrained",
                            "effective_horizon": min(horizon, env.horizon),
                            "exact_true_value": true_value,
                            "exact_regret": oracle_value - true_value,
                            "predicted_model_value": predicted_value,
                            "absolute_exploitation_gap": abs(predicted_value - true_value),
                            "support_mask_bypass": audit["support_mask_bypass"],
                            "ehr_weights_reused": False,
                        }
                    )
    planners = pd.DataFrame(planner_rows)
    if planners["support_mask_bypass"].any():
        raise RuntimeError("support mask bypass in bridge retraining")
    return planners, pd.DataFrame(receipt_rows)


def build_bridge(config: dict[str, Any]) -> tuple[pd.DataFrame, ...]:
    recursive = pd.read_csv(source_paths()["ehr_recursive_rollout"])
    uncertainty = pd.read_csv(source_paths()["ehr_uncertainty"])
    action = pd.read_csv(source_paths()["ehr_action_information"])
    registry = pd.read_csv(source_paths()["ehr_world_model_registry"])
    environment = pd.read_csv(source_paths()["adaptive_environment_contract"])
    planners, training_receipts = retrain_known_value_planners(config)

    seeds = set(config["shared_training_seeds"])
    seed_labels = {str(seed) for seed in seeds}
    ensemble_seed_label = config["ehr_derived_ensemble_seed_label"]
    ehr_methods = set(config["shared_model_family_map"])
    ehr_tasks = set(config["task_map"])
    registry_seed_eligible = registry["seed"].astype(str).isin(seed_labels | {ensemble_seed_label})
    available = registry[
        registry["task"].isin(ehr_tasks)
        & registry["method_id"].isin(ehr_methods)
        & registry_seed_eligible
        & registry["result_available"]
    ]
    expected_registry = len(ehr_tasks) * ((len(ehr_methods) - 1) * len(seeds) + 1)
    if len(available) != expected_registry:
        raise RuntimeError(f"shared EHR family/seed inventory incomplete: {len(available)} != {expected_registry}")

    recursive = recursive[
        recursive["task"].isin(ehr_tasks)
        & recursive["method_id"].isin(ehr_methods)
        & recursive["seed"].astype(str).isin(seed_labels | {ensemble_seed_label})
        & recursive["logged_action_recursive_rollout"]
    ]
    recursive_summary = recursive.groupby(["task", "method_id"], as_index=False).agg(
        ehr_recursive_rmse_mean=("rmse", "mean"),
        ehr_recursive_rmse_sd=("rmse", "std"),
        ehr_recursive_horizon_rows=("rmse", "size"),
        ehr_recursive_max_horizon_hours=("horizon_hours", "max"),
    )
    uncertainty = uncertainty[
        uncertainty["task"].isin(ehr_tasks)
        & uncertainty["method_id"].isin(ehr_methods)
        & uncertainty["seed"].astype(str).isin(seed_labels | {ensemble_seed_label})
        & uncertainty["evaluation_mode"].eq("recursive")
        & uncertainty["status"].eq("available_gaussian_scoring")
    ]
    uncertainty_summary = uncertainty.groupby(["task", "method_id"], as_index=False).agg(
        ehr_recursive_calibration_error=("mean_absolute_coverage_error", "mean"),
        ehr_recursive_nll=("gaussian_nll", "mean"),
        ehr_uncertainty_seed_rows=("seed", "size"),
    )
    action = action[
        action["task"].isin(ehr_tasks)
        & action["method_id"].isin(ehr_methods)
        & action["seed"].astype(str).isin(seed_labels | {ensemble_seed_label})
        & action["diagnostic_stratum"].eq("reward_relevant_features")
        & action["comparator"].eq("matched_shuffle")
        & action["semantic_valid"]
    ]
    action_summary = action.groupby(["task", "method_id"], as_index=False).agg(
        ehr_factual_action_information_delta=("comparator_minus_observed_delta_rmse", "mean"),
        ehr_action_cluster_delta=("episode_cluster_mean_delta", "mean"),
        ehr_action_information_seed_rows=("seed", "size"),
    )

    inverse_method = {known: ehr for ehr, known in config["shared_model_family_map"].items()}
    known_rows = []
    for (task, family), group in planners.groupby(["task", "world_model"]):
        if family not in inverse_method:
            continue
        stability = group.groupby(["planner", "planner_variant", "effective_horizon"])["exact_true_value"].std(ddof=1)
        discrimination = group.groupby("seed")["exact_true_value"].agg(lambda values: float(values.max() - values.min()))
        receipt_family = "grud_world_model" if family == "gaussian_recurrent_ensemble" else family
        receipts = training_receipts[
            training_receipts["task"].eq(task) & training_receipts["method"].eq(receipt_family)
        ]
        known_rows.append(
            {
                "known_value_task": task,
                "known_value_model_family": family,
                "known_value_policy_regret_mean": float(group["exact_regret"].mean()),
                "known_value_policy_regret_sd": float(group["exact_regret"].std(ddof=1)),
                "known_value_absolute_exploitation_gap_mean": float((group["predicted_model_value"] - group["exact_true_value"]).abs().mean()),
                "known_value_signed_exploitation_gap_mean": float((group["predicted_model_value"] - group["exact_true_value"]).mean()),
                "known_value_planner_seed_sd_mean": float(stability.mean()),
                "known_value_policy_discrimination_range_mean": float(discrimination.mean()),
                "known_value_planner_rows": len(group),
                "known_value_training_seed_count": int(group["seed"].nunique()),
                "known_value_member_training_seed_count": int(receipts["seed"].nunique()),
                "known_value_selected_epoch_mean": float(receipts["selected_epoch"].mean()),
                "known_value_trained_epoch_mean": float(receipts["trained_epochs"].mean()),
                "known_value_uniform_extension_applied": bool(receipts["uniform_extension_applied"].any()),
                "known_value_architecture_capacity": "hidden48_latent16",
                "known_value_selection_protocol": "KDD098R_40_to_uniform80_validation_composite",
                "ehr_weights_reused": False,
            }
        )
    known = pd.DataFrame(known_rows)

    bridge = recursive_summary.merge(uncertainty_summary, on=["task", "method_id"], how="left")
    bridge = bridge.merge(action_summary, on=["task", "method_id"], how="left")
    bridge["known_value_task"] = bridge["task"].map(
        {key: value["known_value_task"] for key, value in config["task_map"].items()}
    )
    bridge["known_value_model_family"] = bridge["method_id"].map(config["shared_model_family_map"])
    bridge = bridge.merge(known, on=["known_value_task", "known_value_model_family"], how="left", validate="one_to_one")
    bridge["model_family_mapping"] = bridge["method_id"] + " -> " + bridge["known_value_model_family"]
    bridge["ehr_seed_surface"] = np.where(
        bridge["method_id"].eq("gaussian_transition_ensemble"),
        "one_derived_three_member_surface_not_independent_seed_repetitions",
        "three_independent_shared_training_seeds",
    )
    bridge["contract_alignment"] = bridge["task"].map(
        {key: value["contract_alignment"] for key, value in config["task_map"].items()}
    )
    bridge["primary_bridge_eligible"] = bridge["task"].map(
        {key: value["primary_bridge_eligible"] for key, value in config["task_map"].items()}
    )
    bridge["analysis_label"] = config["analysis_label"]
    bridge["claim_boundary"] = config["claim_boundary"]
    expected_bridge = len(ehr_tasks) * len(ehr_methods)
    if len(bridge) != expected_bridge or bridge["known_value_policy_regret_mean"].isna().any():
        raise RuntimeError("bridge inventory or known-value join incomplete")

    prediction = pd.DataFrame(
        _correlation_rows(
            bridge,
            "ehr_recursive_rmse_mean",
            "known_value_policy_regret_mean",
            "ehr_recursive_error_vs_known_value_policy_regret",
            config,
        )
        + _correlation_rows(
            bridge,
            "ehr_recursive_rmse_mean",
            "known_value_absolute_exploitation_gap_mean",
            "ehr_recursive_error_vs_known_value_model_exploitation",
            config,
        )
    )
    uncertainty_consistency = pd.DataFrame(
        _correlation_rows(
            bridge,
            "ehr_recursive_calibration_error",
            "known_value_planner_seed_sd_mean",
            "ehr_uncertainty_calibration_vs_known_value_planner_stability",
            config,
        )
    )
    action_consistency = pd.DataFrame(
        _correlation_rows(
            bridge,
            "ehr_factual_action_information_delta",
            "known_value_policy_discrimination_range_mean",
            "ehr_factual_action_information_vs_known_value_policy_discrimination",
            config,
        )
    )

    rank_rows = []
    for task, group in bridge.groupby("task"):
        group = group.copy()
        group["ehr_recursive_error_rank_lower_is_better"] = rankdata(group["ehr_recursive_rmse_mean"], method="average")
        group["known_value_regret_rank_lower_is_better"] = rankdata(group["known_value_policy_regret_mean"], method="average")
        rho = float(spearmanr(group["ehr_recursive_error_rank_lower_is_better"], group["known_value_regret_rank_lower_is_better"]).statistic)
        for _, row in group.iterrows():
            rank_rows.append(
                {
                    "task": task,
                    "model_family": row["known_value_model_family"],
                    "ehr_recursive_error_rank_lower_is_better": row["ehr_recursive_error_rank_lower_is_better"],
                    "known_value_regret_rank_lower_is_better": row["known_value_regret_rank_lower_is_better"],
                    "within_task_spearman_rank_concordance": rho,
                    "contract_alignment": row["contract_alignment"],
                    "primary_bridge_eligible": row["primary_bridge_eligible"],
                    "analysis_label": config["analysis_label"],
                    "claim_boundary": config["claim_boundary"],
                }
            )
    ranks = pd.DataFrame(rank_rows)
    contracts = contract_matrix(config, environment)
    return contracts, bridge, prediction, uncertainty_consistency, action_consistency, ranks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    verify_sources(config)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    frames = build_bridge(config)
    for name, frame in zip(REQUIRED, frames):
        frame.to_csv(args.output / name, index=False, lineterminator="\n", float_format="%.10g")
    bridge = frames[1]
    prediction = frames[2]
    ranks = frames[5]
    summary = [
        "# KDD-BRIDGE01 EHR/known-value diagnostic bridge",
        "",
        "Decision: `exploratory_diagnostic_consistency_bridge_complete`",
        "",
        f"- Shared task-model-family rows: {len(bridge)}",
        f"- Aligned respiratory/shock rows: {int(bridge['primary_bridge_eligible'].sum())}",
        f"- Named AKI/HF contract-mismatch rows: {int((~bridge['primary_bridge_eligible']).sum())}",
        "- Known-value architectures were retrained independently by KDD-BRIDGE01 with the KDD098R capacity and validation-selection protocol; no EHR-trained weights were reused.",
        "- Associations are descriptive and exploratory. Small task-family samples, shared task structure, and contract mismatches preclude causal or independent-generalization inference.",
        "- No raw EHR row, identifier, timestamp, trajectory, tensor, checkpoint, or retrospective policy value was accessed or exported.",
        "",
        "## Descriptive coefficients",
        "",
    ]
    for _, row in prediction.iterrows():
        summary.append(
            f"- {row.relationship} ({row.analysis_scope}, n={row.n}): Pearson {row.pearson_r:.3f}; Spearman {row.spearman_rho:.3f}."
        )
    aligned_rank = ranks[ranks["primary_bridge_eligible"]].groupby("task")["within_task_spearman_rank_concordance"].first()
    for task, value in aligned_rank.items():
        summary.append(f"- {task} within-task family-rank concordance: Spearman {value:.3f}.")
    (args.output / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    hashes = {
        path.name: _sha256(path)
        for path in sorted(args.output.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    (args.output / "artifact_hashes.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
