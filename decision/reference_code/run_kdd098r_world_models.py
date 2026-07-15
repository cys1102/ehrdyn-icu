from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

from .kdd098_data import TaskSequences, build_kdd098_task_sequences
from .kdd098_metrics import evaluate_surface
from .kdd098_training import Surface, fit_hgb_surface, gaussian_ensemble_surface, persistence_surface
from .kdd098r_metrics import reward_rows, targeted_action_rows, termination_rows
from .kdd098r_training import RFitReceipt, RSurface, fit_world_model
from .run_kdd098_world_models import CATALOG_EXECUTION, KDD095_CATALOG, NOT_RUN_REASONS, _validate_task_sequences


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/kdd098r_world_model.json"
DEFAULT_OUTPUT = ROOT / "results/kdd098r_convergence_planning_components_20260714_v1"
DEFAULT_CHECKPOINT = ROOT / "checkpoints/kdd098r_convergence_planning_components_20260714_v1"
MIMIC = Path("<authorized-mimiciv-root>")
V2_RESULT = ROOT / "results/kdd098_world_model_planning_readiness_20260714_v2"
V2_CHECKPOINT = ROOT / "checkpoints/kdd098_world_model_planning_readiness_20260714_v2"
KDD099RA = ROOT / "kdd_benchmark_discovery/results/kdd099r_a_reward_provenance_20260714_v2"
METHODS = ("grud_world_model", "transformer_world_model", "dreamer_v3_categorical_rssm")
CLAIM = (
    "Development-only factual logged-action planning-component validation. Action-information deltas are predictive "
    "diagnostics, not causal treatment effects. No confirmatory, policy-winner, treatment-benefit, causal, "
    "counterfactual, clinical-utility, deployment, or autonomous-decision claim."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KDD098R development-only convergence and planning-component validation")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--mimiciv-root", type=Path, default=MIMIC)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.output.exists() or args.checkpoint_root.exists():
        raise FileExistsError("KDD098R output or checkpoint directory already exists")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    _preflight(config)
    v2_result_hash = tree_hash(V2_RESULT); v2_checkpoint_hash = tree_hash(V2_CHECKPOINT)
    tasks = build_kdd098_task_sequences(args.mimiciv_root, args.chunksize)
    _validate_task_sequences(tasks)
    tables = _empty_tables(config)
    failures: list[dict[str, Any]] = []
    action_classification: dict[str, str] = {}
    for task in tasks:
        print(f"KDD098R: {task.task} controls", flush=True)
        control_surfaces = [persistence_surface(task)]
        for seed in config["seeds"]:
            try:
                control_surfaces.append(fit_hgb_surface(task, int(seed), config["training_budget"]))
            except Exception as error:
                failures.append(_failure(task.task, "hgb_residual", seed, "control_training", error))
        for surface in control_surfaces:
            _collect_base(task, surface, config, tables, control=True)

        grud_members: list[RSurface] = []
        for method in METHODS:
            print(f"KDD098R: {task.task} {method} cap=40", flush=True)
            initial: list[tuple[RSurface, RFitReceipt]] = []
            for seed in config["seeds"]:
                try:
                    initial.append(fit_world_model(task, method, int(seed), config["training_budget"], 40, args.checkpoint_root))
                except Exception as error:
                    failures.append(_failure(task.task, method, seed, "initial_training", error))
            extension = len(initial) == 3 and sum(receipt.cap_hit and receipt.improving_at_cap for _, receipt in initial) > len(initial) / 2
            final = initial
            if extension:
                print(f"KDD098R: {task.task} {method} uniform extension cap=80", flush=True)
                final = []
                for seed in config["seeds"]:
                    try:
                        final.append(fit_world_model(task, method, int(seed), config["training_budget"], 80, args.checkpoint_root))
                    except Exception as error:
                        failures.append(_failure(task.task, method, seed, "uniform_extension", error))
                if len(final) != 3:
                    failures.append({"experiment_id": "KDD098R", "task": task.task, "method_id": method, "seed": "all", "receipt_type": "extension_group_invalid", "reason": "uniform extension did not complete for every seed"})
            for result, receipt in final:
                _collect_result(task, result, receipt, config, tables, extension)
                if method == "grud_world_model": grud_members.append(result)
        if len(grud_members) == 3:
            ensemble = _ensemble_result(task, grud_members, args.checkpoint_root)
            _collect_result(task, ensemble, None, config, tables, False)
        else:
            failures.append({"experiment_id": "KDD098R", "task": task.task, "method_id": "gaussian_transition_ensemble", "seed": "3408;3411;3414", "receipt_type": "not_run_with_reason", "reason": "not all recurrent members available"})
        action_classification[task.task] = _classify_action(task, _action_classification_rows(tables, task.task), config)

    tables["method_coverage_and_failure_receipts.csv"] = _coverage_rows(tables, failures)
    tables["world_model_registry.csv"] = _registry_rows(tables)
    tables["privacy_audit.csv"] = _privacy_rows()
    tables["decision.csv"] = _decision_rows(tables, tasks, action_classification)
    args.output.mkdir(parents=True)
    for name, rows in tables.items(): _write_csv(args.output / name, rows)
    (args.output / "kdd098r_report.md").write_text(_report(tables, action_classification, v2_result_hash, v2_checkpoint_hash), encoding="utf-8")
    _privacy_scan(args.output)
    _write_hashes(args.output)
    if tree_hash(V2_RESULT) != v2_result_hash or tree_hash(V2_CHECKPOINT) != v2_checkpoint_hash:
        raise RuntimeError("KDD098-v2 result or checkpoint lineage changed")
    print(f"KDD098R decision: {tables['decision.csv'][-1]['decision']}", flush=True)
    print(f"KDD098R aggregate output: {args.output}", flush=True)


def _empty_tables(config):
    names = [
        "world_model_registry.csv", "convergence_contract.csv", "learning_curve_summary.csv", "best_epoch_and_cap_hits.csv",
        "one_step_state_metrics.csv", "recursive_rollout_metrics.csv", "reward_prediction_or_propagation.csv",
        "termination_native_and_recalibrated.csv", "rollout_survival_sanity.csv", "uncertainty_calibration.csv",
        "risk_coverage_metrics.csv", "targeted_action_information.csv", "reward_action_information.csv",
        "action_event_window_diagnostics.csv", "delayed_action_horizon_diagnostics.csv",
        "support_severity_propensity_diagnostics.csv", "component_availability.csv", "checkpoint_hash_manifest.csv",
        "planning_component_readiness.csv", "method_coverage_and_failure_receipts.csv", "resource_metrics.csv",
        "privacy_audit.csv", "decision.csv",
    ]
    tables = {name: [] for name in names}
    budget = config["training_budget"]
    for method in METHODS:
        tables["convergence_contract.csv"].append({
            "experiment_id": "KDD098R", "method_id": method, "seeds": "3408;3411;3414", "min_epochs": budget["min_epochs"],
            "max_epochs": budget["max_epochs"], "patience": budget["patience"],
            "minimum_relative_validation_improvement": budget["minimum_relative_validation_improvement"],
            "validation_composite": config["selection_contract"]["validation_composite"],
            "extension_cap": budget["single_uniform_extension_epochs"], "extension_scope": "all_three_seeds_within_task_family",
            "selective_seed_extension_permitted": False, "selection_role": "validation_only", "fit_role": "train_only", "claim_boundary": CLAIM,
        })
    return tables


def _collect_base(task, surface, config, tables, control=False):
    evaluated = evaluate_surface(task, surface, [0.25, 0.5, 0.75, 1.0])
    for row in evaluated["one_step"]: row.update(experiment_id="KDD098R", claim_boundary=CLAIM); tables["one_step_state_metrics.csv"].append(row)
    for row in evaluated["recursive"]: row.update(experiment_id="KDD098R", claim_boundary=CLAIM); tables["recursive_rollout_metrics.csv"].append(row)
    for row in evaluated["uncertainty"]: row.update(experiment_id="KDD098R", claim_boundary=CLAIM); tables["uncertainty_calibration.csv"].append(row)
    for row in evaluated["risk"]: row.update(experiment_id="KDD098R", claim_boundary=CLAIM); tables["risk_coverage_metrics.csv"].append(row)
    for row in evaluated["support"]:
        row.update(experiment_id="KDD098R", diagnostic_type="state_prediction_support_stratum", claim_boundary=CLAIM)
        tables["support_severity_propensity_diagnostics.csv"].append(row)
    for row in evaluated["resource"]: row.update(experiment_id="KDD098R", training_examples="not_applicable_control" if surface.method == "persistence_locf" else "train_only_rows", claim_boundary=CLAIM); tables["resource_metrics.csv"].append(row)


def _collect_result(task, result: RSurface, receipt: RFitReceipt | None, config, tables, extension):
    surface = result.surface
    _collect_base(task, surface, config, tables)
    tables["reward_prediction_or_propagation.csv"].extend([{**row, "claim_boundary": CLAIM} for row in reward_rows(task, result)])
    termination, survival = termination_rows(task, result)
    tables["termination_native_and_recalibrated.csv"].extend([{**row, "claim_boundary": CLAIM} for row in termination])
    tables["rollout_survival_sanity.csv"].extend([{**row, "claim_boundary": CLAIM} for row in survival])
    targeted, reward_action, event, delayed, strata = targeted_action_rows(task, result, int(config["action_information_contract"]["bootstrap_replicates"]), int(config["action_information_contract"]["bootstrap_seed"]))
    for destination, rows in (("targeted_action_information.csv", targeted), ("reward_action_information.csv", reward_action), ("action_event_window_diagnostics.csv", event), ("delayed_action_horizon_diagnostics.csv", delayed), ("support_severity_propensity_diagnostics.csv", strata)):
        tables[destination].extend([{**row, "shuffle_contract": "within_train_frozen_severity_propensity_time_strata", "claim_boundary": CLAIM} for row in rows])
    reward_available = task.task in {"sepsis", "respiratory", "shock"}
    convergence = receipt is None or (receipt.selected_epoch >= 1 and receipt.trained_epochs >= int(config["training_budget"]["min_epochs"]))
    ready = reward_available and convergence and bool(surface.checkpoint_sha256)
    component = {"experiment_id": "KDD098R", "task": task.task, "method_id": surface.method, "seed": surface.seed,
                 "P_available": True, "R_available": reward_available, "T_available": True, "O_available": False,
                 "R_source": "KDD099R-A_candidate_deterministic_state_to_reward" if reward_available else "no_reward_invented_or_transferred",
                 "convergence_disclosed": True, "required_hash_available": bool(surface.checkpoint_sha256), "planning_component_ready": ready, "claim_boundary": CLAIM}
    tables["component_availability.csv"].append(component)
    tables["planning_component_readiness.csv"].append({**component, "decision": "planning_component_ready" if ready else "world_model_only_not_planning_component_ready",
                                                        "policy_result_used": False, "policy_winner_declared": False})
    tables["checkpoint_hash_manifest.csv"].append({
        "experiment_id": "KDD098R", "checkpoint_id": f"KDD098R::{task.task}::{surface.method}::{surface.seed}", "task": task.task,
        "method_id": surface.method, "seed": surface.seed, "checkpoint_sha256": surface.checkpoint_sha256,
        "checkpoint_stored_internal_only": True, "checkpoint_exported_in_aggregate_package": False,
        "selected_epoch": receipt.selected_epoch if receipt else "derived", "trained_epochs": receipt.trained_epochs if receipt else "derived",
        "epoch_cap": (80 if extension else 40) if receipt else "derived", "planning_component_ready": ready, "claim_boundary": CLAIM,
    })
    if receipt:
        for epoch, (rmse, mae, composite) in enumerate(zip(receipt.epoch_rmse, receipt.epoch_mae, receipt.epoch_composite, strict=True), 1):
            tables["learning_curve_summary.csv"].append({"experiment_id": "KDD098R", "task": task.task, "method_id": surface.method, "seed": surface.seed,
                "epoch": epoch, "validation_rmse": rmse, "validation_mae": mae, "validation_composite": composite,
                "selected_epoch": epoch == receipt.selected_epoch, "selection_role": "validation", "claim_boundary": CLAIM})
        tables["best_epoch_and_cap_hits.csv"].append({"experiment_id": "KDD098R", "task": task.task, "method_id": surface.method, "seed": surface.seed,
            "selected_epoch": receipt.selected_epoch, "trained_epochs": receipt.trained_epochs, "cap_hit": receipt.cap_hit,
            "improving_at_cap": receipt.improving_at_cap, "early_stop": receipt.early_stop, "uniform_extension_applied": extension,
            "train_fit_episodes": receipt.fit_episodes, "train_calibration_episodes": receipt.calibration_episodes,
            "validation_episodes": receipt.validation_episodes, "checkpoint_opportunities": receipt.checkpoint_opportunities,
            "checkpoint_sha256": receipt.checkpoint_sha256, "claim_boundary": CLAIM})


def _ensemble_result(task, members, checkpoint_root):
    surface = gaussian_ensemble_surface(task, [member.surface for member in members], checkpoint_root)
    bundle = {"experiment_id": "KDD098R", "task": task.task, "method": "gaussian_transition_ensemble",
              "member_checkpoint_sha256": [member.surface.checkpoint_sha256 for member in members]}
    payload = (json.dumps(bundle, sort_keys=True, separators=(",", ":")) + "\n").encode(); digest = hashlib.sha256(payload).hexdigest()
    (checkpoint_root / f"{task.task}__gaussian_transition_ensemble__bundle.json").write_bytes(payload)
    surface.checkpoint_sha256 = digest
    native = np.mean(np.stack([member.termination_native for member in members]), axis=0).astype(np.float32)
    recal = np.mean(np.stack([member.termination_recalibrated for member in members]), axis=0).astype(np.float32)
    return RSurface(surface, np.mean(np.stack([m.previous_action_one for m in members]), axis=0).astype(np.float32),
                    surface.shuffled_one, surface.shuffled_recursive, native, recal,
                    float(np.mean([m.termination_recalibration_slope for m in members])), float(np.mean([m.termination_recalibration_intercept for m in members])))


def _classify_action(task, rows, config):
    if task.action_dim < 2 or len(np.unique(task.action_classes[task.valid_steps])) < 2: return "invalid_action_contract"
    threshold = float(config["action_information_contract"]["classification_threshold_absolute_delta_rmse"])
    separated = [row for row in rows if row["semantic_valid"] and np.isfinite(row["comparator_minus_observed_delta_rmse"]) and row["comparator_minus_observed_delta_rmse"] >= threshold and row["episode_cluster_ci95_low"] > 0]
    positive_strata = {row["diagnostic_stratum"] for row in rows if row["semantic_valid"] and np.isfinite(row["comparator_minus_observed_delta_rmse"]) and row["comparator_minus_observed_delta_rmse"] >= threshold}
    return "action_information_supported" if len(positive_strata) >= 2 and separated else "low_action_signal_stress_candidate"


def _action_classification_rows(tables, task):
    sources = (
        "targeted_action_information.csv", "reward_action_information.csv", "action_event_window_diagnostics.csv",
        "delayed_action_horizon_diagnostics.csv", "support_severity_propensity_diagnostics.csv",
    )
    return [row for source in sources for row in tables[source]
            if row.get("task") == task and "comparator_minus_observed_delta_rmse" in row]


def _coverage_rows(tables, failures):
    catalog = pd.read_csv(KDD095_CATALOG); completed = {(r["task"], r["method_id"]) for r in tables["checkpoint_hash_manifest.csv"]}
    rows = list(failures)
    for item in catalog[catalog["category"].eq("transition")].itertuples(index=False):
        execution, status = CATALOG_EXECUTION.get(item.method_id, ("not_run", "not_run_with_reason"))
        if item.method_id == "dreamer_v1_rssm": status = "not_run_with_reason"
        completed_tasks = sorted({task for task, method in completed if method == execution})
        if item.method_id in {"persistence_locf", "hgb_residual"}: completed_tasks = ["all_authorized_tasks"]
        rows.append({"experiment_id": "KDD098R", "task": "all_authorized_tasks", "method_id": item.method_id, "seed": "3408;3411;3414",
                     "receipt_type": "result" if completed_tasks else "not_run_with_reason", "fidelity_label": item.fidelity_label,
                     "execution_method_id": execution, "reason": "core paper suite or sanity control" if completed_tasks else NOT_RUN_REASONS.get(item.method_id, "non_core_KDD095_coverage_receipt_no_algorithm_zoo_retraining"),
                     "completed_task_count": 6 if completed_tasks else 0, "policy_winner_eligible": False, "claim_boundary": CLAIM})
    return rows


def _registry_rows(tables):
    rows = []
    for item in tables["checkpoint_hash_manifest.csv"]:
        rows.append({"experiment_id": "KDD098R", "task": item["task"], "method_id": item["method_id"], "seed": item["seed"],
                     "method_role": "core_world_model" if item["method_id"] != "gaussian_transition_ensemble" else "derived_probabilistic_world_model",
                     "fidelity_label": "local_control" if item["method_id"] == "gaussian_transition_ensemble" else "official_contract_adapter",
                     "result_available": True, "planning_component_ready": item["planning_component_ready"], "claim_boundary": CLAIM})
    rows.extend([{"experiment_id": "KDD098R", "task": "all_authorized_tasks", "method_id": x, "seed": "not_applicable", "method_role": "sanity_control", "fidelity_label": "local_control", "result_available": True, "planning_component_ready": False, "claim_boundary": CLAIM} for x in ("persistence_locf", "hgb_residual")])
    return rows


def _decision_rows(tables, tasks, classifications):
    rows = [{"experiment_id": "KDD098R", "decision_scope": "task", "task": task.task, "decision": classifications[task.task],
             "planning_ready_checkpoint_count": sum(r["task"] == task.task and r["planning_component_ready"] for r in tables["checkpoint_hash_manifest.csv"]),
             "policy_result_used": False, "existing_test_outcomes_opened": False, "claim_boundary": CLAIM} for task in tasks]
    expected = len(tasks) * 10; actual = len(tables["checkpoint_hash_manifest.csv"])
    ready = sum(r["planning_component_ready"] for r in tables["checkpoint_hash_manifest.csv"])
    failures = sum(r.get("receipt_type") not in {"result", "not_run_with_reason"} for r in tables["method_coverage_and_failure_receipts.csv"])
    rows.append({"experiment_id": "KDD098R", "decision_scope": "overall", "task": "all", "decision": "planning_candidate_checkpoint_lineage_frozen" if actual == expected and failures == 0 else "partial_planning_candidate_lineage_with_failures",
                 "frozen_checkpoint_count": actual, "expected_checkpoint_count": expected, "planning_ready_checkpoint_count": ready,
                 "action_information_supported_tasks": sum(v == "action_information_supported" for v in classifications.values()),
                 "low_action_signal_tasks": sum(v == "low_action_signal_stress_candidate" for v in classifications.values()),
                 "policy_winner_declared": False, "policy_result_used": False, "confirmatory_evaluation_authorized": False,
                 "existing_test_outcomes_opened": False, "claim_boundary": CLAIM})
    return rows


def _preflight(config):
    paths = {"prompt_v5": ROOT / ".omx/research/kdd_decision_pipeline_benchmark_prompts_v5.md",
             "kdd097_artifact_manifest": ROOT / "results/kdd097_rich_task_materialization_20260714_v2/artifact_hashes.json",
             "kdd097_decision": ROOT / "results/kdd097_rich_task_materialization_20260714_v2/decision.csv",
             "kdd099r_a_artifact_manifest": KDD099RA / "artifact_hashes.json", "kdd099r_a_candidate_decisions": KDD099RA / "candidate_reward_decisions.csv",
             "kdd099r_a_reward_formula": KDD099RA / "reward_formula_and_timing.csv", "kdd098_v2_artifact_manifest": V2_RESULT / "artifact_hashes.json",
             "kdd098_v2_checkpoint_manifest": V2_RESULT / "checkpoint_hash_manifest.csv"}
    for key, path in paths.items():
        if sha(path) != config["upstream_sha256"][key]: raise RuntimeError(f"KDD098R upstream hash drift: {key}")
    if tree_hash(V2_RESULT) != config["upstream_sha256"]["kdd098_v2_result_tree"] or tree_hash(V2_CHECKPOINT) != config["upstream_sha256"]["kdd098_v2_checkpoint_tree"]: raise RuntimeError("KDD098-v2 preservation hash drift")
    decisions = pd.read_csv(KDD099RA / "candidate_reward_decisions.csv")
    selected = set(decisions.loc[decisions["selected_for_KDD098R_KDD100R"].astype(bool), "candidate_reward"])
    expected = {"sepsis_lactate_delta_component", "resp_meddreamer_spo2_mbp", "resp_spo2_target_component", "shock_next_mbp_component", "shock_lactate_delta_component"}
    if selected != expected: raise RuntimeError("KDD099R-A selected candidate drift")


def _privacy_rows():
    return [{"experiment_id": "KDD098R", "check": check, "status": "pass", "detail": detail} for check, detail in (
        ("development_roles_only", "train_validation_only"), ("patient_membership", "not_exported"),
        ("row_level_trajectories_predictions_probabilities", "not_exported"), ("exact_timestamps_and_identifiers", "not_exported"),
        ("checkpoint_payload", "internal_only_hashes_in_aggregate_output"), ("test_lockbox_RV02R_policy_results", "not_opened"))]


def _report(tables, classifications, v2_result_hash, v2_checkpoint_hash):
    decision = tables["decision.csv"][-1]; cap = pd.DataFrame(tables["best_epoch_and_cap_hits.csv"])
    termination = pd.DataFrame(tables["termination_native_and_recalibrated.csv"])
    return "\n".join(["# KDD098R convergence and planning-component validation", "", "## Decision", "", f"`{decision['decision']}`", "",
        f"KDD098R froze {decision['frozen_checkpoint_count']} new core/ensemble checkpoint receipts; {decision['planning_ready_checkpoint_count']} have P, a KDD099R-A benchmark-candidate R, and T. No policy result was opened or used.", "",
        "## Convergence", "", f"Median selected epoch was {cap['selected_epoch'].median():.1f}; {int(cap['cap_hit'].sum())} seed runs hit their final cap and {int(cap['uniform_extension_applied'].sum())} seed runs used a uniform task-family extension. Selection used validation only after train-only fitting.", "",
        "## Components", "", "Sepsis uses the lactate-delta benchmark proxy; respiratory uses the SpO2/MBP adapter and SpO2 component; shock uses next-MBP and lactate-delta components. AF/flutter, AKI, and heart failure remain world-model-only. These reward weights are benchmark sensitivities, not clinical utilities.", "",
        f"Native mean termination Brier was {termination[termination['probability_version'].eq('native')]['brier'].mean():.6f}; train-calibration-subset recalibrated mean Brier was {termination[termination['probability_version'].eq('train_calibration_subset_recalibrated')]['brier'].mean():.6f}. Native and recalibrated rows remain separate.", "",
        "## Action information", "", *[f"- {task}: `{status}`" for task, status in classifications.items()], "",
        "All action comparisons are factual predictive diagnostics with episode-cluster intervals. They are not causal treatment effects. Low action information retains a checkpoint but changes the downstream stress-test role.", "",
        "## Preservation and boundary", "", f"KDD098-v2 result tree SHA256: `{v2_result_hash}`.", f"KDD098-v2 checkpoint tree SHA256: `{v2_checkpoint_hash}`.", "", CLAIM, ""])


def _failure(task, method, seed, stage, error):
    return {"experiment_id": "KDD098R", "task": task, "method_id": method, "seed": seed, "receipt_type": "execution_failure", "failure_stage": stage,
            "reason": re.sub(r"/[^ ]+", "[path_redacted]", f"{type(error).__name__}: {error}")[:300], "claim_boundary": CLAIM}


def _write_csv(path, rows):
    if not rows: raise RuntimeError(f"required KDD098R table is empty: {path.name}")
    fields = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def _privacy_scan(output):
    restricted = re.compile(r"(^|,)(subject_id|hadm_id|stay_id|patient_id|charttime|intime|outtime|filename|file_path)(,|$)", re.I)
    for path in output.iterdir():
        if path.suffix not in {".csv", ".md"}: continue
        text = path.read_text(encoding="utf-8", errors="replace"); first = text.splitlines()[0] if text else ""
        if restricted.search(first) or "/" + "home" + "/" in text or "mimiciv/" in text.lower(): raise RuntimeError(f"restricted aggregate content: {path.name}")


def _write_hashes(output):
    payload = {"experiment_id": "KDD098R", "version": "kdd098r-convergence-planning-components-v1", "aggregate_only": True, "hash_algorithm": "sha256",
               "artifacts": {path.name: sha(path) for path in sorted(output.iterdir()) if path.name != "artifact_hashes.json"}}
    (output / "artifact_hashes.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hash(path):
    rows = [f"{item.relative_to(path)}:{sha(item)}" for item in sorted(path.rglob("*")) if item.is_file()]
    return hashlib.sha256(("\n".join(rows) + "\n").encode()).hexdigest()


if __name__ == "__main__": main()
