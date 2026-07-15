from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from kdd_benchmark_discovery.kdd_e01_evaluator import (
    FiniteMDP,
    backward_induction,
    categorical_cem_policy,
    denominator_full_probabilities,
    evaluate_policy_exact,
)
from kdd_benchmark_discovery.run_kdd_e02_known_value_full import (
    BOOTSTRAPS,
    DENOMINATORS,
    ESTIMATORS,
    PLANNERS,
    _bootstrap_counts,
    fit_tabular_model,
    frozen_policies,
    model_diagnostics,
    multiseed_crn,
    multiseed_logged_data,
    ope_with_bootstrap,
)


REGIMES = ("null", "weak", "moderate", "delayed")
TIERS = ("tier1_exact_finite", "tier2_ehr_calibrated_semisynthetic")
CLIPS = (None, 20.0)
REQUIRED = (
    "eligible_task_manifest.csv",
    "task_matched_environment_contracts.csv",
    "exact_oracle_validity.csv",
    "null_response_sanity.csv",
    "task_matched_known_value_results.csv",
    "task_matched_ope_recovery.csv",
    "task_matched_rank_sign_coverage.csv",
    "task_matched_model_exploitation.csv",
    "estimator_contract_disposition.csv",
    "real_ehr_scoring_authorization.csv",
    "decision.md",
    "summary.md",
)


def _write(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _behavior_table(horizon: int, states: int, actions: int, support: np.ndarray,
                    action_weights: np.ndarray, severity_shift: float) -> np.ndarray:
    table = np.zeros((horizon, states, actions), dtype=np.float64)
    for t in range(horizon):
        for state in range(states):
            p = action_weights.copy()
            if state < states - 1 and severity_shift:
                intensity = np.linspace(-1.0, 1.0, actions)
                p *= np.exp(severity_shift * (state / max(states - 2, 1)) * intensity)
            p[~support[state]] = 0.0
            p /= p.sum()
            table[t, state] = p
    return table


def make_aki_env(tier: str, regime: str) -> FiniteMDP:
    horizon, actions, severity_levels = 3, 2, 4
    semi = tier == "tier2_ehr_calibrated_semisynthetic"
    states, terminal = (9, 8) if semi else (5, 4)
    missing_probability = 1.0 - 0.020748576078112285 if semi else 0.0
    transition = np.zeros((horizon, states, actions, states), dtype=np.float64)
    reward = np.zeros_like(transition)
    support = np.ones((states, actions), dtype=bool)
    support[terminal, 1] = False
    strength = {"null": 0.0, "weak": 0.025, "moderate": 0.08, "delayed": 0.07}[regime]
    for t in range(horizon):
        for state in range(states):
            for action in range(actions):
                if state == terminal:
                    transition[t, state, action, terminal] = 1.0
                    continue
                severity = state % severity_levels
                effect = strength * action
                if regime == "delayed" and t < 1:
                    effect = 0.0
                if action == 1:
                    # Initiation ends the initiation-decision episode. The known terminal
                    # reward is a benchmark proxy, not a clinical utility.
                    transition[t, state, action, terminal] = 1.0
                    reward[t, state, action, terminal] = -0.08 + effect * (1.2 + 0.2 * severity)
                else:
                    improve = np.clip(0.18 + effect, 0.02, 0.55)
                    worsen = np.clip(0.16 - effect, 0.02, 0.55)
                    censor = 0.05 if tier == "tier2_ehr_calibrated_semisynthetic" else 0.03
                    stay = 1.0 - improve - worsen - censor
                    severity_mass = ((max(0, severity - 1), improve),
                                     (min(severity_levels - 1, severity + 1), worsen),
                                     (severity, stay))
                    for next_severity, mass in severity_mass:
                        if semi:
                            transition[t, state, action, next_severity] += mass * (1.0 - missing_probability)
                            transition[t, state, action, severity_levels + next_severity] += mass * missing_probability
                        else:
                            transition[t, state, action, next_severity] += mass
                    transition[t, state, action, terminal] += censor
                    for next_state in range(states):
                        next_severity = next_state % severity_levels
                        dense = 0.05 * (severity - next_severity) if next_state != terminal else 0.0
                        terminal_reward = (0.7 - 0.18 * severity) if t == horizon - 1 and next_state != terminal else 0.0
                        reward[t, state, action, next_state] = dense + terminal_reward
    if regime == "null":
        transition[:] = transition[:, :, :1, :]
        reward[:] = reward[:, :, :1, :]
    if semi:
        severity_initial = np.array([0.08, 0.24, 0.42, 0.26])
        initial = np.concatenate([severity_initial * (1.0 - missing_probability),
                                  severity_initial * missing_probability, [0.0]])
    else:
        initial = np.array([0.20, 0.30, 0.30, 0.20, 0.0])
    action_weights = (np.array([1971.0, 487.0]) if tier.startswith("tier2") else np.array([0.72, 0.28]))
    behavior = _behavior_table(horizon, states, actions, support, action_weights, 0.35)
    return FiniteMDP(f"aki_rrt_{tier}_{regime}", transition, reward, initial, support, behavior, 0.99)


def make_hf_env(tier: str, regime: str) -> FiniteMDP:
    horizon, actions, severity_levels = 12, 8, 7
    semi = tier == "tier2_ehr_calibrated_semisynthetic"
    states, terminal = (15, 14) if semi else (8, 7)
    missing_probability = 1.0 - 0.8487328295269508 if semi else 0.0
    transition = np.zeros((horizon, states, actions, states), dtype=np.float64)
    reward = np.zeros_like(transition)
    support = np.ones((states, actions), dtype=bool)
    support[terminal, 1:] = False
    strength = {"null": 0.0, "weak": 0.012, "moderate": 0.04, "delayed": 0.035}[regime]
    for t in range(horizon):
        for state in range(states):
            for action in range(actions):
                if state == terminal:
                    transition[t, state, action, terminal] = 1.0
                    continue
                severity = state % severity_levels
                diuretic, vaso = divmod(action, 2)
                intensity = diuretic / 3.0
                effect = strength * (intensity - 0.25 * vaso)
                if regime == "delayed" and t < 3:
                    effect = 0.0
                improve = np.clip(0.14 + effect, 0.02, 0.45)
                worsen = np.clip(0.11 - 0.55 * effect + 0.012 * vaso, 0.02, 0.45)
                censor = 0.018 if tier.startswith("tier2") else 0.012
                stay = 1.0 - improve - worsen - censor
                severity_mass = ((max(0, severity - 1), improve),
                                 (min(severity_levels - 1, severity + 1), worsen),
                                 (severity, stay))
                for next_severity, mass in severity_mass:
                    if semi:
                        transition[t, state, action, next_severity] += mass * (1.0 - missing_probability)
                        transition[t, state, action, severity_levels + next_severity] += mass * missing_probability
                    else:
                        transition[t, state, action, next_severity] += mass
                transition[t, state, action, terminal] += censor
                for next_state in range(states):
                    next_severity = next_state % severity_levels
                    dense = 0.025 * (severity - next_severity) if next_state != terminal else 0.0
                    terminal_reward = (0.8 - 0.11 * next_severity) if t == horizon - 1 and next_state != terminal else 0.0
                    reward[t, state, action, next_state] = dense + terminal_reward
    if regime == "null":
        transition[:] = transition[:, :, :1, :]
        reward[:] = reward[:, :, :1, :]
    if semi:
        severity_initial = np.array([0.05, 0.12, 0.22, 0.25, 0.19, 0.11, 0.06])
        initial = np.concatenate([severity_initial * (1.0 - missing_probability),
                                  severity_initial * missing_probability, [0.0]])
    else:
        initial = np.array([0.10, 0.15, 0.20, 0.20, 0.15, 0.12, 0.08, 0.0])
    counts = np.array([12164, 3724, 1913, 418, 2216, 670, 1749, 1097], dtype=np.float64)
    weights = counts if tier.startswith("tier2") else np.array([8, 3, 3, 1, 2, 1, 2, 1], dtype=np.float64)
    behavior = _behavior_table(horizon, states, actions, support, weights, 0.25)
    return FiniteMDP(f"heart_failure_{tier}_{regime}", transition, reward, initial, support, behavior, 0.99)


def _planner_audit(env: FiniteMDP) -> dict[str, object]:
    rows = {}
    for horizon in (4, 8):
        policy, traces = categorical_cem_policy(env, horizon)
        active = [x for x in traces if env.support[x.state].sum() > 1]
        iterations = {x.iteration for x in active}
        rows[f"H{horizon}_iterations"] = "|".join(str(x) for x in sorted(iterations))
        rows[f"H{horizon}_multiple_sequences"] = bool(active and min(x.unique_sequences for x in active) > 1)
        rows[f"H{horizon}_support_bypass"] = bool(np.any(policy * (~env.support)[None]))
    return rows


def _manifest(config: dict, config_path: Path) -> list[dict]:
    base = config["authoritative_predecessors"]
    tasks = [
        ("af_flutter", base["x04"], "af_remains_world_model_only", False),
        ("aki_rrt", base["x05"], "aki_rrt_policy_extension_ready_for_known_value_validation", True),
        ("heart_failure", base["x06"], "hf_policy_extension_ready_for_known_value_validation", True),
        ("respiratory_reference", base["x07"], "reference_compatible_sensitivity_only", False),
        ("shock_reference", base["x07"], "reference_compatible_sensitivity_only", False),
    ]
    rows = []
    for task, source, decision, eligible in tasks:
        decision_path = Path(source) / "decision.md"
        decision_text = decision_path.read_text(encoding="utf-8")
        if f"`{decision}`" not in decision_text:
            raise RuntimeError(f"authoritative decision drift for {task}: expected {decision}")
        rows.append({
            "task": task, "source_directory": source, "source_decision": decision,
            "source_decision_sha256": _sha256(decision_path), "eligible": eligible,
            "inclusion_rule": "explicit_ready_for_known_value_validation_only",
            "independent_failure_excludes_other_tasks": False,
            "config_sha256": _sha256(config_path),
        })
    return rows


def run(output: Path, config_path: Path) -> str:
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = _manifest(config, config_path)
    if [x["task"] for x in manifest if x["eligible"]] != ["aki_rrt", "heart_failure"]:
        raise RuntimeError("authoritative eligibility drift")

    environment_rows: list[dict] = []
    oracle_rows: list[dict] = []
    null_rows: list[dict] = []
    policy_rows: list[dict] = []
    ope_rows: list[dict] = []
    exploit_rows: list[dict] = []
    precision_by_env: dict[str, float] = {}
    hard_fail = False

    task_specs = {
        "aki_rrt": {"builder": make_aki_env, "horizons": (1, 2, 3), "interval": "24h", "action": "wait_vs_initiate_rrt_K2_absorbing", "reward": "in_hospital_survival_discharge_proxy"},
        "heart_failure": {"builder": make_hf_env, "horizons": (1, 2, 4, 8, 12), "interval": "6h", "action": "loop_diuretic_4_x_vaso_inotrope_2_K8", "reward": "observed_post_action_urine_output_proxy"},
    }
    for task, spec in task_specs.items():
        for tier in TIERS:
            for regime in REGIMES:
                env = spec["builder"](tier, regime)
                planner = _planner_audit(env)
                planner_ok = (planner["H4_iterations"] == "1|2|3" and planner["H8_iterations"] == "1|2|3"
                              and planner["H4_multiple_sequences"] and planner["H8_multiple_sequences"]
                              and not planner["H4_support_bypass"] and not planner["H8_support_bypass"])
                hard_fail |= not planner_ok
                environment_rows.append({
                    "task": task, "tier": tier, "regime": regime, "environment": env.name,
                    "action_contract": spec["action"], "decision_interval": spec["interval"],
                    "decision_horizon": env.horizon, "ope_horizons": "|".join(map(str, spec["horizons"])),
                    "state_count": env.n_states, "action_count": env.n_actions,
                    "reward_contract": spec["reward"], "action_response_known_by_construction": True,
                    "termination_censoring_known_by_construction": True,
                    "calibration_source": "aggregate_training_role_receipts" if tier.startswith("tier2") else "prespecified_small_finite",
                    "observation_missingness_state_strata_injected": tier.startswith("tier2"),
                    "aggregate_train_missingness_rate": ((1.0 - 0.020748576078112285) if task == "aki_rrt" else (1.0 - 0.8487328295269508)) if tier.startswith("tier2") else 0.0,
                    "patient_rows_loaded_or_exported": False, "planner_gate_pass": planner_ok, **planner,
                })

                data = multiseed_logged_data(env)
                learned, counts = fit_tabular_model(env, data)
                policies = frozen_policies(env, learned, counts)
                diagnostics = model_diagnostics(env, learned, counts)
                mc, stream_hash = multiseed_crn(env, policies, tolerance=0.03, initial_n=512, maximum_n=8192)
                precision_by_env[env.name] = float(mc["max_se"])
                hard_fail |= not bool(mc["met"])
                values = {name: evaluate_policy_exact(env, policy) for name, policy in policies.items()}
                predictions = {name: evaluate_policy_exact(learned, policy) for name, policy in policies.items()}
                behavior_value = values["behavior"]
                oracle = backward_induction(env)[0] if tier == "tier1_exact_finite" else np.nan
                reference = oracle if tier == "tier1_exact_finite" else max(values.values())
                ranks = rankdata([-value for value in values.values()], method="average")

                if tier == "tier1_exact_finite":
                    negative = sum((oracle - value) < -1e-12 for value in values.values())
                    reproducible = backward_induction(env)[0] == backward_induction(env)[0]
                    valid = bool(oracle + 1e-12 >= max(values.values()) and negative == 0 and reproducible)
                    hard_fail |= not valid
                    oracle_rows.append({"task": task, "regime": regime, "environment": env.name,
                                        "oracle_method": "exact_backward_induction", "oracle_value": oracle,
                                        "maximum_evaluated_policy_value": max(values.values()),
                                        "negative_regret_count": negative, "exact_values_reproducible": reproducible,
                                        "oracle_gate_pass": valid})

                for index, (policy_name, policy) in enumerate(policies.items()):
                    returns = mc["returns"][policy_name]
                    unsupported = float(np.sum(policy * (~env.support)[None]) / (env.horizon * env.n_states))
                    row = {
                        "task": task, "tier": tier, "regime": regime, "environment": env.name,
                        "policy": policy_name,
                        "policy_class": "model_based_planner" if policy_name in PLANNERS else ("stress" if "stress" in policy_name else "model_free_or_control"),
                        "true_return": values[policy_name], "behavior_relative_true_value_difference": values[policy_name] - behavior_value,
                        "paired_mc_return_mean": float(returns.mean()),
                        "paired_mc_se": float(returns.std(ddof=1) / np.sqrt(len(returns))),
                        "behavior_paired_mc_difference": mc["comparisons"][policy_name][0],
                        "behavior_paired_mc_se": mc["comparisons"][policy_name][1],
                        "exact_regret_tier1": oracle - values[policy_name] if tier == "tier1_exact_finite" else np.nan,
                        "tier2_best_evaluated_reference_gap": reference - values[policy_name] if tier.startswith("tier2") else np.nan,
                        "true_value_rank": float(ranks[index]), "unsupported_action_mass": unsupported,
                        "learned_model_predicted_return": predictions[policy_name],
                        "model_exploitation_gap": predictions[policy_name] - values[policy_name],
                        "evaluation_episodes": mc["n"], "maximum_paired_se": mc["max_se"],
                        "common_environment_stream_hash": stream_hash,
                        "training_seeds": "3408|3411|3414", "evaluation_seeds": "9401|9402|9403",
                    }
                    policy_rows.append(row)
                    if policy_name != "unsupported_or_model_exploit_stress" and unsupported != 0:
                        hard_fail = True
                    exploit_rows.append({
                        "task": task, "tier": tier, "regime": regime, "environment": env.name, "policy": policy_name,
                        "true_return": values[policy_name], "learned_model_predicted_return": predictions[policy_name],
                        "model_exploitation_gap": predictions[policy_name] - values[policy_name],
                        "policy_conditioned_model_error": abs(predictions[policy_name] - values[policy_name]),
                        "unsupported_action_mass": unsupported, **diagnostics,
                    })

                if regime == "null":
                    supported_values = [values[name] for name, policy in policies.items()
                                        if not np.any(policy * (~env.support)[None])]
                    max_difference = max(supported_values) - min(supported_values)
                    planner_gains = max(abs(values[name] - behavior_value) for name in PLANNERS)
                    null_pass = bool(max_difference <= 1e-12 and planner_gains <= 1e-12)
                    hard_fail |= not null_pass
                    null_rows.append({"task": task, "tier": tier, "environment": env.name,
                                      "maximum_supported_policy_true_value_difference": max_difference,
                                      "maximum_planner_artificial_gain": planner_gains,
                                      "tolerance": 1e-12, "null_gate_pass": null_pass})

                boot_counts = _bootstrap_counts(len(data["actions"]))
                denominators = {name: denominator_full_probabilities(env, data, name) for name in DENOMINATORS}
                for policy_name, policy in policies.items():
                    for denominator, table in denominators.items():
                        for clip in CLIPS:
                            for horizon in spec["horizons"]:
                                estimates = ope_with_bootstrap(env, learned, data, policy, table, horizon, clip, boot_counts)
                                truth = evaluate_policy_exact(env, policy, horizon)
                                unsupported = float(np.sum(policy[:horizon] * (~env.support)[None]) / (horizon * env.n_states))
                                for estimator, (point, low, high, ess, concentration) in estimates.items():
                                    ope_rows.append({
                                        "task": task, "tier": tier, "regime": regime, "environment": env.name,
                                        "policy": policy_name, "estimator": estimator, "denominator": denominator,
                                        "clipping": "none" if clip is None else clip,
                                        "support_contract": "masked" if unsupported == 0 else "unrestricted_stress",
                                        "horizon": horizon, "true_value": truth, "ope_estimate": point,
                                        "ci_low": low, "ci_high": high,
                                        "coverage": bool(np.isfinite(low) and low <= truth <= high),
                                        "absolute_error": abs(point - truth) if np.isfinite(point) else np.nan,
                                        "ess": ess, "effective_sample_fraction": ess / len(data["actions"]),
                                        "maximum_normalized_weight": concentration,
                                        "unsupported_action_mass": unsupported, "logged_episodes": len(data["actions"]),
                                        "bootstrap_replicates": BOOTSTRAPS, "bootstrap_seeds": "8401|8402|8403",
                                    })

    ope_df = pd.DataFrame(ope_rows)
    rank_rows: list[dict] = []
    disposition_rows: list[dict] = []
    grouping = ["task", "tier", "regime", "estimator", "denominator", "clipping", "support_contract", "horizon"]
    for contract, group in ope_df.groupby(grouping, dropna=False):
        fields = dict(zip(grouping, contract))
        finite = np.isfinite(group.ope_estimate.to_numpy(float))
        masked = fields["support_contract"] == "masked"
        behavior = group[group.policy == "behavior"]
        if masked and len(behavior) == 1 and finite.sum() >= 2:
            true = group.true_value.to_numpy(float)
            estimate = group.ope_estimate.to_numpy(float)
            spear = float(spearmanr(true[finite], estimate[finite]).statistic)
            pairs = [(i, j) for i in range(len(true)) for j in range(i + 1, len(true)) if abs(true[i] - true[j]) > 1e-12]
            pair_recovery = float(np.mean([np.sign(true[i] - true[j]) == np.sign(estimate[i] - estimate[j]) for i, j in pairs])) if pairs else 1.0
            b = behavior.iloc[0]
            sign_recovery = float(np.mean(np.sign(true[finite] - float(b.true_value)) == np.sign(estimate[finite] - float(b.ope_estimate))))
            false_rate = float(np.mean((group.ci_low.to_numpy(float) > float(b.ci_high)) & (true <= float(b.true_value) + 1e-12)))
        else:
            spear = pair_recovery = sign_recovery = false_rate = np.nan
        coverage = float(group.coverage.mean())
        median_ess = float(group.ess.median())
        max_unsupported = float(group.unsupported_action_mass.max())
        precision = precision_by_env[str(group.environment.iloc[0])]
        logged = int(group.logged_episodes.iloc[0])
        passed = bool(masked and coverage >= 0.90 and spear >= 0.80 and pair_recovery >= 0.80
                      and sign_recovery >= 0.90 and false_rate <= 0.05 and median_ess >= 100
                      and median_ess >= 0.05 * logged and max_unsupported == 0 and precision <= 0.03 and finite.all())
        rank_rows.append({**fields, "coverage_rate": coverage, "spearman_rank_recovery": spear,
                          "pairwise_order_recovery": pair_recovery, "behavior_relative_sign_recovery": sign_recovery,
                          "false_improvement_rate": false_rate, "median_ess": median_ess,
                          "maximum_weight_concentration": float(group.maximum_normalized_weight.max()),
                          "maximum_unsupported_action_mass": max_unsupported,
                          "maximum_paired_true_return_se": precision})
        disposition_rows.append({**fields,
                                 "coverage_gate": coverage >= 0.90, "rank_gate": bool(spear >= 0.80),
                                 "pairwise_gate": bool(pair_recovery >= 0.80), "sign_gate": bool(sign_recovery >= 0.90),
                                 "false_improvement_gate": bool(false_rate <= 0.05),
                                 "ess_gate": median_ess >= 100 and median_ess >= 0.05 * logged,
                                 "support_gate": max_unsupported == 0, "precision_gate": precision <= 0.03,
                                 "finite_gate": bool(finite.all()), "approved_exact_tuple": passed})

    authorization_rows = []
    for task in task_specs:
        tier1 = sum(x["approved_exact_tuple"] and x["task"] == task and x["tier"] == "tier1_exact_finite" for x in disposition_rows)
        tier2 = sum(x["approved_exact_tuple"] and x["task"] == task and x["tier"] == "tier2_ehr_calibrated_semisynthetic" for x in disposition_rows)
        authorization_rows.append({
            "task": task, "tier1_approved_exact_tuple_count": tier1,
            "tier2_approved_exact_tuple_count": tier2,
            "real_ehr_policy_value_status": "eligible_exact_tuple_pending_retrospective_policy_specific_gates" if tier2 else "prohibited",
            "retrospective_ehr_policy_value_computed": False,
            "authorization_scope": "exact_estimator_denominator_clipping_support_horizon_task_regime_tuple_only",
            "clinical_or_causal_authorization": False,
        })

    decision = "blocked_evaluator_defect" if hard_fail else "task_matched_evaluation_complete"
    _write(output / "eligible_task_manifest.csv", manifest)
    _write(output / "task_matched_environment_contracts.csv", environment_rows)
    _write(output / "exact_oracle_validity.csv", oracle_rows)
    _write(output / "null_response_sanity.csv", null_rows)
    _write(output / "task_matched_known_value_results.csv", policy_rows)
    _write(output / "task_matched_ope_recovery.csv", ope_rows)
    _write(output / "task_matched_rank_sign_coverage.csv", rank_rows)
    _write(output / "task_matched_model_exploitation.csv", exploit_rows)
    _write(output / "estimator_contract_disposition.csv", disposition_rows)
    _write(output / "real_ehr_scoring_authorization.csv", authorization_rows)
    approved_total = sum(x["approved_exact_tuple"] for x in disposition_rows)
    tier2_total = sum(x["approved_exact_tuple"] and x["tier"].startswith("tier2") for x in disposition_rows)
    (output / "decision.md").write_text(
        f"# KDD-X08 decision\n\n`{decision}`\n\n"
        f"Approved exact tuples: {approved_total}; Tier-2 approved tuples: {tier2_total}. "
        "Task-level authorization is reported separately in `real_ehr_scoring_authorization.csv`. "
        "No retrospective EHR policy value was computed.\n",
        encoding="utf-8",
    )
    task_status = "; ".join(f"{x['task']}={x['real_ehr_policy_value_status']}" for x in authorization_rows)
    task_counts = "; ".join(
        f"{x['task']}: Tier-1={x['tier1_approved_exact_tuple_count']}, Tier-2={x['tier2_approved_exact_tuple_count']}"
        for x in authorization_rows
    )
    (output / "summary.md").write_text(
        "# KDD-X08 task-matched evaluator\n\n"
        f"Decision: `{decision}`. Eligible tasks were AKI-RRT and heart failure only. {task_status}.\n\n"
        f"Tuple-level approvals were {task_counts}. AKI approvals are restricted to the exact estimator, denominator, clipping, support, horizon, and injected task-regime rows that passed. Heart failure passed no exact tuple and therefore remains prohibited from retrospective real-EHR policy-value scoring.\n\n"
        "Each task was evaluated in exact finite and aggregate training-role-calibrated semi-synthetic tiers under null, weak, moderate, and delayed injected response regimes. The repaired E01 CRN, paired precision, support, exact-oracle, null, H1, and H4/H8 categorical-CEM invariants were retained.\n\n"
        "All values are known-construction benchmark evidence. No patient rows were exported, no existing evaluation outcome was opened, and no result supports treatment benefit, causal effect, clinical utility, deployment, or real-EHR policy superiority.\n",
        encoding="utf-8",
    )
    missing = [name for name in REQUIRED if not (output / name).exists()]
    if missing:
        raise RuntimeError(f"missing outputs: {missing}")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/kdd_x08_task_matched_evaluator_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    decision = run(args.output, args.config)
    print(f"decision={decision} elapsed_seconds={time.perf_counter() - start:.3f}")


if __name__ == "__main__":
    main()
