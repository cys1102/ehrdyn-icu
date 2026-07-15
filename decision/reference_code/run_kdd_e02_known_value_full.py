from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import time
import hashlib

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

from kdd_benchmark_discovery.kdd_e01_evaluator import (
    HORIZONS,
    FiniteMDP,
    adaptive_crn_evaluation,
    backward_induction,
    behavior_policy,
    categorical_cem_policy,
    denominator_full_probabilities,
    evaluate_policy_exact,
    exact_qv,
    generate_logged_data,
    h1_exhaustive_policy,
    make_finite_mdp,
    make_sepsis_semisynthetic_mdp,
    make_streams,
    simulate_policy,
    paired_precision,
)


BOOTSTRAPS = 500
DENOMINATORS = ("exact_behavior", "misspecified_behavior", "paper_lstm_h16", "crossfit_stronger")
CLIPS = (None, 20.0)
ESTIMATORS = ("IS", "WIS", "PDIS", "WPDIS", "CWPDIS", "DR", "WDR", "FQE", "support_restricted_WPDIS")
PLANNERS = ("H1_exhaustive", "H4_categorical_CEM", "H8_categorical_CEM", "uncertainty_penalized_H4_categorical_CEM")
REQUIRED = (
    "known_value_policy_results.csv", "paired_policy_return_differences.csv",
    "ope_value_recovery.csv", "ope_rank_and_sign_recovery.csv",
    "ope_coverage_false_improvement.csv", "model_exploitation_results.csv",
    "prediction_to_policy_relationship.csv", "estimator_contract_disposition.csv",
    "planning_guardrail_disposition.csv", "decision.md", "summary.md",
)


def _write(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")


def _normalize(policy: np.ndarray, support: np.ndarray | None = None) -> np.ndarray:
    p = np.asarray(policy, dtype=np.float64).copy()
    if support is not None:
        p *= support[None]
    total = p.sum(axis=-1, keepdims=True)
    if support is None:
        fallback = np.full_like(p, 1.0 / p.shape[-1])
    else:
        fallback = np.broadcast_to(support / support.sum(axis=1, keepdims=True), p.shape)
    return np.where(total > 0, p / np.maximum(total, 1e-12), fallback)


def fit_tabular_model(env: FiniteMDP, data: dict[str, np.ndarray]) -> tuple[FiniteMDP, np.ndarray]:
    states, actions, rewards = data["states"], data["actions"], data["rewards"]
    counts = np.zeros((env.horizon, env.n_states, env.n_actions), dtype=np.float64)
    transitions = np.full((env.horizon, env.n_states, env.n_actions, env.n_states), 0.25, dtype=np.float64)
    reward_sum = np.zeros_like(transitions)
    reward_count = np.zeros_like(transitions)
    for t in range(env.horizon):
        np.add.at(counts[t], (states[:, t], actions[:, t]), 1.0)
        np.add.at(transitions[t], (states[:, t], actions[:, t], states[:, t + 1]), 1.0)
        np.add.at(reward_sum[t], (states[:, t], actions[:, t], states[:, t + 1]), rewards[:, t])
        np.add.at(reward_count[t], (states[:, t], actions[:, t], states[:, t + 1]), 1.0)
    transitions /= transitions.sum(axis=-1, keepdims=True)
    reward = np.divide(reward_sum, reward_count, out=np.zeros_like(reward_sum), where=reward_count > 0)
    learned = FiniteMDP(f"learned_{env.name}", transitions, reward, env.initial.copy(), env.support.copy(),
                        env.behavior.copy(), env.discount)
    return learned, counts


def frozen_policies(env: FiniteMDP, learned: FiniteMDP, counts: np.ndarray) -> dict[str, np.ndarray]:
    h, s, a = env.horizon, env.n_states, env.n_actions
    policies: dict[str, np.ndarray] = {"behavior": behavior_policy(env)}
    policies["supported_random"] = np.broadcast_to(env.support / env.support.sum(axis=1, keepdims=True), (h, s, a)).copy()
    for name, choose_max in (("no_min_action", False), ("max_supported_action", True)):
        p = np.zeros((h, s, a))
        for state in range(s):
            choices = np.flatnonzero(env.support[state])
            p[:, state, choices[-1] if choose_max else choices[0]] = 1.0
        policies[name] = p
    severity = np.zeros((h, s, a))
    for state in range(s):
        choices = np.flatnonzero(env.support[state])
        index = min(len(choices) - 1, int((state / max(s - 2, 1)) * len(choices)))
        severity[:, state, choices[index]] = 1.0
    policies["severity_rule"] = severity
    bc = _normalize(counts + 0.5, env.support)
    policies["tabular_behavior_cloning"] = bc
    _, greedy, q = backward_induction(learned)
    bcq = np.zeros_like(greedy)
    cql = np.zeros_like(greedy)
    spibb = np.zeros_like(greedy)
    for t in range(h):
        for state in range(s):
            supported = env.support[state]
            bc_threshold = np.maximum(0.02, 0.1 * bc[t, state].max())
            eligible = supported & (bc[t, state] >= bc_threshold)
            if not eligible.any():
                eligible = supported
            score = q[t, state].copy()
            score[~eligible] = -np.inf
            bcq[t, state, np.argmax(score)] = 1.0
            penalized = q[t, state] - 0.5 / np.sqrt(counts[t, state] + 1.0)
            penalized[~supported] = -np.inf
            cql[t, state, np.argmax(penalized)] = 1.0
            if counts[t, state].sum() >= 50:
                spibb[t, state] = 0.5 * bc[t, state] + 0.5 * greedy[t, state]
            else:
                spibb[t, state] = bc[t, state]
    policies["tabular_discrete_BCQ"] = bcq
    policies["tabular_discrete_CQL"] = cql
    policies["tabular_Soft_SPIBB"] = _normalize(spibb, env.support)
    policies["H1_exhaustive"] = h1_exhaustive_policy(learned)
    policies["H4_categorical_CEM"] = categorical_cem_policy(learned, 4)[0]
    policies["H8_categorical_CEM"] = categorical_cem_policy(learned, 8)[0]
    policies["uncertainty_penalized_H4_categorical_CEM"] = categorical_cem_policy(learned, 4, uncertainty_penalty=0.25)[0]
    stress = np.zeros((h, s, a))
    for state in range(s):
        unsupported = np.flatnonzero(~env.support[state])
        action = int(unsupported[-1]) if len(unsupported) else int(np.argmax(q[0, state]))
        stress[:, state, action] = 1.0
    policies["unsupported_or_model_exploit_stress"] = stress
    return policies


def model_diagnostics(env: FiniteMDP, learned: FiniteMDP, counts: np.ndarray) -> dict[str, float]:
    support4 = env.support[None, :, :, None]
    transition_error = float(np.sqrt(np.mean(np.square((learned.transition - env.transition)[support4.repeat(env.horizon, 0).repeat(env.n_states, 3)]))))
    reward_error = float(np.sqrt(np.mean(np.square((learned.reward - env.reward)[support4.repeat(env.horizon, 0).repeat(env.n_states, 3)]))))
    behavior_true = evaluate_policy_exact(env, env.behavior)
    behavior_pred = evaluate_policy_exact(learned, env.behavior)
    rare = counts < 10
    support_error = float(np.mean(np.abs(learned.transition - env.transition)[np.repeat(rare[..., None], env.n_states, axis=-1)]))
    calibration = float(abs(np.mean(counts < 1) - np.mean(np.max(np.abs(learned.transition - env.transition), axis=-1) > 0.25)))
    return {"one_step_error": transition_error, "recursive_rollout_error": abs(behavior_pred - behavior_true),
            "reward_error": reward_error, "uncertainty_calibration_error": calibration,
            "support_stratified_error": support_error}


def _bootstrap_counts(n: int) -> np.ndarray:
    probability = np.full(n, 1.0 / n)
    rows = []
    allocation = (167, 167, 166)
    for seed, replicates in zip((8401, 8402, 8403), allocation):
        rng = np.random.default_rng(seed)
        rows.extend(rng.multinomial(n, probability) for _ in range(replicates))
    return np.asarray(rows, dtype=np.float64)


def multiseed_logged_data(env: FiniteMDP) -> dict[str, np.ndarray]:
    parts = [generate_logged_data(env, n=n, seed=seed) for seed, n in zip((3408, 3411, 3414), (342, 341, 341))]
    return {key: np.concatenate([part[key] for part in parts], axis=0)
            for key in ("states", "actions", "rewards")}


def multiseed_crn(env: FiniteMDP, policies: dict[str, np.ndarray], tolerance: float = 0.03,
                  initial_n: int = 512, maximum_n: int = 8192) -> tuple[dict, str]:
    names = list(policies)
    n = initial_n
    while True:
        base, remainder = divmod(n, 3)
        sizes = [base + (i < remainder) for i in range(3)]
        combined = {name: [] for name in names}
        hashes = []
        unsupported = defaultdict(int)
        for index, (seed, size) in enumerate(zip((9401, 9402, 9403), sizes)):
            streams = make_streams(size, env.horizon, names, environment_seed=seed,
                                   policy_seed_base=19401 + index * 1000)
            hashes.append(streams.environment_hash())
            for name, policy in policies.items():
                returns, count = simulate_policy(env, policy, streams, name)
                combined[name].append(returns)
                unsupported[name] += count
        results = {name: np.concatenate(value) for name, value in combined.items()}
        reference = results[names[0]]
        comparisons = {name: paired_precision(value, reference) for name, value in results.items()}
        max_se = max(se for _, se in comparisons.values())
        if max_se <= tolerance or n >= maximum_n:
            break
        n = min(maximum_n, n * 2)
    digest = hashlib.sha256("|".join(hashes).encode()).hexdigest()
    return {"n": n, "max_se": max_se, "met": max_se <= tolerance,
            "returns": results, "comparisons": comparisons, "unsupported": dict(unsupported)}, digest


def ope_with_bootstrap(env: FiniteMDP, learned: FiniteMDP, data: dict[str, np.ndarray], target: np.ndarray,
                       denominator_full: np.ndarray, horizon: int, clip: float | None,
                       bootstrap_counts: np.ndarray) -> dict[str, tuple[float, float, float, float, float]]:
    h = horizon
    states, actions, rewards = data["states"][:, :h + 1], data["actions"][:, :h], data["rewards"][:, :h]
    n = len(actions)
    target_prob = np.empty((n, h))
    for t in range(h):
        target_prob[:, t] = target[t, states[:, t], actions[:, t]]
    denom = np.take_along_axis(denominator_full[:, :h], actions[:, :, None], axis=2)[:, :, 0]
    ratio = np.divide(target_prob, denom, out=np.zeros_like(target_prob), where=denom > 0)
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
    wis_num, wis_den = final * returns, final
    estimates["WIS"] = (float(wis_num.sum() / max(wis_den.sum(), 1e-12)),
                         (b @ wis_num) / np.maximum(b @ wis_den, 1e-12))
    pdis_contrib = np.sum(cumulative * rewards * discount, axis=1)
    estimates["PDIS"] = (float(pdis_contrib.mean()), b @ pdis_contrib / b.sum(axis=1))
    step_num = cumulative * rewards
    step_den = cumulative
    wp_point = float(np.sum(np.divide(step_num.sum(axis=0), step_den.sum(axis=0), out=np.zeros(h), where=step_den.sum(axis=0)>0) * discount))
    wp_boot = np.zeros(BOOTSTRAPS)
    for t in range(h):
        wp_boot += discount[t] * (b @ step_num[:, t]) / np.maximum(b @ step_den[:, t], 1e-12)
    estimates["WPDIS"] = (wp_point, wp_boot)
    at_risk = states[:, :-1] != env.n_states - 1
    cw_num, cw_den = cumulative * rewards * at_risk, cumulative * at_risk
    cw_point = float(np.sum(np.divide(cw_num.sum(axis=0), cw_den.sum(axis=0), out=np.zeros(h), where=cw_den.sum(axis=0)>0) * discount))
    cw_boot = np.zeros(BOOTSTRAPS)
    for t in range(h):
        cw_boot += discount[t] * (b @ cw_num[:, t]) / np.maximum(b @ cw_den[:, t], 1e-12)
    estimates["CWPDIS"] = (cw_point, cw_boot)
    q, v = exact_qv(learned, target, h)
    dr_contrib = v[0, states[:, 0]].copy()
    for t in range(h):
        residual = rewards[:, t] + env.discount * v[t + 1, states[:, t + 1]] - q[t, states[:, t], actions[:, t]]
        dr_contrib += discount[t] * cumulative[:, t] * residual
    estimates["DR"] = (float(dr_contrib.mean()), b @ dr_contrib / b.sum(axis=1))
    wdr_point = float(v[0, states[:, 0]].mean())
    wdr_boot = b @ v[0, states[:, 0]] / b.sum(axis=1)
    for t in range(h):
        residual = rewards[:, t] + env.discount * v[t + 1, states[:, t + 1]] - q[t, states[:, t], actions[:, t]]
        wdr_point += discount[t] * float(np.sum(cumulative[:, t] * residual) / max(cumulative[:, t].sum(), 1e-12))
        wdr_boot += discount[t] * (b @ (cumulative[:, t] * residual)) / np.maximum(b @ cumulative[:, t], 1e-12)
    estimates["WDR"] = (wdr_point, wdr_boot)
    fqe_episode = v[0, states[:, 0]]
    estimates["FQE"] = (float(fqe_episode.mean()), b @ fqe_episode / b.sum(axis=1))
    unsupported_mass = float(np.sum(target * (~env.support)[None]) / (env.horizon * env.n_states))
    if unsupported_mass == 0:
        estimates["support_restricted_WPDIS"] = estimates["WPDIS"]
    else:
        estimates["support_restricted_WPDIS"] = (np.nan, np.full(BOOTSTRAPS, np.nan))
    output = {}
    for name, (point, samples) in estimates.items():
        finite = samples[np.isfinite(samples)]
        low, high = (float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))) if len(finite) else (np.nan, np.nan)
        output[name] = (point, low, high, ess, concentration)
    return output


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(np.nan_to_num(p_values, nan=1.0))
    adjusted = np.ones(len(p_values))
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * (1.0 if not np.isfinite(p_values[index]) else p_values[index]))
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def run(output: Path) -> str:
    output.mkdir(parents=True, exist_ok=False)
    policy_rows, pair_rows, ope_rows, rank_rows, coverage_rows = [], [], [], [], []
    exploit_rows, relation_source, disposition_rows, guardrail_rows = [], [], [], []
    ope_boot: dict[tuple, np.ndarray] = {}
    hard_fail = False
    environments = [make_finite_mdp(x) for x in ("null", "weak", "moderate", "delayed")]
    environments += [make_sepsis_semisynthetic_mdp(x) for x in ("null", "weak", "moderate", "delayed")]
    for env_index, env in enumerate(environments):
        tier = "tier2_ehr_calibrated_semisynthetic" if env.n_actions == 25 else "tier1_exact_finite"
        regime = env.name.replace("sepsis_K25_", "")
        data = multiseed_logged_data(env)
        learned, counts = fit_tabular_model(env, data)
        policies = frozen_policies(env, learned, counts)
        diagnostics = model_diagnostics(env, learned, counts)
        mc, stream_hash = multiseed_crn(env, policies, tolerance=0.03, initial_n=512, maximum_n=8192)
        exact_values = {name: evaluate_policy_exact(env, policy) for name, policy in policies.items()}
        predicted_values = {name: evaluate_policy_exact(learned, policy) for name, policy in policies.items()}
        behavior_value = exact_values["behavior"]
        oracle = backward_induction(env)[0] if tier == "tier1_exact_finite" else np.nan
        reference = oracle if tier == "tier1_exact_finite" else max(exact_values.values())
        rank_map = {name: float(rankdata([-value for value in exact_values.values()], method="average")[i])
                    for i, name in enumerate(exact_values)}
        for name, policy in policies.items():
            returns = mc["returns"][name]
            mean = float(returns.mean())
            se = float(returns.std(ddof=1) / np.sqrt(len(returns)))
            paired_delta, paired_se = mc["comparisons"][name]
            unsupported_mass = float(np.sum(policy * (~env.support)[None]) / (env.horizon * env.n_states))
            row = {"tier": tier, "regime": regime, "environment": env.name, "policy": name,
                   "policy_class": "model_based_planner" if name in PLANNERS else ("stress" if "stress" in name else "model_free_or_control"),
                   "evaluation_episodes": mc["n"], "true_return": exact_values[name],
                   "paired_mc_return_mean": mean, "paired_mc_se": se,
                   "paired_mc_ci_low": mean - 1.96 * se, "paired_mc_ci_high": mean + 1.96 * se,
                   "behavior_relative_true_value_difference": exact_values[name] - behavior_value,
                   "behavior_paired_mc_difference": paired_delta, "behavior_paired_mc_se": paired_se,
                   "exact_regret_tier1": oracle - exact_values[name] if tier == "tier1_exact_finite" else np.nan,
                   "tier2_best_evaluated_reference_gap": reference - exact_values[name] if tier != "tier1_exact_finite" else np.nan,
                   "true_value_rank": rank_map[name], "unsupported_action_mass": unsupported_mass,
                   "learned_model_predicted_return": predicted_values[name],
                   "model_exploitation_gap": predicted_values[name] - exact_values[name],
                   "environment_stream_hash": stream_hash,
                   "training_seeds": "3408|3411|3414", "evaluation_seeds": "9401|9402|9403"}
            policy_rows.append(row)
            exploit_rows.append({**{k: row[k] for k in ("tier", "regime", "environment", "policy")},
                                 "learned_model_predicted_return": predicted_values[name], "true_return": exact_values[name],
                                 "model_exploitation_gap": predicted_values[name] - exact_values[name],
                                 "policy_conditioned_model_error": abs(predicted_values[name] - exact_values[name]),
                                 "planner_disagreement": abs(predicted_values[name] - behavior_value),
                                 "predictive_uncertainty_proxy": float(1.0 / np.sqrt(np.maximum(counts.sum(), 1.0))),
                                 "unsupported_action_mass": unsupported_mass})
            relation_source.append({**{k: row[k] for k in ("tier", "regime", "environment", "policy", "true_return", "model_exploitation_gap")},
                                    **diagnostics,
                                    "policy_conditioned_model_error": abs(predicted_values[name] - exact_values[name])})
        names = list(policies)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                delta = mc["returns"][left] - mc["returns"][right]
                pair_rows.append({"tier": tier, "regime": regime, "environment": env.name,
                                  "policy_a": left, "policy_b": right,
                                  "exact_true_difference": exact_values[left] - exact_values[right],
                                  "paired_mc_difference": float(delta.mean()),
                                  "paired_mc_se": float(delta.std(ddof=1) / np.sqrt(len(delta))),
                                  "paired_ci_low": float(delta.mean() - 1.96 * delta.std(ddof=1) / np.sqrt(len(delta))),
                                  "paired_ci_high": float(delta.mean() + 1.96 * delta.std(ddof=1) / np.sqrt(len(delta)))})
        boot_counts = _bootstrap_counts(len(data["actions"]))
        denominator_tables = {name: denominator_full_probabilities(env, data, name) for name in DENOMINATORS}
        for policy_name, policy in policies.items():
            for denominator, denominator_table in denominator_tables.items():
                for clip in CLIPS:
                    for horizon in HORIZONS:
                        estimates = ope_with_bootstrap(env, learned, data, policy, denominator_table, horizon, clip, boot_counts)
                        truth = evaluate_policy_exact(env, policy, horizon)
                        unsupported_mass = float(np.sum(policy[:horizon] * (~env.support)[None]) / (horizon * env.n_states))
                        for estimator, (point, low, high, ess, concentration) in estimates.items():
                            key = (tier, regime, estimator, denominator, "none" if clip is None else str(clip),
                                   "masked" if unsupported_mass == 0 else "unrestricted_stress", horizon, policy_name)
                            # Reconstructing bootstrap samples is unnecessary for the final table; paired false-improvement
                            # uses conservative interval separation against the behavior interval below.
                            ope_rows.append({"tier": tier, "regime": regime, "environment": env.name,
                                             "policy": policy_name, "estimator": estimator,
                                             "denominator": denominator, "clipping": "none" if clip is None else clip,
                                             "support_contract": key[5], "horizon": horizon,
                                             "true_value": truth, "ope_estimate": point, "ci_low": low, "ci_high": high,
                                             "coverage": bool(np.isfinite(low) and low <= truth <= high),
                                             "absolute_error": abs(point - truth) if np.isfinite(point) else np.nan,
                                             "relative_error": abs(point - truth) / max(abs(truth), 0.1) if np.isfinite(point) else np.nan,
                                             "ess": ess, "effective_sample_fraction": ess / len(data["actions"]),
                                             "maximum_normalized_weight": concentration,
                                             "unsupported_action_mass": unsupported_mass,
                                             "bootstrap_replicates": BOOTSTRAPS,
                                             "bootstrap_seeds": "8401|8402|8403",
                                             "logged_episodes": len(data["actions"])})
        if tier == "tier1_exact_finite" and any(row["exact_regret_tier1"] < -1e-12 for row in policy_rows if row["environment"] == env.name):
            hard_fail = True
        if not mc["met"]:
            hard_fail = True

    ope_df = pd.DataFrame(ope_rows)
    for contract, group in ope_df.groupby(["tier", "regime", "estimator", "denominator", "clipping", "support_contract", "horizon"], dropna=False):
        tier, regime, estimator, denominator, clipping, support_contract, horizon = contract
        if support_contract != "masked":
            rank_rows.append({"tier": tier, "regime": regime, "estimator": estimator, "denominator": denominator,
                              "clipping": clipping, "support_contract": support_contract, "horizon": horizon,
                              "spearman_rank_recovery": np.nan, "pairwise_order_recovery": np.nan,
                              "behavior_relative_sign_recovery": np.nan, "eligible_policy_count": 0})
            coverage_rows.append({"tier": tier, "regime": regime, "estimator": estimator, "denominator": denominator,
                                  "clipping": clipping, "support_contract": support_contract, "horizon": horizon,
                                  "coverage_rate": float(group.coverage.mean()), "false_improvement_rate": np.nan,
                                  "median_ess": float(group.ess.median()),
                                  "max_unsupported_action_mass": float(group.unsupported_action_mass.max()),
                                  "maximum_paired_true_return_se": np.nan})
            disposition_rows.append({"tier": tier, "regime": regime, "estimator": estimator, "denominator": denominator,
                                     "clipping": clipping, "support_contract": support_contract, "horizon": horizon,
                                     "coverage_gate": False, "rank_gate": False, "pairwise_gate": False,
                                     "sign_gate": False, "false_improvement_gate": False, "ess_gate": False,
                                     "support_gate": False, "precision_gate": False,
                                     "finite_gate": bool(np.isfinite(group.ope_estimate).all()),
                                     "approved_exact_contract": False})
            continue
        behavior = group[group.policy == "behavior"].iloc[0]
        eligible = group[group.support_contract == "masked"].copy()
        true = eligible.true_value.to_numpy(float)
        estimate = eligible.ope_estimate.to_numpy(float)
        finite = np.isfinite(estimate)
        if finite.sum() >= 2:
            spear = float(spearmanr(true[finite], estimate[finite]).statistic)
            pairs = [(i, j) for i in range(len(true)) for j in range(i + 1, len(true)) if abs(true[i]-true[j]) > 1e-12]
            pair_recovery = float(np.mean([np.sign(true[i]-true[j]) == np.sign(estimate[i]-estimate[j]) for i, j in pairs])) if pairs else 1.0
        else:
            spear, pair_recovery = np.nan, np.nan
        true_sign = np.sign(true - float(behavior.true_value))
        est_sign = np.sign(estimate - float(behavior.ope_estimate))
        sign_recovery = float(np.mean(true_sign[finite] == est_sign[finite])) if finite.any() else 0.0
        declared = eligible.ci_low.to_numpy(float) > float(behavior.ci_high)
        false_improvement = declared & (true <= float(behavior.true_value) + 1e-12)
        false_rate = float(false_improvement.mean())
        coverage = float(eligible.coverage.mean())
        median_ess = float(eligible.ess.median())
        max_unsupported = float(eligible.unsupported_action_mass.max())
        max_precision = max(float(r["behavior_paired_mc_se"]) for r in policy_rows if r["tier"] == tier and r["regime"] == regime and r["unsupported_action_mass"] == 0)
        logged_episodes = int(eligible.logged_episodes.iloc[0])
        passed = bool(coverage >= 0.90 and spear >= 0.80 and pair_recovery >= 0.80 and sign_recovery >= 0.90
                      and false_rate <= 0.05 and median_ess >= 100 and median_ess >= 0.05 * logged_episodes
                      and max_unsupported == 0 and max_precision <= 0.03 and finite.all())
        rank_rows.append({"tier": tier, "regime": regime, "estimator": estimator, "denominator": denominator,
                          "clipping": clipping, "support_contract": support_contract, "horizon": horizon,
                          "spearman_rank_recovery": spear, "pairwise_order_recovery": pair_recovery,
                          "behavior_relative_sign_recovery": sign_recovery, "eligible_policy_count": len(eligible)})
        coverage_rows.append({"tier": tier, "regime": regime, "estimator": estimator, "denominator": denominator,
                              "clipping": clipping, "support_contract": support_contract, "horizon": horizon,
                              "coverage_rate": coverage, "false_improvement_rate": false_rate,
                              "median_ess": median_ess, "max_unsupported_action_mass": max_unsupported,
                              "maximum_paired_true_return_se": max_precision})
        disposition_rows.append({"tier": tier, "regime": regime, "estimator": estimator, "denominator": denominator,
                                 "clipping": clipping, "support_contract": support_contract, "horizon": horizon,
                                 "coverage_gate": coverage >= 0.90, "rank_gate": spear >= 0.80,
                                 "pairwise_gate": pair_recovery >= 0.80, "sign_gate": sign_recovery >= 0.90,
                                 "false_improvement_gate": false_rate <= 0.05,
                                 "ess_gate": median_ess >= 100 and median_ess >= 0.05 * logged_episodes,
                                 "support_gate": max_unsupported == 0, "precision_gate": max_precision <= 0.03,
                                 "finite_gate": bool(finite.all()), "approved_exact_contract": passed})

    policy_df = pd.DataFrame(policy_rows)
    exploit_df = pd.DataFrame(exploit_rows)
    for _, row in policy_df[policy_df.policy.isin(PLANNERS)].iterrows():
        null_gain = abs(row.behavior_relative_true_value_difference) if row.regime == "null" else 0.0
        passed = bool(row.unsupported_action_mass == 0 and row.behavior_paired_mc_se <= 0.03
                      and null_gain <= 1e-12 and abs(row.model_exploitation_gap) <= 0.25)
        guardrail_rows.append({"tier": row.tier, "regime": row.regime, "planner": row.policy,
                               "unsupported_action_mass": row.unsupported_action_mass,
                               "paired_true_return_se": row.behavior_paired_mc_se,
                               "null_artificial_improvement": null_gain,
                               "absolute_model_exploitation_gap": abs(row.model_exploitation_gap),
                               "guardrail_pass": passed})

    relation = pd.DataFrame(relation_source)
    relation_rows = []
    predictors = ("one_step_error", "recursive_rollout_error", "reward_error", "uncertainty_calibration_error",
                  "support_stratified_error", "policy_conditioned_model_error")
    outcomes = ("true_return", "model_exploitation_gap")
    p_values = []
    for predictor in predictors:
        for outcome in outcomes:
            x, y = relation[predictor].to_numpy(float), relation[outcome].to_numpy(float)
            pear = pearsonr(x, y) if np.std(x) > 0 and np.std(y) > 0 else (np.nan, np.nan)
            spear = spearmanr(x, y) if np.std(x) > 0 and np.std(y) > 0 else (np.nan, np.nan)
            relation_rows.append({"predictor": predictor, "outcome": outcome, "sample_unit": "environment_policy_row",
                                  "n": len(x), "pearson_r": float(pear.statistic), "pearson_p": float(pear.pvalue),
                                  "spearman_rho": float(spear.statistic), "spearman_p": float(spear.pvalue),
                                  "analysis_label": "exploratory_shared_environment_and_model_dependence"})
            p_values.append(float(spear.pvalue))
    adjusted = holm_adjust(p_values)
    for row, value in zip(relation_rows, adjusted):
        row["holm_adjusted_spearman_p"] = value

    approved = sum(bool(row["approved_exact_contract"]) for row in disposition_rows)
    tier2_approved = sum(bool(row["approved_exact_contract"]) and row["tier"] == "tier2_ehr_calibrated_semisynthetic"
                         for row in disposition_rows)
    known_value_disposition = "exact_known_value_contracts_approved" if approved else "no_exact_known_value_contract_approved"
    real_ehr_disposition = ("tier2_exact_tuple_prerequisites_passed_pending_retrospective_gates"
                            if tier2_approved else "no_real_ehr_policy_value_estimator_approved")
    decision = "known_value_full_run_complete" if not hard_fail else "blocked_known_value_integrity_failure"
    _write(output / "known_value_policy_results.csv", policy_rows)
    _write(output / "paired_policy_return_differences.csv", pair_rows)
    _write(output / "ope_value_recovery.csv", ope_rows)
    _write(output / "ope_rank_and_sign_recovery.csv", rank_rows)
    _write(output / "ope_coverage_false_improvement.csv", coverage_rows)
    _write(output / "model_exploitation_results.csv", exploit_rows)
    _write(output / "prediction_to_policy_relationship.csv", relation_rows)
    _write(output / "estimator_contract_disposition.csv", disposition_rows)
    _write(output / "planning_guardrail_disposition.csv", guardrail_rows)
    (output / "decision.md").write_text(
        f"# KDD-E02 decision\n\n`{decision}`\n\n"
        f"`known_value_exact_tuple_disposition = {known_value_disposition}`\n\n"
        f"`real_ehr_policy_value_disposition = {real_ehr_disposition}`\n\n"
        f"Approved exact known-value tuples: {approved}; Tier-2 approved tuples: {tier2_approved}. "
        "Approval is tuple-specific and does not by itself authorize retrospective EHR OPE.\n",
        encoding="utf-8")
    (output / "summary.md").write_text(
        "# KDD-E02 known-value full run\n\n"
        f"Decision: `{decision}`. Known-value disposition: `{known_value_disposition}`. "
        f"Real-EHR disposition: `{real_ehr_disposition}`.\n\n"
        "Both exact finite and aggregate-calibrated K25 semi-synthetic tiers were evaluated with matched common random numbers. "
        "Tier-1 regret uses backward induction; Tier-2 gaps use the best evaluated true-model policy and are not oracle regret.\n\n"
        f"The frozen matrix produced {len(policy_rows)} policy rows and {len(ope_rows)} policy-estimator rows. "
        f"Exactly {approved} tuples passed, of which {tier2_approved} were Tier 2. "
        "Unsupported/exploitative stress policies are retained but ineligible for approval. Correlation analyses are exploratory.\n\n"
        "No result supports a real-EHR causal response, treatment benefit, clinical utility, deployment, or policy-winner claim.\n",
        encoding="utf-8")
    missing = [name for name in REQUIRED if not (output / name).exists()]
    if missing:
        raise RuntimeError(missing)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    start = time.perf_counter()
    decision = run(args.output)
    print(f"decision={decision} elapsed_seconds={time.perf_counter()-start:.3f}")


if __name__ == "__main__":
    main()
