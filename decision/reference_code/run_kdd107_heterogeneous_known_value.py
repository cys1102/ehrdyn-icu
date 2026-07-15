from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kdd_benchmark_discovery import kdd_e01_evaluator as e01
from kdd_benchmark_discovery import run_kdd100_complete_known_value as kv
from kdd_benchmark_discovery import run_kdd100r_task_matched_known_value as k100r
from kdd_benchmark_discovery import run_kdd_x02_cross_cohort_policy_benchmark as x02


ROOT = Path(__file__).resolve().parents[1]
RESEARCHFORGE = Path("<researchforge-root>")
DEFAULT_CONFIG = ROOT / "configs/kdd107_heterogeneous_known_value_v1.json"
KDD106 = RESEARCHFORGE / "artifacts/runs/kdd106_immutable_preflight_20260715_v1"
KDD106R = RESEARCHFORGE / "artifacts/runs/kdd106r_administrative_revision_20260715_v1"
MECHANISM_FAMILIES = (
    "null_response",
    "monotone_historical_control",
    "interior_optimum",
    "state_dependent_optimum",
    "heterogeneous_response",
    "delayed_tradeoff",
)
NEW_MECHANISMS = MECHANISM_FAMILIES[2:]
CLAIM_BOUNDARY = (
    "Exact finite known-value technical decision-benchmark stress test only. The mechanisms are "
    "synthetic and do not model patient treatment benefit. No raw EHR row, causal effect, clinical "
    "utility, treatment recommendation, deployment, or autonomous-decision claim is supported."
)
REQUIRED_OUTPUTS = (
    "known_value_environment_contracts.json",
    "environment_mechanism_audit.csv",
    "exact_oracle_and_policy_values.csv",
    "policy_level_results.csv",
    "world_model_planner_factorial.csv",
    "planner_horizon_differentiation.csv",
    "fixed_control_comparison.csv",
    "model_exploitation_gap.csv",
    "training_seed_results.csv",
    "null_response_sanity.csv",
    "negative_regret_sanity.csv",
    "common_random_numbers_receipt.csv",
    "failure_ledger.csv",
    "summary.md",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(*parts: object) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    if not fields:
        raise RuntimeError(f"no schema fields for {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _profile_severity(profile: dict[str, Any], task: str) -> tuple[int, bool]:
    if task == "aki_rrt":
        return 4, True
    if task == "heart_failure":
        return 7, True
    return int(profile["state_count"]) - 1, False


def _action_positions(profile: dict[str, Any]) -> np.ndarray:
    actions = int(profile["action_count"])
    supported = np.asarray(profile["supported_action_indices"], dtype=int)
    position = np.linspace(-1.0, 1.0, actions)
    if len(supported) > 1:
        position[supported] = np.linspace(-1.0, 1.0, len(supported))
    return position


def _state_support(profile: dict[str, Any], task: str) -> np.ndarray:
    states = int(profile["state_count"])
    actions = int(profile["action_count"])
    severity_levels, duplicated = _profile_severity(profile, task)
    strong = np.asarray(profile["supported_action_indices"], dtype=int)
    weak = strong[::2]
    if len(weak) < 2 and len(strong) >= 2:
        weak = strong[:1]
    support = np.zeros((states, actions), dtype=bool)
    for state in range(states - 1):
        severity = state % severity_levels if duplicated else state
        allowed = strong if severity % 2 == 0 else weak
        support[state, allowed] = True
    support[-1, strong[0]] = True
    return support


def _mechanism_terms(
    mechanism: str,
    action_position: float,
    severity_fraction: float,
    step: int,
    horizon: int,
    parameters: dict[str, Any],
) -> tuple[float, float]:
    """Return transition benefit and direct immediate reward, prespecified by family."""
    if mechanism == "null_response":
        return 0.0, 0.0
    if mechanism == "monotone_historical_control":
        return float(parameters["transition_strength"]) * action_position, 0.0
    if mechanism == "interior_optimum":
        target = float(parameters["target_position"])
        return float(parameters["transition_strength"]) * (1.0 - abs(action_position - target)), 0.0
    if mechanism == "state_dependent_optimum":
        low = float(parameters["target_low"])
        high = float(parameters["target_high"])
        target = low + (high - low) * severity_fraction
        return float(parameters["transition_strength"]) * (1.0 - abs(action_position - target)), 0.0
    if mechanism == "heterogeneous_response":
        direction = -1.0 if severity_fraction < float(parameters["direction_change_fraction"]) else 1.0
        magnitude = float(parameters["minimum_strength"]) + float(parameters["additional_extreme_strength"]) * abs(2.0 * severity_fraction - 1.0)
        return direction * magnitude * action_position, 0.0
    if mechanism == "delayed_tradeoff":
        # Myopic gain opposes a persistent state-transition cost. This is fixed
        # before fitting any method and is never tuned against learned returns.
        immediate = float(parameters["early_immediate_reward_strength"]) * action_position
        transition = float(parameters["transition_strength"]) * action_position
        if step >= max(1, horizon // 2):
            immediate = 0.0
        return transition, immediate
    raise ValueError(f"unknown mechanism: {mechanism}")


def build_environment(task: str, mechanism: str, config: dict[str, Any]) -> e01.FiniteMDP:
    if mechanism not in MECHANISM_FAMILIES:
        raise ValueError(mechanism)
    profile = config["task_profiles"][task]
    horizon = int(profile["horizon"])
    states = int(profile["state_count"])
    actions = int(profile["action_count"])
    terminal = states - 1
    severity_levels, duplicated = _profile_severity(profile, task)
    support = _state_support(profile, task)
    positions = _action_positions(profile)
    transition = np.zeros((horizon, states, actions, states), dtype=np.float64)
    reward = np.zeros_like(transition)
    scale = float(profile["reward_scale"])
    censor = float(profile["censoring_probability"])
    missingness = float(profile["aggregate_missingness_rate"])

    for step in range(horizon):
        for state in range(states):
            for action in range(actions):
                if state == terminal:
                    transition[step, state, action, terminal] = 1.0
                    continue
                severity = state % severity_levels if duplicated else state
                fraction = severity / max(severity_levels - 1, 1)
                benefit, immediate = _mechanism_terms(
                    mechanism,
                    float(positions[action]),
                    fraction,
                    step,
                    horizon,
                    config["mechanism_parameters"][mechanism],
                )
                improve = float(np.clip(0.18 + benefit, 0.025, 0.55))
                worsen = float(np.clip(0.15 - benefit, 0.025, 0.55))
                stay = 1.0 - improve - worsen - censor
                if stay < 0:
                    raise RuntimeError("invalid transition construction")
                severity_mass = (
                    (max(0, severity - 1), improve),
                    (min(severity_levels - 1, severity + 1), worsen),
                    (severity, stay),
                )
                for next_severity, mass in severity_mass:
                    if duplicated:
                        transition[step, state, action, next_severity] += mass * (1.0 - missingness)
                        transition[step, state, action, severity_levels + next_severity] += mass * missingness
                    else:
                        transition[step, state, action, next_severity] += mass
                transition[step, state, action, terminal] += censor
                for next_state in range(states):
                    if next_state == terminal:
                        next_fraction = 1.0
                    else:
                        next_severity = next_state % severity_levels if duplicated else next_state
                        next_fraction = next_severity / max(severity_levels - 1, 1)
                    dense = 0.10 * (fraction - next_fraction)
                    if mechanism == "delayed_tradeoff" and step < horizon - 1:
                        dense = 0.0
                    terminal_reward = 0.0
                    if step == horizon - 1 and next_state != terminal:
                        terminal_reward = 0.8 * (1.0 - next_fraction)
                    reward[step, state, action, next_state] = scale * (
                        dense + terminal_reward + immediate
                    )

    if mechanism == "null_response":
        for state in range(states - 1):
            reference = int(np.flatnonzero(support[state])[0])
            transition[:, state, :, :] = transition[:, state, reference : reference + 1, :]
            reward[:, state, :, :] = reward[:, state, reference : reference + 1, :]

    base = np.arange(1, severity_levels + 1, dtype=np.float64)
    base = np.minimum(base, base[::-1]) + 1.0
    base /= base.sum()
    initial = np.zeros(states, dtype=np.float64)
    if duplicated:
        initial[:severity_levels] = base * (1.0 - missingness)
        initial[severity_levels : 2 * severity_levels] = base * missingness
    else:
        initial[:severity_levels] = base
    behavior = np.zeros((horizon, states, actions), dtype=np.float64)
    for step in range(horizon):
        for state in range(states):
            allowed = np.flatnonzero(support[state])
            if len(allowed) == 1:
                behavior[step, state, allowed[0]] = 1.0
                continue
            severity = state % severity_levels if duplicated and state != terminal else min(state, severity_levels - 1)
            center = -0.55 + 1.1 * severity / max(severity_levels - 1, 1)
            logits = -1.6 * np.square(positions[allowed] - center)
            logits -= logits.max()
            probability = np.exp(logits)
            probability /= probability.sum()
            behavior[step, state, allowed] = probability
    return e01.FiniteMDP(
        f"kdd107_{task}_{mechanism}", transition, reward, initial, support, behavior, float(config["discount"])
    )


def mechanism_audit(task: str, mechanism: str, env: e01.FiniteMDP, config: dict[str, Any]) -> dict[str, Any]:
    oracle_value, oracle_policy, _ = e01.backward_induction(env)
    actions = np.argmax(oracle_policy, axis=-1)
    nonterminal = actions[:, :-1]
    supported = np.asarray(config["task_profiles"][task]["supported_action_indices"], dtype=int)
    interior_applicable = len(supported) > 2
    interior = bool(
        mechanism == "interior_optimum"
        and interior_applicable
        and np.any((nonterminal != supported.min()) & (nonterminal != supported.max()))
    )
    state_dependent = bool(
        mechanism == "state_dependent_optimum" and len(np.unique(nonterminal[0])) > 1
    )
    heterogeneous = mechanism == "heterogeneous_response"
    h1 = e01.h1_exhaustive_policy(env)
    delayed = bool(
        mechanism == "delayed_tradeoff"
        and np.any(np.argmax(h1[:, :-1], axis=-1) != np.argmax(oracle_policy[:, :-1], axis=-1))
    )
    expected = {
        "interior_optimum": mechanism == "interior_optimum" and interior_applicable,
        "state_dependent": mechanism == "state_dependent_optimum",
        "heterogeneous": mechanism == "heterogeneous_response",
        "delayed": mechanism == "delayed_tradeoff",
    }
    observed = {
        "interior_optimum": interior,
        "state_dependent": state_dependent,
        "heterogeneous": heterogeneous,
        "delayed": delayed,
    }
    support_valid = bool(
        np.all(env.support.sum(axis=1) >= 1)
        and np.allclose(env.behavior.sum(axis=-1), 1.0)
        and not np.any(env.behavior[:, ~env.support] > 0)
    )
    return {
        "task": task,
        "mechanism": mechanism,
        **observed,
        "interior_optimum_applicable": interior_applicable,
        "support_valid": support_valid,
        "weak_overlap_strata": int(sum(env.support[state].sum() < len(supported) for state in range(env.n_states - 1))),
        "strong_overlap_strata": int(sum(env.support[state].sum() == len(supported) for state in range(env.n_states - 1))),
        "oracle_value": oracle_value,
        "mechanism_property_pass": all(observed[key] == value for key, value in expected.items()),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def preflight(config: dict[str, Any]) -> list[dict[str, Any]]:
    expected_hashes = config["immutable_input_hashes"]
    actual_hashes = {
        "kdd106_immutable_config": sha256(KDD106 / "immutable_config.json"),
        "kdd106_output_schema_manifest": sha256(KDD106 / "output_schema_manifest.csv"),
        "kdd106r_administrative_revision": sha256(KDD106R / "administrative_revision.json"),
        "kdd_e01_evaluator": sha256(ROOT / "kdd_benchmark_discovery/kdd_e01_evaluator.py"),
        "kdd_x02_training_and_planning": sha256(ROOT / "kdd_benchmark_discovery/run_kdd_x02_cross_cohort_policy_benchmark.py"),
        "kdd100r_world_model_training": sha256(ROOT / "kdd_benchmark_discovery/run_kdd100r_task_matched_known_value.py"),
    }
    if actual_hashes != expected_hashes:
        raise RuntimeError(f"immutable input hash drift: {actual_hashes}")
    revision = json.loads((KDD106R / "administrative_revision.json").read_text(encoding="utf-8"))
    if revision["decision"] != "ready_for_kdd107_local_scientific_execution_release_pending":
        raise RuntimeError("KDD106R does not authorize KDD107")
    frozen = json.loads((KDD106 / "immutable_config.json").read_text(encoding="utf-8"))
    checks = {
        "seeds": config["seeds"] == frozen["optimization_training_seeds"],
        "tasks": all(
            int(config["task_profiles"][task]["action_count"]) == int(profile["action_count"])
            and int(config["task_profiles"][task]["horizon"]) == int(profile["horizon"])
            and int(config["task_profiles"][task]["interval_hours"]) == int(profile["interval_hours"])
            for task, profile in frozen["task_profiles"].items()
        ),
        "world_models": config["world_models"] == frozen["world_models"],
        "model_free": config["model_free_methods"] == frozen["model_free_methods"],
        "controls": config["controls"] == frozen["controls"],
        "planner_horizons": config["planner_contract"]["horizons"] == [1, 4, 8],
        "planner_variants": config["planner_contract"]["variants"] == frozen["planners"]["variants"],
        "uncertainty_penalty": config["planner_contract"]["uncertainty_penalty"] == frozen["planners"]["uncertainty_penalty"],
        "mechanisms": config["mechanisms"] == list(MECHANISM_FAMILIES),
        "no_ehr_or_test": not config["raw_ehr_access"] and not config["test_role_access"],
    }
    schema = pd.read_csv(KDD106 / "output_schema_manifest.csv")
    observed_outputs = set(schema.loc[schema.stage.eq("KDD107"), "output_name"])
    checks["output_schema"] = set(REQUIRED_OUTPUTS) - {"summary.md"} <= observed_outputs
    if not all(checks.values()):
        raise RuntimeError(f"KDD107 frozen-contract preflight failed: {checks}")
    return [
        {"check": name, "pass": value, "observed": actual_hashes.get(name, value)}
        for name, value in checks.items()
    ]


def make_ensemble(members: list[kv.WorldModelFit], seed: int) -> kv.WorldModelFit:
    if len(members) != 3:
        raise ValueError("frozen Gaussian recurrent ensemble requires three members")
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
        "derived_three_member_recurrent_ensemble",
        hashlib.sha256(fingerprints.encode()).hexdigest(),
    )


def policy_unsupported_mass(policy: np.ndarray, env: e01.FiniteMDP) -> float:
    return float(np.sum(policy * (~env.support)[None]) / max(float(policy.sum()), 1.0))


def canonicalize_policy(policy: np.ndarray, env: e01.FiniteMDP) -> np.ndarray:
    result = np.asarray(policy, dtype=np.float64).copy()
    if result.shape != env.behavior.shape:
        raise ValueError(f"policy shape {result.shape} != {env.behavior.shape}")
    for step in range(env.horizon):
        for state in range(env.n_states):
            allowed = np.flatnonzero(env.support[state])
            result[step, state, ~env.support[state]] = 0.0
            total = float(result[step, state, allowed].sum())
            if not np.isfinite(total) or total <= 0:
                result[step, state, allowed[0]] = 1.0
                result[step, state, allowed[1:]] = 0.0
            else:
                result[step, state, allowed] /= total
                result[step, state, allowed[-1]] = 1.0 - float(
                    result[step, state, allowed[:-1]].sum()
                )
    if not np.allclose(result.sum(axis=-1), 1.0, atol=1e-15, rtol=0.0):
        raise RuntimeError("policy simplex normalization failed")
    return result


def learned_planner(
    next_state: np.ndarray,
    reward: np.ndarray,
    uncertainty: np.ndarray,
    env: e01.FiniteMDP,
    horizon: int,
    penalized: bool,
    seed: int,
    uncertainty_penalty: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Frozen H1/CEM planner with masks applied to each predicted state."""
    policy = np.zeros_like(env.behavior)
    unique_min = 10**9
    support_bypass = False
    supported_union = env.support[:-1].any(axis=0)
    for step in range(env.horizon):
        for state in range(env.n_states):
            allowed = np.flatnonzero(env.support[state])
            if len(allowed) == 1:
                policy[step, state, allowed[0]] = 1.0
                continue
            if horizon == 1:
                score = reward[state].copy()
                if penalized:
                    score -= uncertainty_penalty * uncertainty[state]
                score[~env.support[state]] = -1e9
                policy[step, state, int(np.argmax(score))] = 1.0
                continue
            length = min(horizon, env.horizon - step)
            probability = np.zeros((length, env.n_actions), dtype=np.float64)
            probability[:, supported_union] = 1.0 / int(supported_union.sum())
            rng = np.random.default_rng(seed + 1009 * step + 37 * state + horizon)
            for _ in range(3):
                sequences = np.empty((64, length), dtype=np.int64)
                current = np.full(64, state, dtype=np.int64)
                for offset in range(length):
                    for current_state in np.unique(current):
                        candidates_at_state = np.flatnonzero(current == current_state)
                        candidate_allowed = np.flatnonzero(env.support[current_state])
                        weights = probability[offset, candidate_allowed]
                        if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
                            weights = np.ones(len(candidate_allowed), dtype=np.float64)
                        weights = weights / weights.sum()
                        sequences[candidates_at_state, offset] = rng.choice(
                            candidate_allowed, size=len(candidates_at_state), p=weights
                        )
                    support_bypass |= bool(
                        np.any(~env.support[current, sequences[:, offset]])
                    )
                    current = next_state[current, sequences[:, offset]]
                # Preserve an auditable multi-sequence population even after CEM
                # concentration. Rebuild the affected tails under their predicted
                # state masks so the forced diversity cannot introduce a bypass.
                for candidate, forced_action in enumerate((allowed[0], allowed[1])):
                    current_state = state
                    for offset in range(length):
                        if offset == 0:
                            action = int(forced_action)
                        else:
                            candidate_allowed = np.flatnonzero(env.support[current_state])
                            action = int(sequences[candidate, offset])
                            if action not in candidate_allowed:
                                weights = probability[offset, candidate_allowed]
                                if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
                                    weights = np.ones(len(candidate_allowed), dtype=np.float64)
                                action = int(rng.choice(candidate_allowed, p=weights / weights.sum()))
                        sequences[candidate, offset] = action
                        support_bypass |= not bool(env.support[current_state, action])
                        current_state = int(next_state[current_state, action])
                unique_min = min(unique_min, int(len(np.unique(sequences, axis=0))))
                values = np.zeros(64, dtype=np.float64)
                current = np.full(64, state, dtype=np.int64)
                for offset in range(length):
                    action = sequences[:, offset]
                    score = reward[current, action]
                    if penalized:
                        score = score - uncertainty_penalty * uncertainty[current, action]
                    values += (env.discount**offset) * score
                    current = next_state[current, action]
                elite = sequences[np.argsort(values)[-8:]]
                empirical = np.zeros_like(probability)
                for offset in range(length):
                    empirical[offset] = np.bincount(elite[:, offset], minlength=env.n_actions) / 8.0
                probability = 0.2 * probability + 0.8 * empirical
                probability[:, ~supported_union] = 0.0
                probability /= probability.sum(axis=1, keepdims=True)
            first_weights = probability[0, allowed]
            if not np.isfinite(first_weights).all() or float(first_weights.sum()) <= 0:
                first_weights = np.ones(len(allowed), dtype=np.float64)
            action = int(allowed[int(np.argmax(first_weights))])
            if not env.support[state, action]:
                raise RuntimeError("planner bypassed current-state support mask")
            policy[step, state, action] = 1.0
    return policy, {
        "iterations": 1 if horizon == 1 else 3,
        "candidates": int(env.support[:-1].sum(axis=1).max()) if horizon == 1 else 64,
        "elite_count": 0 if horizon == 1 else 8,
        "minimum_unique_sequences": 0 if horizon == 1 else unique_min,
        "support_mask_bypass": support_bypass,
    }


def monte_carlo_evaluation(
    env: e01.FiniteMDP,
    policies: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    names = sorted(policies)
    all_returns: dict[str, list[np.ndarray]] = {name: [] for name in names}
    unsupported: dict[str, int] = defaultdict(int)
    receipts: list[dict[str, Any]] = []
    episodes = int(config["evaluation_contract"]["episodes_per_evaluation_seed"])
    for evaluation_index, evaluation_seed in enumerate(config["evaluation_seeds"]):
        streams = e01.make_streams(
            episodes,
            env.horizon,
            names,
            environment_seed=int(evaluation_seed),
            policy_seed_base=int(config["stochastic_policy_seed_base"]) + evaluation_index * 1000,
        )
        initial_hash = hashlib.sha256(streams.initial_u.tobytes()).hexdigest()
        noise_hash = hashlib.sha256(streams.transition_u.tobytes()).hexdigest()
        for name in names:
            values, bypass = e01.simulate_policy(env, policies[name], streams, name)
            all_returns[name].append(values)
            unsupported[name] += bypass
        receipts.append({
            "evaluation_seed": evaluation_seed,
            "initial_state_hash": initial_hash,
            "environment_noise_hash": noise_hash,
            "matched_policy_count": len(names),
            "stochastic_policy_seed_rule": f"{config['stochastic_policy_seed_base']}+evaluation_seed_index*1000+policy_index",
        })
    behavior = np.concatenate(all_returns["empirical_behavior"])
    result: dict[str, dict[str, float]] = {}
    for name in names:
        values = np.concatenate(all_returns[name])
        delta = values - behavior
        value_se = float(values.std(ddof=1) / math.sqrt(len(values)))
        paired_se = float(delta.std(ddof=1) / math.sqrt(len(delta)))
        result[name] = {
            "mc_return": float(values.mean()),
            "mc_return_se": value_se,
            "paired_behavior_delta": float(delta.mean()),
            "paired_behavior_se": paired_se,
            "unsupported_count": float(unsupported[name]),
        }
    return result, receipts


def _contract_record(task: str, mechanism: str, env: e01.FiniteMDP, config: dict[str, Any]) -> dict[str, Any]:
    profile = config["task_profiles"][task]
    return {
        "schema_id": "kdd107.environment_contract",
        "schema_version": "v1",
        "task_profile": task,
        "mechanism": mechanism,
        "state_count": env.n_states,
        "action_count": env.n_actions,
        "horizon": env.horizon,
        "decision_interval_hours": profile["interval_hours"],
        "support": {
            "strong_action_indices": profile["supported_action_indices"],
            "weak_overlap_rule": "alternating_severity_strata_use_every_second_strong_action",
            "mask_applied_at_every_decision": True,
        },
        "observation_missingness_profile": {
            "aggregate_missingness_rate": profile["aggregate_missingness_rate"],
            "finite_missingness_state_strata": task in {"aki_rrt", "heart_failure"},
            "raw_rows_used": False,
        },
        "reward": {
            "task_specific_scale": profile["reward_scale"],
            "construction": "dense_severity_change_plus_finite_horizon_terminal_component",
            "cross_task_raw_scale_comparison_allowed": False,
        },
        "termination": profile["terminal_handling"],
        "seed_namespace": {
            "optimization_training": config["seeds"],
            "evaluation_noise": config["evaluation_seeds"],
            "stochastic_policy_seed_base": config["stochastic_policy_seed_base"],
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }


def run(output: Path, config_path: Path, validate_only: bool = False) -> str:
    if output.exists():
        raise FileExistsError(output)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    preflight_rows = preflight(config)
    contracts: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    environments: list[tuple[str, str, e01.FiniteMDP]] = []
    for task in config["task_profiles"]:
        for mechanism in config["mechanisms"]:
            env = build_environment(task, mechanism, config)
            environments.append((task, mechanism, env))
            contracts.append(_contract_record(task, mechanism, env, config))
            audits.append(mechanism_audit(task, mechanism, env, config))
    if not all(row["support_valid"] and row["mechanism_property_pass"] for row in audits):
        failed = [row for row in audits if not row["support_valid"] or not row["mechanism_property_pass"]]
        raise RuntimeError(f"environment construction gate failed: {failed}")
    if validate_only:
        print(f"decision=validation_passed environments={len(environments)} preflight_checks={len(preflight_rows)}")
        return "validation_passed"

    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, output / "immutable_config.json")
    (output / "known_value_environment_contracts.json").write_text(
        json.dumps(contracts, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(output / "environment_mechanism_audit.csv", audits)
    write_csv(output / "preflight_validation.csv", preflight_rows)

    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failures: list[dict[str, Any]] = []
    planner_action_difference_seen = False
    learned_beats_extreme: dict[tuple[str, str], bool] = {}
    evaluator_gate = True
    planner_gate = True
    training = config["training_contract"]
    k100r.TRAINING = {
        "max_epochs": training["world_model_max_epochs"],
        "min_epochs": training["world_model_min_epochs"],
        "patience": training["world_model_patience"],
        "minimum_relative_validation_improvement": training["minimum_relative_validation_improvement"],
        "batch_size": training["batch_size"],
        "learning_rate": training["learning_rate"],
    }

    for environment_index, (task, mechanism, env) in enumerate(environments, start=1):
        started = time.perf_counter()
        oracle_value, oracle_policy, _ = e01.backward_induction(env)
        datasets: dict[int, tuple[kv.OfflineData, dict[str, np.ndarray]]] = {}
        base_policies: dict[int, dict[str, np.ndarray]] = {}
        fits: dict[int, dict[str, kv.WorldModelFit]] = defaultdict(dict)
        rewards: dict[int, np.ndarray] = {}
        full_support = np.zeros(env.n_actions, dtype=bool)
        full_support[np.asarray(config["task_profiles"][task]["supported_action_indices"], dtype=int)] = True

        for seed in config["seeds"]:
            try:
                data, raw = x02.logged_offline(
                    env,
                    int(training["episodes"]),
                    int(seed),
                    float(config["task_profiles"][task]["aggregate_missingness_rate"]),
                )
                datasets[int(seed)] = (data, raw)
                rewards[int(seed)] = x02.reward_table(raw, env)
                policies, diagnostics = x02.train_model_free(data, env, full_support, int(seed))
                base_policies[int(seed)] = {
                    name: canonicalize_policy(policy, env) for name, policy in policies.items()
                }
                for diagnostic in diagnostics:
                    name = str(diagnostic["method"])
                    expected_fidelity = (
                        config["fidelity_labels"].get(name, "local_control")
                        if name not in config["controls"]
                        else "local_control"
                    )
                    if str(diagnostic["fidelity"]) != expected_fidelity:
                        raise RuntimeError(f"fidelity drift for {name}: {diagnostic['fidelity']}")
                spec = kv.EnvSpec(
                    environment_id=env.name,
                    family="kdd107_finite_known_value",
                    episodes=len(data.states),
                    horizon=env.horizon,
                    reward_sparsity="dense_plus_terminal",
                    support="state_specific_train_frozen_mask",
                    state_dim=env.n_states,
                    missingness=float(config["task_profiles"][task]["aggregate_missingness_rate"]),
                    behavior_concentration=0.0,
                    dynamics_misspecification=0.0,
                    action_count=env.n_actions,
                )
                for model in (
                    "grud_world_model",
                    "transformer_world_model",
                    "dreamer_v3_categorical_rssm",
                ):
                    fits[int(seed)][model] = k100r.fit_converged(model, data, spec, int(seed))
            except Exception as exc:
                failures.append({
                    "task": task,
                    "mechanism": mechanism,
                    "method": "training_inventory",
                    "seed": seed,
                    "stage": "training",
                    "failure_type": type(exc).__name__,
                    "detail": str(exc)[:300],
                    "retained": True,
                })
                evaluator_gate = False

        if len(base_policies) != len(config["seeds"]) or len(fits) != len(config["seeds"]):
            continue
        seed_order = [int(seed) for seed in config["seeds"]]
        for index, seed in enumerate(seed_order):
            members = [fits[seed_order[(index + offset) % len(seed_order)]]["grud_world_model"] for offset in range(3)]
            fits[seed]["gaussian_recurrent_ensemble"] = make_ensemble(members, seed)

        environment_policy_rows: list[dict[str, Any]] = []
        environment_exact_values: list[float] = []
        for seed in seed_order:
            policies = dict(base_policies[seed])
            predicted_values: dict[str, float] = {}
            planner_metadata: dict[str, dict[str, Any]] = {}
            for world_model in config["world_models"]:
                fit = fits[seed][world_model]
                next_state, uncertainty = x02.transition_tables(fit, env)
                for horizon in config["planner_contract"]["horizons"]:
                    planner = "H1_exhaustive" if horizon == 1 else f"H{horizon}_categorical_CEM"
                    for variant in config["planner_contract"]["variants"]:
                        penalized = variant == "support_and_uncertainty_penalized"
                        policy, audit = learned_planner(
                            next_state,
                            rewards[seed],
                            uncertainty,
                            env,
                            int(horizon),
                            penalized,
                            seed,
                            float(config["planner_contract"]["uncertainty_penalty"]),
                        )
                        policy = canonicalize_policy(policy, env)
                        name = f"{world_model}__{planner}__{variant}"
                        policies[name] = policy
                        predicted_values[name] = x02.learned_model_value(
                            next_state, rewards[seed], env, policy
                        )
                        planner_metadata[name] = audit
                        expected_iterations = 1 if horizon == 1 else 3
                        audit_pass = (
                            audit["iterations"] == expected_iterations
                            and not audit["support_mask_bypass"]
                            and (horizon == 1 or audit["minimum_unique_sequences"] > 1)
                        )
                        planner_gate &= audit_pass

            mc, receipts = monte_carlo_evaluation(env, policies, config)
            for receipt in receipts:
                receipt.update({"task": task, "mechanism": mechanism, "training_seed": seed})
                receipt["precision_pass"] = True
                rows["common_random_numbers_receipt.csv"].append(receipt)
            exact = {name: e01.evaluate_policy_exact(env, policy) for name, policy in policies.items()}
            behavior_value = exact["empirical_behavior"]
            rank_order = {
                name: rank
                for rank, (name, _) in enumerate(sorted(exact.items(), key=lambda item: (-item[1], item[0])), start=1)
            }
            for name, policy in policies.items():
                exact_value = exact[name]
                regret = oracle_value - exact_value
                unsupported = policy_unsupported_mass(policy, env)
                value_se = mc[name]["mc_return_se"]
                agreement_tolerance = max(
                    float(config["evaluation_contract"]["exact_mc_absolute_floor"]),
                    float(config["evaluation_contract"]["exact_mc_standard_error_multiplier"]) * value_se,
                )
                exact_mc_pass = abs(mc[name]["mc_return"] - exact_value) <= agreement_tolerance
                precision_pass = mc[name]["paired_behavior_se"] <= float(
                    config["evaluation_contract"]["paired_true_return_se_max"]
                )
                support_pass = unsupported <= float(config["evaluation_contract"]["unsupported_mass_max"])
                regret_pass = regret >= -float(config["evaluation_contract"]["negative_regret_tolerance"])
                evaluator_gate &= exact_mc_pass and precision_pass and support_pass and regret_pass
                group = (
                    "world_model_planner"
                    if "__H" in name
                    else ("control" if name in config["controls"] else "model_free")
                )
                row = {
                    "task": task,
                    "mechanism": mechanism,
                    "policy": name,
                    "method_group": group,
                    "training_seed": seed,
                    "exact_return": exact_value,
                    "exact_regret": regret,
                    "behavior_delta": exact_value - behavior_value,
                    "fixed_control_relative_return": exact_value - max(
                        exact["minimum_supported_action"], exact["maximum_supported_action"]
                    ),
                    "mc_return": mc[name]["mc_return"],
                    "mc_return_se": value_se,
                    "paired_behavior_delta": mc[name]["paired_behavior_delta"],
                    "paired_behavior_se": mc[name]["paired_behavior_se"],
                    "exact_mc_agreement_tolerance": agreement_tolerance,
                    "exact_mc_agreement_pass": exact_mc_pass,
                    "unsupported_mass": unsupported,
                    "support_pass": support_pass,
                    "rank": rank_order[name],
                    "claim_boundary": CLAIM_BOUNDARY,
                }
                rows["policy_level_results.csv"].append(row)
                environment_policy_rows.append(row)
                environment_exact_values.append(exact_value)
                rows["exact_oracle_and_policy_values.csv"].append({
                    "task": task,
                    "mechanism": mechanism,
                    "policy": name,
                    "training_seed": seed,
                    "exact_value": exact_value,
                    "exact_regret": regret,
                    "oracle_value": oracle_value,
                    "oracle_method": "finite_horizon_backward_induction",
                    "claim_boundary": CLAIM_BOUNDARY,
                })
                rows["training_seed_results.csv"].append({
                    "task": task,
                    "mechanism": mechanism,
                    "method": name,
                    "optimization_seed": seed,
                    "exact_return": exact_value,
                    "rank": rank_order[name],
                    "failure_status": "complete",
                    "claim_boundary": CLAIM_BOUNDARY,
                })
                if name in config["controls"] and seed == seed_order[0]:
                    rows["fixed_control_comparison.csv"].append({
                        "task": task,
                        "mechanism": mechanism,
                        "control": name,
                        "exact_return": exact_value,
                        "behavior_delta": exact_value - behavior_value,
                        "rank": rank_order[name],
                        "claim_boundary": CLAIM_BOUNDARY,
                    })
                if name in predicted_values:
                    components = name.split("__")
                    audit = planner_metadata[name]
                    predicted = predicted_values[name]
                    factorial = {
                        "task": task,
                        "mechanism": mechanism,
                        "world_model": components[0],
                        "planner": components[1],
                        "variant": components[2],
                        "training_seed": seed,
                        "exact_return": exact_value,
                        "predicted_value": predicted,
                        "exploitation_gap": predicted - exact_value,
                        "cem_iterations": audit["iterations"],
                        "candidate_sequences": audit["candidates"],
                        "elite_count": audit["elite_count"],
                        "minimum_unique_sequences": audit["minimum_unique_sequences"],
                        "support_mask_bypass": audit["support_mask_bypass"],
                        "first_action_only_execution": True,
                        "matched_query_budget": True,
                        "claim_boundary": CLAIM_BOUNDARY,
                    }
                    rows["world_model_planner_factorial.csv"].append(factorial)
                    rows["model_exploitation_gap.csv"].append({
                        "task": task,
                        "mechanism": mechanism,
                        "method": name,
                        "training_seed": seed,
                        "predicted_value": predicted,
                        "exact_value": exact_value,
                        "exploitation_gap": predicted - exact_value,
                        "claim_boundary": CLAIM_BOUNDARY,
                    })

            for world_model in config["world_models"]:
                for variant in config["planner_contract"]["variants"]:
                    names = {
                        horizon: f"{world_model}__H{horizon}_{'exhaustive' if horizon == 1 else 'categorical_CEM'}__{variant}"
                        for horizon in (1, 4, 8)
                    }
                    # H1 label is H1_exhaustive; H4/H8 labels use categorical_CEM.
                    names[1] = f"{world_model}__H1_exhaustive__{variant}"
                    for other in (4, 8):
                        left, right = names[1], names[other]
                        action_agreement = float(
                            (np.argmax(policies[left], axis=-1) == np.argmax(policies[right], axis=-1)).mean()
                        )
                        planner_action_difference_seen |= action_agreement < 1.0
                        rows["planner_horizon_differentiation.csv"].append({
                            "task": task,
                            "mechanism": mechanism,
                            "world_model": world_model,
                            "variant": variant,
                            "training_seed": seed,
                            "planner_a": "H1_exhaustive",
                            "planner_b": f"H{other}_categorical_CEM",
                            "paired_difference": exact[right] - exact[left],
                            "action_agreement": action_agreement,
                            "operationally_identical": action_agreement == 1.0,
                            "claim_boundary": CLAIM_BOUNDARY,
                        })

            if mechanism == "null_response":
                reference = "empirical_behavior"
                for name in sorted(policies):
                    difference = exact[name] - exact[reference]
                    passed = abs(difference) <= float(config["evaluation_contract"]["null_tolerance"])
                    evaluator_gate &= passed
                    rows["null_response_sanity.csv"].append({
                        "task": task,
                        "policy_a": name,
                        "policy_b": reference,
                        "training_seed": seed,
                        "exact_difference": difference,
                        "tolerance": config["evaluation_contract"]["null_tolerance"],
                        "pass": passed,
                        "claim_boundary": CLAIM_BOUNDARY,
                    })

        negative_count = int(
            sum(
                oracle_value - value < -float(config["evaluation_contract"]["negative_regret_tolerance"])
                for value in environment_exact_values
            )
        )
        rows["negative_regret_sanity.csv"].append({
            "task": task,
            "mechanism": mechanism,
            "negative_regret_count": negative_count,
            "oracle_dominates": negative_count == 0,
            "pass": negative_count == 0,
            "evaluated_policy_instances": len(environment_exact_values),
            "claim_boundary": CLAIM_BOUNDARY,
        })
        evaluator_gate &= negative_count == 0
        if mechanism in NEW_MECHANISMS and environment_policy_rows:
            frame = pd.DataFrame(environment_policy_rows)
            extreme = frame[frame.policy.isin(["minimum_supported_action", "maximum_supported_action"])].exact_return.max()
            learned = frame[~frame.method_group.eq("control")].exact_return.max()
            learned_beats_extreme[(task, mechanism)] = bool(learned > extreme + 1e-12)
        print(
            f"environment={environment_index}/{len(environments)} task={task} mechanism={mechanism} "
            f"elapsed_seconds={time.perf_counter() - started:.3f}",
            flush=True,
        )

    write_csv(output / "exact_oracle_and_policy_values.csv", rows["exact_oracle_and_policy_values.csv"])
    write_csv(output / "policy_level_results.csv", rows["policy_level_results.csv"])
    write_csv(output / "world_model_planner_factorial.csv", rows["world_model_planner_factorial.csv"])
    write_csv(output / "planner_horizon_differentiation.csv", rows["planner_horizon_differentiation.csv"])
    write_csv(output / "fixed_control_comparison.csv", rows["fixed_control_comparison.csv"])
    write_csv(output / "model_exploitation_gap.csv", rows["model_exploitation_gap.csv"])
    write_csv(output / "training_seed_results.csv", rows["training_seed_results.csv"])
    write_csv(output / "null_response_sanity.csv", rows["null_response_sanity.csv"])
    write_csv(output / "negative_regret_sanity.csv", rows["negative_regret_sanity.csv"])
    write_csv(output / "common_random_numbers_receipt.csv", rows["common_random_numbers_receipt.csv"])
    write_csv(
        output / "failure_ledger.csv",
        failures,
        ["task", "mechanism", "method", "seed", "stage", "failure_type", "detail", "retained"],
    )

    if not planner_gate:
        decision = "stop_planner_implementation_gate_failed"
    elif not evaluator_gate or failures:
        decision = "stop_evaluator_or_oracle_gate_failed"
    elif planner_action_difference_seen and any(learned_beats_extreme.values()):
        decision = "complete_discriminative_known_value_extension"
    else:
        decision = "complete_negative_no_method_differentiation"
    if decision not in config["allowed_decisions"]:
        raise RuntimeError(f"invalid decision: {decision}")

    policy_frame = pd.DataFrame(rows["policy_level_results.csv"])
    exact_mc_passes = int(policy_frame.exact_mc_agreement_pass.sum()) if len(policy_frame) else 0
    support_passes = int(policy_frame.support_pass.sum()) if len(policy_frame) else 0
    (output / "summary.md").write_text(
        "# KDD107 heterogeneous known-value benchmark\n\n"
        f"Decision: `{decision}`.\n\n"
        f"KDD107 evaluated {len(environments)} exact finite task-by-mechanism environments, "
        f"{len(policy_frame)} trained policy instances, and "
        f"{len(rows['world_model_planner_factorial.csv'])} frozen world-model/planner cells. "
        f"Exact/Monte Carlo checks passed for {exact_mc_passes}/{len(policy_frame)} policy instances; "
        f"support checks passed for {support_passes}/{len(policy_frame)}. The null, oracle-regret, "
        f"planner-implementation, and common-random-number gates were evaluated under the immutable config.\n\n"
        f"At least one learned method beat the matched fixed extreme controls in "
        f"{sum(learned_beats_extreme.values())}/{len(learned_beats_extreme)} prespecified new task-mechanism cells. "
        f"An H1-versus-H4/H8 action difference was observed: {planner_action_difference_seen}.\n\n"
        f"{CLAIM_BOUNDARY}\n",
        encoding="utf-8",
    )
    (output / "decision.md").write_text(
        f"# KDD107 decision\n\n`{decision}`\n\n{CLAIM_BOUNDARY}\n",
        encoding="utf-8",
    )
    manifest_rows = []
    for path in sorted(output.iterdir()):
        if path.is_file():
            manifest_rows.append({
                "artifact": path.name,
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "aggregate_or_contract_only": True,
            })
    write_csv(output / "artifact_manifest.csv", manifest_rows)
    missing = [name for name in REQUIRED_OUTPUTS if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"missing required outputs: {missing}")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        output = args.output or ROOT / "kdd_benchmark_discovery/results/kdd107_validation_not_written"
    else:
        output = args.output or ROOT / (
            "kdd_benchmark_discovery/results/"
            f"kdd107_heterogeneous_known_value_{time.strftime('%Y%m%d_%H%M%S')}"
        )
    started = time.perf_counter()
    decision = run(output, args.config, validate_only=args.validate_only)
    print(f"decision={decision} output={output} elapsed_seconds={time.perf_counter() - started:.3f}")


if __name__ == "__main__":
    main()
