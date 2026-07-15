from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from kdd_benchmark_discovery import kdd_e01_evaluator as e01
from kdd_benchmark_discovery import run_kdd100_complete_known_value as kv
from kdd_benchmark_discovery import run_kdd100r_task_matched_known_value as k100r
from kdd_benchmark_discovery import run_kdd_x02_cross_cohort_policy_benchmark as x02


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/kdd_adapt01_adaptive_known_value_v1.json"
CLAIM = (
    "Exact finite known-construction technical benchmark only. Task profiles use aggregate EHR-derived "
    "dimensions and support, but action response is injected and is not a treatment benefit, causal effect, "
    "counterfactual claim, clinical utility, policy superiority, deployment, or autonomous-decision result."
)
REQUIRED = (
    "adaptive_environment_contract.csv",
    "ehr_to_known_value_contract_matrix.csv",
    "optimal_action_state_map.csv",
    "adaptive_vs_best_fixed_gap.csv",
    "adaptive_policy_true_returns.csv",
    "adaptive_policy_regret.csv",
    "adaptive_world_model_planner_matrix.csv",
    "adaptive_exploitation_gap.csv",
    "adaptive_environment_preflight_receipt.md",
)


@dataclass(frozen=True)
class Layout:
    severity_levels: int
    subtypes: int = 2
    momentum_levels: int = 3
    observation_levels: int = 2

    @property
    def nonterminal_states(self) -> int:
        return self.severity_levels * self.subtypes * self.momentum_levels * self.observation_levels

    @property
    def terminal(self) -> int:
        return self.nonterminal_states

    @property
    def states(self) -> int:
        return self.nonterminal_states + 1

    def encode(self, severity: int, subtype: int, momentum: int, observed: int) -> int:
        return (((severity * self.subtypes + subtype) * self.momentum_levels + momentum)
                * self.observation_levels + observed)

    def decode(self, state: int) -> tuple[int, int, int, int]:
        observed = state % self.observation_levels
        value = state // self.observation_levels
        momentum = value % self.momentum_levels
        value //= self.momentum_levels
        subtype = value % self.subtypes
        severity = value // self.subtypes
        return severity, subtype, momentum, observed


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"empty required output: {path.name}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verify_sources(config: dict[str, Any]) -> None:
    paths = {
        "kdd097_action_dictionary": ROOT / "results/kdd097_rich_task_materialization_20260714_v2/action_dictionary.csv",
        "kdd_x02_config": ROOT / "configs/kdd_x02_cross_cohort_policy_benchmark_v1.json",
        "kdd_x09_config": ROOT / "configs/kdd_x09_promoted_cohort_policy_benchmark_v1.json",
        "e01_evaluator": ROOT / "kdd_benchmark_discovery/kdd_e01_evaluator.py",
        "x02_runner": ROOT / "kdd_benchmark_discovery/run_kdd_x02_cross_cohort_policy_benchmark.py",
        "x09_runner": ROOT / "kdd_benchmark_discovery/run_kdd_x09_promoted_cohort_policy_benchmark.py",
    }
    actual = {key: _sha256(path) for key, path in paths.items()}
    if actual != config["immutable_source_hashes"]:
        raise RuntimeError(f"immutable source drift: {actual}")


def action_intensity(task: str, action_count: int) -> np.ndarray:
    if action_count == 2:
        return np.array([0.0, 1.0], dtype=np.float64)
    if action_count == 8:
        values = []
        for action in range(8):
            diuretic, vaso = divmod(action, 2)
            values.append(np.clip(diuretic / 3.0 + 0.08 * vaso, 0.0, 1.0))
        return np.asarray(values)
    if action_count == 25:
        return np.asarray([sum(divmod(action, 5)) / 8.0 for action in range(25)], dtype=np.float64)
    raise ValueError((task, action_count))


def observation_features(layout: Layout) -> np.ndarray:
    features = np.eye(layout.states, dtype=np.float32)
    for state in range(layout.nonterminal_states):
        severity, subtype, momentum, observed = layout.decode(state)
        if observed == 0:
            alias = layout.encode(severity, 0, momentum, 0)
            features[state] = 0.0
            features[state, alias] = 1.0
    return features


def build_environment(task: str, profile: dict[str, Any], mechanism: str,
                      mechanism_cfg: dict[str, Any]) -> tuple[e01.FiniteMDP, Layout, np.ndarray]:
    layout = Layout(int(profile["severity_levels"]))
    horizon = int(profile["horizon"])
    actions = int(profile["action_count"])
    supported = np.zeros(actions, dtype=bool)
    supported[np.asarray(profile["supported_actions"], dtype=int)] = True
    support = np.broadcast_to(supported, (layout.states, actions)).copy()
    support[layout.terminal] = False
    support[layout.terminal, int(np.flatnonzero(supported)[0])] = True
    transition = np.zeros((horizon, layout.states, actions, layout.states), dtype=np.float64)
    reward = np.zeros_like(transition)
    intensity = action_intensity(task, actions)
    cfg = mechanism_cfg["adaptive_composite"]
    missingness = float(profile["missingness"])
    terminal_total = float(profile["termination_prevalence"])
    terminal_hazard = 1.0 - (1.0 - terminal_total) ** (1.0 / horizon)
    scale = float(profile["reward_scale"])

    for t in range(horizon):
        for state in range(layout.states):
            for action in range(actions):
                if state == layout.terminal:
                    transition[t, state, action, layout.terminal] = 1.0
                    continue
                severity, subtype, momentum, _ = layout.decode(state)
                severity_norm = severity / max(layout.severity_levels - 1, 1)
                momentum_intensity = momentum / max(layout.momentum_levels - 1, 1)
                target = np.clip(
                    0.12 + 0.72 * severity_norm
                    + float(cfg["subtype_target_shift"]) * (1.0 if subtype else -0.5),
                    float(cfg["intermediate_target_floor"]),
                    float(cfg["intermediate_target_ceiling"]),
                )
                delayed_match = 1.0 - abs(momentum_intensity - target)
                improve = np.clip(0.10 + float(cfg["delayed_momentum_strength"]) * max(delayed_match - 0.45, 0.0), 0.03, 0.42)
                worsen = np.clip(0.10 + float(cfg["delayed_momentum_strength"]) * max(0.55 - delayed_match, 0.0), 0.03, 0.42)
                stay = 1.0 - improve - worsen - terminal_hazard
                if stay < 0:
                    raise RuntimeError("invalid transition probability")
                action_momentum = int(np.clip(round(float(intensity[action]) * 2.0), 0, 2))
                severity_rows = (
                    (max(0, severity - 1), improve),
                    (min(layout.severity_levels - 1, severity + 1), worsen),
                    (severity, stay),
                )
                for next_severity, mass in severity_rows:
                    for next_observed, obs_mass in ((0, missingness), (1, 1.0 - missingness)):
                        following = layout.encode(next_severity, subtype, action_momentum, next_observed)
                        transition[t, state, action, following] += mass * obs_mass
                transition[t, state, action, layout.terminal] += terminal_hazard

                mismatch = abs(float(intensity[action]) - target)
                toxicity = float(cfg["low_severity_high_action_toxicity"]) * intensity[action] ** 2 * (1.0 - severity_norm)
                insufficient = float(cfg["high_severity_low_action_penalty"]) * max(target - intensity[action], 0.0) * severity_norm
                switch = float(cfg["switch_cost"]) * abs(float(intensity[action]) - momentum_intensity)
                direct = -scale * (float(cfg["mismatch_penalty"]) * mismatch + toxicity + insufficient + switch)
                for following in range(layout.states):
                    if following == layout.terminal:
                        reward[t, state, action, following] = direct - 0.25 * scale
                        continue
                    next_severity = layout.decode(following)[0]
                    dense = scale * float(cfg["dense_improvement_weight"]) * (severity - next_severity)
                    terminal = scale * (1.0 - next_severity / max(layout.severity_levels - 1, 1)) if t == horizon - 1 else 0.0
                    reward[t, state, action, following] = direct + dense + terminal

    if mechanism == "null_response":
        reference = int(np.flatnonzero(supported)[0])
        transition[:] = transition[:, :, reference:reference + 1, :]
        reward[:] = reward[:, :, reference:reference + 1, :]
    elif mechanism != "adaptive_composite":
        raise ValueError(mechanism)

    initial = np.zeros(layout.states, dtype=np.float64)
    severity_weight = np.arange(1, layout.severity_levels + 1, dtype=np.float64)
    severity_weight = np.minimum(severity_weight, severity_weight[::-1] + 0.5)
    severity_weight /= severity_weight.sum()
    for severity in range(layout.severity_levels):
        for subtype in range(2):
            for observed, obs_mass in ((0, missingness), (1, 1.0 - missingness)):
                state = layout.encode(severity, subtype, 1, observed)
                initial[state] = severity_weight[severity] * 0.5 * obs_mass

    action_counts = np.asarray(profile["action_counts"], dtype=np.float64)
    action_counts[~supported] = 0.0
    action_counts += supported * 0.25
    behavior = np.zeros((horizon, layout.states, actions), dtype=np.float64)
    for t in range(horizon):
        for state in range(layout.states):
            if state == layout.terminal:
                behavior[t, state, int(np.flatnonzero(support[state])[0])] = 1.0
                continue
            severity = layout.decode(state)[0] / max(layout.severity_levels - 1, 1)
            weights = action_counts.copy()
            weights *= np.exp(0.35 * severity * intensity)
            weights[~support[state]] = 0.0
            behavior[t, state] = weights / weights.sum()

    env = e01.FiniteMDP(f"{task}_{mechanism}_v1", transition, reward, initial, support, behavior, 0.99)
    return env, layout, observation_features(layout)


def fixed_policy(env: e01.FiniteMDP, action: int) -> np.ndarray:
    policy = np.zeros_like(env.behavior)
    for t in range(env.horizon):
        for state in range(env.n_states):
            chosen = action if env.support[state, action] else int(np.flatnonzero(env.support[state])[0])
            policy[t, state, chosen] = 1.0
    return policy


def table_policy(probability: np.ndarray, env: e01.FiniteMDP) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float64)
    if probability.shape == (env.n_states, env.n_actions):
        probability = np.broadcast_to(probability[None], env.behavior.shape).copy()
    if probability.shape != env.behavior.shape:
        raise ValueError(probability.shape)
    probability[:, ~env.support] = 0.0
    total = probability.sum(axis=-1, keepdims=True)
    for t, state in np.argwhere(total[..., 0] <= 0):
        probability[t, state, int(np.flatnonzero(env.support[state])[0])] = 1.0
    probability /= probability.sum(axis=-1, keepdims=True)
    return probability


def severity_rule(env: e01.FiniteMDP, layout: Layout, intensity: np.ndarray) -> np.ndarray:
    supported = np.flatnonzero(env.support[:-1].all(axis=0))
    probability = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
    for state in range(env.n_states):
        if state == layout.terminal:
            action = int(np.flatnonzero(env.support[state])[0])
        else:
            severity = layout.decode(state)[0] / max(layout.severity_levels - 1, 1)
            action = int(supported[np.argmin(np.abs(intensity[supported] - severity))])
        probability[state, action] = 1.0
    return table_policy(probability, env)


def occupancy(env: e01.FiniteMDP, policy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distribution = env.initial.copy()
    state_mass = np.zeros(env.n_states, dtype=np.float64)
    action_mass = np.zeros(env.n_actions, dtype=np.float64)
    for t in range(env.horizon):
        state_mass += distribution
        action_mass += np.sum(distribution[:, None] * policy[t], axis=0)
        distribution = np.einsum("s,sa,san->n", distribution, policy[t], env.transition[t])
    total = max(float(action_mass.sum()), 1e-12)
    return state_mass / max(float(state_mass.sum()), 1e-12), action_mass / total


def preflight_task(task: str, profile: dict[str, Any], config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    adaptive, layout, _ = build_environment(task, profile, "adaptive_composite", config["mechanisms"])
    null, _, _ = build_environment(task, profile, "null_response", config["mechanisms"])
    optimum, oracle, q = e01.backward_induction(adaptive)
    state_mass, action_mass = occupancy(adaptive, oracle)
    supported = np.flatnonzero(adaptive.support[:-1].all(axis=0))
    fixed_values = {int(action): e01.evaluate_policy_exact(adaptive, fixed_policy(adaptive, int(action))) for action in supported}
    best_fixed_action = max(fixed_values, key=fixed_values.get)
    best_fixed = fixed_values[best_fixed_action]
    gap = optimum - best_fixed
    optimal_actions = np.argmax(oracle, axis=-1)
    map_rows: list[dict[str, Any]] = []
    for state in range(layout.nonterminal_states):
        severity, subtype, momentum, observed = layout.decode(state)
        map_rows.append({
            "task": task,
            "state": state,
            "severity": severity,
            "response_subtype": subtype,
            "prior_action_momentum": momentum,
            "subtype_observed": bool(observed),
            "reachable_state_mass": state_mass[state],
            "optimal_action_t0": int(optimal_actions[0, state]),
            "optimal_action_intensity_t0": action_intensity(task, adaptive.n_actions)[int(optimal_actions[0, state])],
            "supported": True,
            "claim_boundary": CLAIM,
        })

    null_policies = {f"fixed_{action}": fixed_policy(null, int(action)) for action in supported}
    null_policies["behavior"] = null.behavior
    null_policies["random_supported"] = e01.support_aware_stochastic_policy(null)
    exact_null = [e01.evaluate_policy_exact(null, policy) for policy in null_policies.values()]
    streams = e01.make_streams(
        int(config["evaluation"]["maximum_episodes"]), null.horizon, list(null_policies),
        int(config["seeds"]["evaluation_environment"]), int(config["seeds"]["stochastic_policy_base"]),
    )
    null_mc = [e01.simulate_policy(null, policy, streams, name)[0] for name, policy in null_policies.items()]
    null_exact_spread = max(exact_null) - min(exact_null)
    null_mc_spread = max(float(values.mean()) for values in null_mc) - min(float(values.mean()) for values in null_mc)
    cem_audits = []
    for horizon in (4, 8):
        cem_policy, traces = e01.categorical_cem_policy(adaptive, horizon, seed=29401)
        active = [trace for trace in traces if adaptive.support[trace.state].sum() > 1]
        cem_audits.append(bool(active and {trace.iteration for trace in active} == {1, 2, 3}
                               and min(trace.unique_sequences for trace in active) > 1
                               and not np.any(cem_policy * (~adaptive.support)[None])))

    mass_threshold = float(config["evaluation"]["nontrivial_optimal_action_mass"])
    nontrivial = action_mass >= mass_threshold
    min_action, max_action = int(supported[0]), int(supported[-1])
    gates = {
        "null_exact_equal": null_exact_spread <= float(config["evaluation"]["null_exact_tolerance"]),
        "null_mc_equal": null_mc_spread <= float(config["evaluation"]["null_paired_mc_tolerance"]),
        "fixed_extremes_not_globally_optimal": action_mass[min_action] < 1.0 - mass_threshold and action_mass[max_action] < 1.0 - mass_threshold,
        "optimal_action_varies": int(nontrivial.sum()) >= 2,
        "k2_both_actions_nontrivial": adaptive.n_actions != 2 or bool(np.all(action_mass >= float(config["evaluation"]["k2_each_action_optimal_mass"]))),
        "adaptive_gap": gap >= float(config["evaluation"]["adaptive_vs_best_fixed_margin"]),
        "negative_oracle_regret_zero": optimum - e01.evaluate_policy_exact(adaptive, oracle) >= -float(config["evaluation"]["negative_regret_tolerance"]),
        "support_masks": bool(np.all(cem_audits)),
    }
    receipt = {
        "task": task,
        "oracle_value": optimum,
        "best_fixed_action": best_fixed_action,
        "best_fixed_value": best_fixed,
        "adaptive_minus_best_fixed": gap,
        "null_exact_spread": null_exact_spread,
        "null_mc_spread": null_mc_spread,
        "optimal_action_count_nontrivial_mass": int(nontrivial.sum()),
        "minimum_nonzero_optimal_action_mass": float(action_mass[action_mass > 0].min()),
        "all_gates_pass": bool(all(gates.values())),
        **gates,
    }
    gap_row = {
        "task": task,
        "exact_adaptive_value": optimum,
        "best_fixed_action": best_fixed_action,
        "best_fixed_exact_value": best_fixed,
        "paired_exact_gap": gap,
        "prespecified_margin": config["evaluation"]["adaptive_vs_best_fixed_margin"],
        "gate_pass": gates["adaptive_gap"],
        "claim_boundary": CLAIM,
    }
    return receipt, map_rows, gap_row


def logged_offline(env: e01.FiniteMDP, features: np.ndarray, episodes: int, seed: int,
                   missingness: float) -> tuple[kv.OfflineData, dict[str, np.ndarray]]:
    raw = e01.generate_logged_data(env, n=episodes, seed=seed)
    state = features[raw["states"][:, :-1]]
    following = features[raw["states"][:, 1:]]
    rng = np.random.default_rng(seed + 991)
    mask = rng.random(state.shape) >= missingness
    observed = np.where(mask, state, 0.0).astype(np.float32)
    delta = np.zeros_like(state)
    for t in range(1, env.horizon):
        delta[:, t] = np.where(mask[:, t - 1], 1.0, delta[:, t - 1] + 1.0)
    terminal = raw["states"][:, 1:] == env.n_states - 1
    behavior = env.behavior[np.arange(env.horizon)[None, :], raw["states"][:, :-1]]
    data = kv.OfflineData(
        state.astype(np.float32), observed, mask, delta, raw["actions"].astype(np.int16),
        raw["rewards"].astype(np.float32), following.astype(np.float32), terminal,
        behavior.astype(np.float32), state[:, 0].astype(np.float32),
    )
    return data, raw


def train_dt(data: kv.OfflineData, env: e01.FiniteMDP, support: np.ndarray, features: np.ndarray,
             seed: int, cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    torch.manual_seed(seed)
    x = data.observed
    rtg = np.zeros_like(data.rewards)
    running = np.zeros(len(x), dtype=np.float32)
    for t in range(env.horizon - 1, -1, -1):
        running = data.rewards[:, t] + 0.99 * running
        rtg[:, t] = running
    split = int(len(x) * 0.75)
    model = x02.DecisionTransformerAdapter(env.n_states, env.n_actions, 64, env.horizon)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    dataset = TensorDataset(
        torch.from_numpy(x[:split]), torch.from_numpy(rtg[:split]),
        torch.ones_like(torch.from_numpy(rtg[:split])), torch.from_numpy(data.actions[:split].astype(np.int64)),
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed))
    best, best_loss = None, math.inf
    for _ in range(8):
        model.train()
        for xb, rb, kb, ab in loader:
            logits = model(xb, rb, kb)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, env.n_actions), ab.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            logits = model(torch.from_numpy(x[split:]), torch.from_numpy(rtg[split:]), torch.ones_like(torch.from_numpy(rtg[split:])))
            logits[..., ~torch.from_numpy(support)] = -1e9
            value = float(torch.nn.functional.cross_entropy(logits.reshape(-1, env.n_actions), torch.from_numpy(data.actions[split:].astype(np.int64)).reshape(-1)))
        if value < best_loss:
            best_loss, best = value, copy.deepcopy(model.state_dict())
    if best is None:
        raise RuntimeError("Decision Transformer checkpoint unavailable")
    model.load_state_dict(best)
    desired = float(np.quantile(rtg[:split], 0.75))
    sequence = np.broadcast_to(features[:, None, :], (env.n_states, env.horizon, env.n_states)).copy()
    with torch.inference_mode():
        logits = model(
            torch.from_numpy(sequence), torch.full((env.n_states, env.horizon), desired),
            torch.ones((env.n_states, env.horizon)),
        ).numpy()
    logits[..., ~support] = -1e9
    logits -= logits.max(axis=-1, keepdims=True)
    probability = np.exp(logits)
    probability /= probability.sum(axis=-1, keepdims=True)
    return table_policy(probability.transpose(1, 0, 2), env), {"epochs": 8, "validation_nll": best_loss, "target_return": desired}


def train_model_free(data: kv.OfflineData, env: e01.FiniteMDP, support: np.ndarray,
                     features: np.ndarray, layout: Layout, task: str, seed: int) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    n = len(data.states)
    split = int(n * 0.75)
    xtr = data.observed[:split].reshape(-1, env.n_states)
    xval = data.observed[split:].reshape(-1, env.n_states)
    ytr = data.actions[:split].reshape(-1).astype(np.int64)
    yval = data.actions[split:].reshape(-1).astype(np.int64)
    rtr, rval = data.rewards[:split].reshape(-1), data.rewards[split:].reshape(-1)
    ntr = data.next_states[:split].reshape(-1, env.n_states)
    nval = data.next_states[split:].reshape(-1, env.n_states)
    dtr = data.terminal[:split].reshape(-1).astype(np.float32)
    dval = data.terminal[split:].reshape(-1).astype(np.float32)
    cfg = {"hidden_dim": 64, "learning_rate": 1e-3, "weight_decay": 1e-5, "batch_size": 256,
           "max_epochs": 12, "min_epochs": 4, "patience": 3, "discount": 0.99,
           "cql_alpha": 1.0, "bcq_behavior_threshold": 0.3, "soft_spibb_mix": 0.5}
    bc, meta = x02.fit_classifier(xtr, ytr, xval, yval, env.n_actions, support, cfg, seed)
    bc_state = x02.policy_probs(bc, features, support)
    policies = {"behavior_cloning": table_policy(bc_state, env)}
    diagnostics = [{"method": "behavior_cloning", **meta, "fidelity": "independent_reimplementation"}]
    behavior_state = x02.policy_probs(bc, features, support)
    for method in ("discrete_bcq", "discrete_cql", "soft_spibb"):
        q, q_meta = x02.fit_q(xtr, ytr, rtr, ntr, dtr, xval, yval, rval, nval, dval,
                              env.n_actions, support, cfg, seed, method)
        with torch.inference_mode():
            q_state = q(torch.from_numpy(features)).numpy()
        probability = x02.q_policy(q_state, behavior_state, support, method, cfg)
        policies[method] = table_policy(probability, env)
        diagnostics.append({"method": method, **q_meta,
                            "fidelity": "conceptual_adapter" if method == "soft_spibb" else "independent_reimplementation"})
    dt, dt_meta = train_dt(data, env, support, features, seed, cfg)
    policies["decision_transformer_adapter"] = dt
    diagnostics.append({"method": "decision_transformer_adapter", **dt_meta, "fidelity": "official_contract_adapter"})
    supported = np.flatnonzero(support)
    uniform = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
    uniform[:, supported] = 1.0 / len(supported)
    controls = {
        "empirical_behavior": env.behavior.copy(),
        "random_supported": table_policy(uniform, env),
        "minimum_supported_action": fixed_policy(env, int(supported[0])),
        "maximum_supported_action": fixed_policy(env, int(supported[-1])),
        "severity_rule": severity_rule(env, layout, action_intensity(task, env.n_actions)),
    }
    policies.update(controls)
    diagnostics.extend({"method": name, "epochs_run": 0, "fidelity": "local_control"} for name in controls)
    return policies, diagnostics


def transition_tables(fit: kv.WorldModelFit, env: e01.FiniteMDP, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spec = kv.EnvSpec(env.name, "adaptive_known_value", 1, env.horizon, "mixed_dense_terminal",
                      "train_frozen_mask", env.n_states, 0.0, 0.0, 0.0, env.n_actions)
    mean, scale = kv.predict_actions(fit, spec, features.astype(np.float32))
    distance = np.square(mean[:, :, None, :] - features[None, None, :, :]).sum(axis=-1)
    next_state = np.argmin(distance, axis=-1)
    uncertainty = np.mean(scale, axis=-1)
    return next_state, uncertainty


def make_ensemble(members: list[kv.WorldModelFit], seed: int) -> kv.WorldModelFit:
    fingerprint = hashlib.sha256(";".join(member.fingerprint for member in members).encode()).hexdigest()
    return kv.WorldModelFit(
        "gaussian_recurrent_ensemble", seed, tuple(member.model for member in members),
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
        "derived_rolling_three_of_five_recurrent_seed_ensemble", fingerprint,
    )


def evaluate_crn(env: e01.FiniteMDP, policies: dict[str, np.ndarray], config: dict[str, Any]) -> tuple[dict[str, tuple[float, float]], int, float]:
    names = list(policies)
    n = int(config["evaluation"]["initial_episodes"])
    maximum = int(config["evaluation"]["maximum_episodes"])
    tolerance = float(config["evaluation"]["paired_mc_se_tolerance"])
    while True:
        streams = e01.make_streams(n, env.horizon, names, int(config["seeds"]["evaluation_environment"]),
                                   int(config["seeds"]["stochastic_policy_base"]))
        returns = {name: e01.simulate_policy(env, policy, streams, name)[0] for name, policy in policies.items()}
        behavior = returns["empirical_behavior"]
        summary = {name: e01.paired_precision(value, behavior) for name, value in returns.items()}
        max_se = max(se for _, se in summary.values())
        if max_se <= tolerance or n >= maximum:
            return summary, n, max_se
        n = min(maximum, n * 2)


def run(config_path: Path, output: Path, preflight_only: bool) -> None:
    if output.exists():
        raise FileExistsError(output)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    verify_sources(config)
    receipts, map_rows, gap_rows = [], [], []
    for task, profile in config["tasks"].items():
        receipt, task_map, gap = preflight_task(task, profile, config)
        receipts.append(receipt)
        map_rows.extend(task_map)
        gap_rows.append(gap)
    if not all(row["all_gates_pass"] for row in receipts):
        raise RuntimeError(f"adaptive environment preflight failed: {receipts}")
    if preflight_only:
        print(json.dumps(receipts, indent=2, default=lambda value: value.item()))
        return

    output.mkdir(parents=True)
    contract_rows, matrix_rows = [], []
    policy_rows, regret_rows, planner_rows, exploitation_rows = [], [], [], []
    failures: list[dict[str, Any]] = []
    k100r.TRAINING = dict(config["training"])
    seed_list = [int(seed) for seed in config["seeds"]["optimization_training"]]

    for task_index, (task, profile) in enumerate(config["tasks"].items()):
        env, layout, features = build_environment(task, profile, "adaptive_composite", config["mechanisms"])
        oracle_value, oracle_policy, _ = e01.backward_induction(env)
        counts = np.asarray(profile["action_counts"], dtype=float)
        supported = np.asarray(profile["supported_actions"], dtype=int)
        concentration = float(counts.max() / max(counts.sum(), 1.0))
        contract_rows.append({
            "contract_id": f"{task}_adaptive_known_value_v1", "task": task,
            "mechanism": "adaptive_composite", "action_count": env.n_actions,
            "supported_action_count": len(supported), "decision_horizon": env.horizon,
            "state_count": env.n_states, "missingness_rate": profile["missingness"],
            "termination_prevalence": profile["termination_prevalence"],
            "behavior_top_action_share": concentration, "reward_scale": profile["reward_scale"],
            "reward_sparsity": profile["reward_sparsity"],
            "response_components": "low_state_harm;high_state_insufficient_action;intermediate_optimum;toxicity;delayed_benefit;subtype_heterogeneity;switch_cost;terminal_dense_tradeoff;partial_subtype_observation",
            "truth": "exact_dynamic_programming", "claim_boundary": CLAIM,
        })
        matrix_rows.extend([
            {"task": task, "surface": "historical_known_value", "role": "monotone_control_planning_smoke_test",
             "source": "KDD-X02" if task in {"respiratory", "shock"} else "KDD-X09",
             "rerun_or_overwrite": False, "claim_boundary": CLAIM},
            {"task": task, "surface": "adaptive_known_value_v1", "role": "state_dependent_exact_finite_benchmark",
             "source": f"{task}_adaptive_known_value_v1", "rerun_or_overwrite": False, "claim_boundary": CLAIM},
        ])

        task_policies: dict[int, dict[str, np.ndarray]] = {}
        task_fits: dict[int, dict[str, kv.WorldModelFit]] = {}
        task_raw: dict[int, dict[str, np.ndarray]] = {}
        for seed in seed_list:
            logged_seed = 7_401_000 + task_index * 10_000 + seed
            data, raw = logged_offline(
                env, features, int(config["training"]["training_episodes"]) + int(config["training"]["validation_episodes"]),
                logged_seed, float(profile["missingness"]),
            )
            task_raw[seed] = raw
            try:
                policies, _ = train_model_free(data, env, env.support[:-1].all(axis=0), features, layout, task, seed)
                task_policies[seed] = policies
            except Exception as exc:
                failures.append({"task": task, "component": "model_free", "seed": seed,
                                 "failure": type(exc).__name__, "detail": str(exc)[:200]})
                task_policies[seed] = {}
            fits: dict[str, kv.WorldModelFit] = {}
            spec = kv.EnvSpec(env.name, "adaptive_known_value", len(data.states), env.horizon,
                              profile["reward_sparsity"], "train_frozen_mask", env.n_states,
                              float(profile["missingness"]), concentration, 0.0, env.n_actions)
            for method in ("grud_world_model", "transformer_world_model", "dreamer_v3_categorical_rssm"):
                try:
                    fits[method] = k100r.fit_converged(method, data, spec, seed)
                except Exception as exc:
                    failures.append({"task": task, "component": method, "seed": seed,
                                     "failure": type(exc).__name__, "detail": str(exc)[:200]})
            task_fits[seed] = fits

        recurrent = {seed: fits["grud_world_model"] for seed, fits in task_fits.items() if "grud_world_model" in fits}
        if len(recurrent) == len(seed_list):
            for index, seed in enumerate(seed_list):
                member_seeds = [seed_list[(index + offset) % len(seed_list)] for offset in range(3)]
                task_fits[seed]["gaussian_recurrent_ensemble"] = make_ensemble([recurrent[item] for item in member_seeds], seed)

        for seed in seed_list:
            policies = task_policies[seed]
            raw = task_raw[seed]
            reward_table = x02.reward_table(raw, env)
            predicted_values: dict[str, float] = {}
            for model_name, fit in task_fits[seed].items():
                try:
                    next_state, uncertainty = transition_tables(fit, env, features)
                    for planning_horizon in config["planner_contract"]["horizons"]:
                        for penalized in (False, True):
                            variant = "support_and_uncertainty_penalized" if penalized else "support_constrained"
                            planner_label = "H1_exhaustive" if planning_horizon == 1 else f"H{planning_horizon}_categorical_CEM"
                            policy, audit = x02.learned_planner(next_state, reward_table, uncertainty, env,
                                                               int(planning_horizon), penalized, seed)
                            policy = table_policy(policy, env)
                            name = f"{model_name}__{planner_label}__{variant}"
                            policies[name] = policy
                            predicted = x02.learned_model_value(next_state, reward_table, env, policy)
                            true = e01.evaluate_policy_exact(env, policy)
                            predicted_values[name] = predicted
                            planner_rows.append({
                                "task": task, "world_model": model_name, "planner": planner_label,
                                "planner_variant": variant, "seed": seed,
                                "effective_horizon": min(int(planning_horizon), env.horizon),
                                "iterations": audit["iterations"], "candidates": audit["candidates"],
                                "elite_count": audit["elite_count"],
                                "minimum_unique_sequences": audit["minimum_unique_sequences"],
                                "support_mask_bypass": audit["support_mask_bypass"],
                                "predicted_model_value": predicted, "exact_true_value": true,
                                "exact_regret": oracle_value - true, "claim_boundary": CLAIM,
                            })
                            exploitation_rows.append({
                                "task": task, "method": name, "seed": seed,
                                "predicted_model_value": predicted, "exact_true_value": true,
                                "model_exploitation_gap": predicted - true,
                                "absolute_exploitation_gap": abs(predicted - true), "claim_boundary": CLAIM,
                            })
                except Exception as exc:
                    failures.append({"task": task, "component": f"planner:{model_name}", "seed": seed,
                                     "failure": type(exc).__name__, "detail": str(exc)[:200]})
            policies["exact_adaptive_oracle"] = oracle_policy
            crn, evaluation_n, max_se = evaluate_crn(env, policies, config)
            for method, policy in policies.items():
                exact = e01.evaluate_policy_exact(env, policy)
                unsupported = float(np.sum(policy * (~env.support)[None]) / max(np.sum(policy), 1e-12))
                paired_delta, paired_se = crn[method]
                policy_rows.append({
                    "task": task, "method": method, "seed": seed,
                    "method_group": "oracle" if method == "exact_adaptive_oracle" else ("world_model_planner" if "__H" in method else ("control" if method in config["controls"] else "model_free")),
                    "exact_true_return": exact, "behavior_relative_exact_return": exact - e01.evaluate_policy_exact(env, env.behavior),
                    "paired_mc_behavior_delta": paired_delta, "paired_mc_se": paired_se,
                    "evaluation_episodes": evaluation_n, "maximum_pairwise_se": max_se,
                    "unsupported_action_mass": unsupported, "claim_boundary": CLAIM,
                })
                regret = oracle_value - exact
                regret_rows.append({
                    "task": task, "method": method, "seed": seed, "exact_oracle_value": oracle_value,
                    "exact_policy_value": exact, "exact_regret": regret,
                    "negative_regret": regret < -float(config["evaluation"]["negative_regret_tolerance"]),
                    "claim_boundary": CLAIM,
                })

    if failures:
        raise RuntimeError(f"required adaptive benchmark rows failed: {failures[:5]}")
    if any(row["negative_regret"] for row in regret_rows):
        raise RuntimeError("negative exact regret detected")
    if any(row["support_mask_bypass"] for row in planner_rows):
        raise RuntimeError("support mask bypass detected")
    if any(row["planner"] != "H1_exhaustive" and (row["iterations"] != 3 or row["minimum_unique_sequences"] <= 1)
           for row in planner_rows):
        raise RuntimeError("categorical CEM execution gate failed")

    _write_csv(output / "adaptive_environment_contract.csv", contract_rows)
    _write_csv(output / "ehr_to_known_value_contract_matrix.csv", matrix_rows)
    _write_csv(output / "optimal_action_state_map.csv", map_rows)
    _write_csv(output / "adaptive_vs_best_fixed_gap.csv", gap_rows)
    _write_csv(output / "adaptive_policy_true_returns.csv", policy_rows)
    _write_csv(output / "adaptive_policy_regret.csv", regret_rows)
    _write_csv(output / "adaptive_world_model_planner_matrix.csv", planner_rows)
    _write_csv(output / "adaptive_exploitation_gap.csv", exploitation_rows)
    lines = ["# KDD-ADAPT01 adaptive environment preflight", "", "Decision: `adaptive_known_value_benchmark_complete`", ""]
    for row in receipts:
        lines.append(f"## {row['task']}")
        lines.append("")
        lines.append(f"- All gates pass: `{row['all_gates_pass']}`")
        lines.append(f"- Exact adaptive minus best fixed: `{row['adaptive_minus_best_fixed']:.6f}`")
        lines.append(f"- Nontrivial optimal actions: `{row['optimal_action_count_nontrivial_mass']}`")
        lines.append(f"- Null exact spread: `{row['null_exact_spread']:.3g}`")
        lines.append(f"- Null paired-MC spread: `{row['null_mc_spread']:.3g}`")
        lines.append("")
    lines.extend([
        "All action response is known by construction. Existing known-value results are retained unchanged and labeled monotone-control planning smoke tests.",
        "No raw EHR row, identifier, timestamp, trajectory, checkpoint, or retrospective EHR outcome was accessed or exported.",
        CLAIM,
    ])
    (output / "adaptive_environment_preflight_receipt.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    missing = [name for name in REQUIRED if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"missing outputs: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    output = args.output or ROOT / f"kdd_benchmark_discovery/results/kdd_adapt01_adaptive_known_value_{time.strftime('%Y%m%d_%H%M%S')}"
    run(args.config, output, args.preflight_only)


if __name__ == "__main__":
    main()
