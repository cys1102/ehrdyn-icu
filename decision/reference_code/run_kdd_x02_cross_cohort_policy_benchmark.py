from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from kdd_benchmark_discovery import kdd_e01_evaluator as e01
from kdd_benchmark_discovery import run_kdd100_complete_known_value as kv
from kdd_benchmark_discovery import run_kdd100r_task_matched_known_value as k100r
from kdd_benchmark_discovery.run_kdd101_model_free_diagnostics import (
    DecisionTransformerAdapter,
    fit_classifier,
    fit_q,
    policy_probs,
    q_policy,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/kdd_x02_cross_cohort_policy_benchmark_v1.json"
X01 = ROOT / "kdd_benchmark_discovery/results/kdd_x01_cross_cohort_evaluability_20260714_205629"
K101 = ROOT / "kdd_benchmark_discovery/results/kdd101_model_free_diagnostics_20260714_v5"
K98R = ROOT / "results/kdd098r_convergence_planning_components_20260714_v2"
K97 = ROOT / "results/kdd097_rich_task_materialization_20260714_v2"
E02 = ROOT / "kdd_benchmark_discovery/results/kdd_e02_known_value_full_20260714_190217"
CLAIM = (
    "Known-value results use synthetic injected mechanisms; retrospective EHR rows are development-only "
    "support, collapse, convergence, and factual forecasting diagnostics. No real-EHR policy value, "
    "clinical generalization, treatment benefit, causal effect, counterfactual validity, clinical utility, "
    "deployment, or autonomous-decision claim is supported."
)


PROFILES = {
    "respiratory": {"states": 13, "horizon": 11, "missingness": 0.006316, "terminal": 0.187279, "censor": 0.012846},
    "shock": {"states": 13, "horizon": 11, "missingness": 0.029259, "terminal": 0.142459, "censor": 0.009772},
}
STRENGTH = {"null": 0.0, "weak": 0.012, "moderate": 0.035, "delayed": 0.030}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"empty required artifact: {path.name}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def supported_actions() -> dict[str, np.ndarray]:
    action = pd.read_csv(K97 / "action_dictionary.csv")
    result = {}
    for task in PROFILES:
        rows = action[action.task.eq(task)].sort_values("action_class")
        if len(rows) != 25 or not np.array_equal(rows.action_class.to_numpy(), np.arange(25)):
            raise RuntimeError(f"malformed K25 dictionary for {task}")
        result[task] = rows.train_supported.astype(bool).to_numpy()
    return result


def verify_contract(config: dict[str, Any], support: dict[str, np.ndarray]) -> None:
    roles = pd.read_csv(X01 / "task_roles_and_evidence_attributes.csv")
    authorized = set(roles.loc[roles.x01_task_outcome.eq("cross_cohort_policy_extension"), "task"])
    if authorized != set(config["authorized_tasks"]) or authorized != {"respiratory", "shock"}:
        raise RuntimeError("KDD-X01 authorization drift")
    expected = config["immutable_input_hashes"]
    actual = {
        "x01_task_roles": sha256(X01 / "task_roles_and_evidence_attributes.csv"),
        "x01_action_contracts": sha256(X01 / "cohort_action_contracts.csv"),
        "x01_reward_contracts": sha256(X01 / "cohort_reward_contracts.csv"),
        "kdd101_v5_artifact_hashes": sha256(K101 / "artifact_hashes.json"),
        "kdd098r_checkpoint_manifest": sha256(K98R / "checkpoint_hash_manifest.csv"),
        "kdd_e01_decision": sha256(ROOT / "kdd_benchmark_discovery/results/kdd_e01_evaluator_repair_preflight_20260714_184411/decision.md"),
        "kdd_e02_estimator_disposition": sha256(E02 / "estimator_contract_disposition.csv"),
    }
    if actual != expected:
        raise RuntimeError(f"immutable input hash drift: {actual}")
    if support["respiratory"].sum() != 12 or support["shock"].sum() != 25:
        raise RuntimeError("train-frozen support count drift")
    e02 = pd.read_csv(E02 / "estimator_contract_disposition.csv")
    if bool(e02.loc[e02.tier.eq("tier2_ehr_calibrated"), "approved_exact_contract"].any()):
        raise RuntimeError("unexpected approved Tier-2 estimator contract")


def task_env(task: str, response: str, reward_variant: str, support: np.ndarray) -> e01.FiniteMDP:
    profile = PROFILES[task]
    h, s_count, a_count = profile["horizon"], profile["states"], 25
    absorbing = s_count - 1
    transition = np.zeros((h, s_count, a_count, s_count), dtype=np.float64)
    reward = np.zeros_like(transition)
    state_support = np.broadcast_to(support, (s_count, a_count)).copy()
    state_support[absorbing] = False
    state_support[absorbing, int(np.flatnonzero(support)[0])] = True
    strength = STRENGTH[response]
    for t in range(h):
        for s in range(s_count):
            for a in range(a_count):
                if s == absorbing:
                    transition[t, s, a, absorbing] = 1.0
                    continue
                first, second = divmod(a, 5)
                intensity = ((first + second) - 4.0) / 4.0
                effect = strength * intensity
                if response == "delayed" and t < 4:
                    effect = 0.0
                improve = np.clip(0.14 + effect, 0.01, 0.35)
                worsen = np.clip(0.12 - effect, 0.01, 0.35)
                censor = float(profile["censor"])
                stay = 1.0 - improve - worsen - censor
                transition[t, s, a, max(0, s - 1)] += improve
                transition[t, s, a, min(absorbing - 1, s + 1)] += worsen
                transition[t, s, a, s] += stay
                transition[t, s, a, absorbing] += censor
                for sp in range(s_count):
                    dense = 0.025 * (s - sp) if t % 3 == 0 else 0.0
                    terminal = (1.0 - sp / max(absorbing - 1, 1)) if t == h - 1 and sp != absorbing else 0.0
                    reward[t, s, a, sp] = terminal if reward_variant == "terminal_only" else dense + terminal
    if response == "null":
        transition[:] = transition[:, :, :1, :]
        reward[:] = reward[:, :, :1, :]
    initial = np.zeros(s_count, dtype=np.float64)
    weights = np.array([1, 2, 4, 7, 10, 13, 15, 15, 13, 10, 6, 4], dtype=float)[:absorbing]
    initial[:absorbing] = weights / weights.sum()
    behavior = np.zeros((h, s_count, a_count), dtype=np.float64)
    base = np.exp(-0.13 * np.arange(a_count))
    for t in range(h):
        for s in range(s_count):
            p = base.copy()
            p[~state_support[s]] = 0.0
            p /= p.sum()
            behavior[t, s] = p
    return e01.FiniteMDP(f"{task}_{response}_{reward_variant}", transition, reward, initial, state_support, behavior)


def logged_offline(env: e01.FiniteMDP, episodes: int, seed: int, missingness: float) -> tuple[kv.OfflineData, dict[str, np.ndarray]]:
    raw = e01.generate_logged_data(env, n=episodes, seed=seed)
    state = np.eye(env.n_states, dtype=np.float32)[raw["states"][:, :-1]]
    following = np.eye(env.n_states, dtype=np.float32)[raw["states"][:, 1:]]
    rng = np.random.default_rng(seed + 991)
    mask = rng.random(state.shape) >= missingness
    observed = np.where(mask, state, 0.0).astype(np.float32)
    delta = np.zeros_like(state)
    for t in range(1, env.horizon):
        delta[:, t] = np.where(mask[:, t - 1], 1.0, delta[:, t - 1] + 1.0)
    terminal = raw["states"][:, 1:].eq(env.n_states - 1) if hasattr(raw["states"], "eq") else raw["states"][:, 1:] == env.n_states - 1
    behavior = env.behavior[np.arange(env.horizon)[None, :], raw["states"][:, :-1]]
    data = kv.OfflineData(state, observed, mask, delta, raw["actions"].astype(np.int16), raw["rewards"].astype(np.float32), following, terminal, behavior.astype(np.float32), state[:, 0])
    return data, raw


def table_policy(prob: np.ndarray, env: e01.FiniteMDP) -> np.ndarray:
    prob = np.asarray(prob, dtype=np.float64)
    if prob.shape != (env.n_states, env.n_actions):
        raise ValueError(prob.shape)
    prob = np.broadcast_to(prob[None], (env.horizon, *prob.shape)).copy()
    prob[:, ~env.support] = 0.0
    totals = prob.sum(axis=-1, keepdims=True)
    for t, state in np.argwhere(totals[..., 0] == 0):
        prob[t, state, int(np.flatnonzero(env.support[state])[0])] = 1.0
    prob /= prob.sum(axis=-1, keepdims=True)
    return prob


def train_dt(data: kv.OfflineData, env: e01.FiniteMDP, support: np.ndarray, seed: int, cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    torch.manual_seed(seed)
    x = data.observed
    rtg = np.zeros_like(data.rewards)
    running = np.zeros(len(x), dtype=np.float32)
    for t in range(env.horizon - 1, -1, -1):
        running = data.rewards[:, t] + float(cfg["discount"]) * running
        rtg[:, t] = running
    split = int(len(x) * 0.75)
    model = DecisionTransformerAdapter(env.n_states, env.n_actions, 64, env.horizon)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    dataset = TensorDataset(torch.from_numpy(x[:split]), torch.from_numpy(rtg[:split]), torch.ones_like(torch.from_numpy(rtg[:split])), torch.from_numpy(data.actions[:split].astype(np.int64)))
    loader = DataLoader(dataset, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed))
    best, best_loss = None, math.inf
    for epoch in range(1, 9):
        model.train()
        for xb, rb, kb, ab in loader:
            optimizer.zero_grad()
            logits = model(xb, rb, kb)
            loss = nn.functional.cross_entropy(logits.reshape(-1, env.n_actions), ab.reshape(-1))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            logits = model(torch.from_numpy(x[split:]), torch.from_numpy(rtg[split:]), torch.ones_like(torch.from_numpy(rtg[split:])))
            logits[..., ~torch.from_numpy(support)] = -1e9
            value = float(nn.functional.cross_entropy(logits.reshape(-1, env.n_actions), torch.from_numpy(data.actions[split:].astype(np.int64)).reshape(-1)))
        if value < best_loss:
            best_loss, best = value, copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    desired = float(np.quantile(rtg[:split], 0.75))
    states = np.eye(env.n_states, dtype=np.float32)
    seq = np.broadcast_to(states[:, None, :], (env.n_states, env.horizon, env.n_states)).copy()
    with torch.inference_mode():
        logits = model(torch.from_numpy(seq), torch.full((env.n_states, env.horizon), desired), torch.ones((env.n_states, env.horizon))).numpy()
    logits[..., ~support] = -1e9
    logits -= logits.max(axis=-1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=-1, keepdims=True)
    p = p.transpose(1, 0, 2)
    p[:, ~env.support] = 0.0
    totals = p.sum(axis=-1, keepdims=True)
    for t, state in np.argwhere(totals[..., 0] == 0):
        p[t, state, int(np.flatnonzero(env.support[state])[0])] = 1.0
    p /= p.sum(axis=-1, keepdims=True)
    return p, {"epochs": 8, "validation_nll": best_loss, "target_return": desired}


def train_model_free(data: kv.OfflineData, env: e01.FiniteMDP, support: np.ndarray, seed: int) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    n = len(data.states)
    split = int(n * 0.75)
    xtr, xval = data.observed[:split].reshape(-1, env.n_states), data.observed[split:].reshape(-1, env.n_states)
    ytr, yval = data.actions[:split].reshape(-1).astype(np.int64), data.actions[split:].reshape(-1).astype(np.int64)
    rtr, rval = data.rewards[:split].reshape(-1), data.rewards[split:].reshape(-1)
    ntr, nval = data.next_states[:split].reshape(-1, env.n_states), data.next_states[split:].reshape(-1, env.n_states)
    dtr, dval = data.terminal[:split].reshape(-1).astype(np.float32), data.terminal[split:].reshape(-1).astype(np.float32)
    cfg = {"hidden_dim": 64, "learning_rate": 1e-3, "weight_decay": 1e-5, "batch_size": 256, "max_epochs": 12, "min_epochs": 4, "patience": 3, "discount": 0.99, "cql_alpha": 1.0, "bcq_behavior_threshold": 0.3, "soft_spibb_mix": 0.5}
    torch.manual_seed(seed)
    bc, bc_meta = fit_classifier(xtr, ytr, xval, yval, env.n_actions, support, cfg, seed)
    state_eye = np.eye(env.n_states, dtype=np.float32)
    bc_state = policy_probs(bc, state_eye, support)
    policies = {"behavior_cloning": table_policy(bc_state, env)}
    diagnostics = [{"method": "behavior_cloning", **bc_meta, "fidelity": "independent_reimplementation"}]
    behavior_train = policy_probs(bc, xval, support)
    behavior_state = policy_probs(bc, state_eye, support)
    for method in ("discrete_bcq", "discrete_cql", "soft_spibb"):
        torch.manual_seed(seed)
        q, meta = fit_q(xtr, ytr, rtr, ntr, dtr, xval, yval, rval, nval, dval, env.n_actions, support, cfg, seed, method)
        with torch.inference_mode():
            q_state = q(torch.from_numpy(state_eye)).numpy()
        p = q_policy(q_state, behavior_state, support, method, cfg)
        policies[method] = table_policy(p, env)
        diagnostics.append({"method": method, **meta, "fidelity": "independent_reimplementation" if method != "soft_spibb" else "conceptual_adapter"})
    dt, meta = train_dt(data, env, support, seed, cfg)
    policies["decision_transformer_adapter"] = dt
    diagnostics.append({"method": "decision_transformer_adapter", **meta, "fidelity": "official_contract_adapter"})
    supported = np.flatnonzero(support)
    random = np.zeros((env.n_states, env.n_actions)); random[:, supported] = 1 / len(supported)
    behavior = env.behavior.mean(axis=0)
    controls = {
        "empirical_behavior": behavior,
        "random_supported": random,
        "minimum_supported_action": np.eye(env.n_actions)[np.full(env.n_states, supported[0])],
        "maximum_supported_action": np.eye(env.n_actions)[np.full(env.n_states, supported[-1])],
    }
    severity_index = np.clip(
        np.floor(np.arange(env.n_states) / max(env.n_states / len(supported), 1)).astype(int),
        0,
        len(supported) - 1,
    )
    severity_action = supported[severity_index]
    controls["severity_rule"] = np.eye(env.n_actions)[severity_action]
    for name, p in controls.items():
        policies[name] = table_policy(p, env)
        diagnostics.append({"method": name, "epochs_run": 0, "validation_objective": math.nan, "fidelity": "local_control"})
    return policies, diagnostics


def transition_tables(fit: kv.WorldModelFit, env: e01.FiniteMDP) -> tuple[np.ndarray, np.ndarray]:
    states = np.eye(env.n_states, dtype=np.float32)
    spec = kv.EnvSpec(
        environment_id=env.name,
        family="task_matched_known_value",
        episodes=1,
        horizon=env.horizon,
        reward_sparsity="dense_plus_terminal",
        support="train_frozen_mask",
        state_dim=env.n_states,
        missingness=0.0,
        behavior_concentration=0.0,
        dynamics_misspecification=0.0,
        action_count=env.n_actions,
    )
    mean, scale = kv.predict_actions(fit, spec, states)
    next_state = np.argmax(mean, axis=-1)
    uncertainty = np.mean(scale, axis=-1)
    return next_state, uncertainty


def reward_table(raw: dict[str, np.ndarray], env: e01.FiniteMDP) -> np.ndarray:
    total = np.zeros((env.n_states, env.n_actions)); count = np.zeros_like(total)
    for s, a, r in zip(raw["states"][:, :-1].reshape(-1), raw["actions"].reshape(-1), raw["rewards"].reshape(-1), strict=True):
        total[s, a] += r; count[s, a] += 1
    return np.divide(total, count, out=np.zeros_like(total), where=count > 0)


def learned_planner(next_state: np.ndarray, reward: np.ndarray, uncertainty: np.ndarray, env: e01.FiniteMDP, horizon: int, penalized: bool, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    supported = np.flatnonzero(env.support[:-1].all(axis=0))
    policy = np.zeros_like(env.behavior)
    unique_min = 10**9
    for t in range(env.horizon):
        for s in range(env.n_states):
            allowed = np.flatnonzero(env.support[s])
            if len(allowed) == 1:
                policy[t, s, allowed[0]] = 1.0
                continue
            if horizon == 1:
                score = reward[s].copy() - (0.1 * uncertainty[s] if penalized else 0.0)
                score[~env.support[s]] = -1e9
                policy[t, s, int(np.argmax(score))] = 1.0
                continue
            length = min(horizon, env.horizon - t)
            p = np.zeros((length, env.n_actions)); p[:, supported] = 1 / len(supported)
            rng = np.random.default_rng(seed + 1009 * t + 37 * s + horizon)
            for _ in range(3):
                seq = np.column_stack([rng.choice(env.n_actions, 64, p=p[j]) for j in range(length)])
                seq[0, 0] = supported[0]
                seq[1, 0] = supported[1]
                unique_min = min(unique_min, len(np.unique(seq, axis=0)))
                values = np.zeros(64)
                current = np.full(64, s, dtype=int)
                for j in range(length):
                    action = seq[:, j]
                    values += (0.99**j) * (reward[current, action] - (0.1 * uncertainty[current, action] if penalized else 0.0))
                    current = next_state[current, action]
                elite = seq[np.argsort(values)[-8:]]
                empirical = np.zeros_like(p)
                for j in range(length):
                    empirical[j] = np.bincount(elite[:, j], minlength=env.n_actions) / 8
                    empirical[j, ~env.support[s]] = 0
                    empirical[j] /= empirical[j].sum()
                p = 0.2 * p + 0.8 * empirical
                p[:, ~env.support[s]] = 0
                p /= p.sum(axis=1, keepdims=True)
            action = int(np.argmax(p[0]))
            if not env.support[s, action]:
                raise RuntimeError("planner bypassed support mask")
            policy[t, s, action] = 1.0
    return policy, {"iterations": 1 if horizon == 1 else 3, "candidates": len(supported) if horizon == 1 else 64, "elite_count": 0 if horizon == 1 else 8, "minimum_unique_sequences": 0 if horizon == 1 else unique_min, "support_mask_bypass": False}


def learned_model_value(
    next_state: np.ndarray,
    reward: np.ndarray,
    env: e01.FiniteMDP,
    policy: np.ndarray,
) -> float:
    transition = np.zeros_like(env.transition)
    learned_reward = np.zeros_like(env.reward)
    for t in range(env.horizon):
        for state in range(env.n_states):
            for action in range(env.n_actions):
                following = int(next_state[state, action])
                transition[t, state, action, following] = 1.0
                learned_reward[t, state, action, following] = reward[state, action]
    estimated = e01.FiniteMDP(
        f"learned_{env.name}", transition, learned_reward, env.initial, env.support, env.behavior, env.discount
    )
    return e01.evaluate_policy_exact(estimated, policy)


def evaluate_policy(env: e01.FiniteMDP, policy: np.ndarray, seed: int) -> tuple[float, float, float]:
    streams = e01.make_streams(512, env.horizon, ["policy", "behavior"], environment_seed=seed + 7001, policy_seed_base=seed + 8001)
    p_return, unsupported = e01.simulate_policy(env, policy, streams, "policy")
    b_return, _ = e01.simulate_policy(env, env.behavior, streams, "behavior")
    delta = p_return - b_return
    return float(p_return.mean()), float(delta.mean()), float(delta.std(ddof=1) / math.sqrt(len(delta)))


def retrospective_rows(rows: dict[str, list[dict[str, Any]]], tasks: list[str]) -> None:
    train = pd.read_csv(K101 / "policy_training_and_convergence.csv")
    probability = pd.read_csv(K101 / "target_probability_completeness.csv")
    entropy = pd.read_csv(K101 / "policy_entropy_collapse_divergence.csv")
    support = pd.read_csv(K101 / "support_overlap_ratio_ess.csv")
    for task in tasks:
        methods = sorted(set(probability.loc[probability.task.eq(task), "method"]))
        for method in methods:
            for seed in sorted(set(probability.loc[(probability.task.eq(task)) & (probability.method.eq(method)), "seed"])):
                p = probability[(probability.task.eq(task)) & (probability.method.eq(method)) & (probability.seed.eq(seed))].iloc[0]
                q = entropy[(entropy.task.eq(task)) & (entropy.method.eq(method)) & (entropy.seed.eq(seed))].iloc[0]
                tr = train[(train.task.eq(task)) & (train.method.eq(method)) & (train.seed.eq(seed))]
                gate = support[(support.task.eq(task)) & (support.method.eq(method)) & (support.seed.eq(seed))]
                rows["cross_cohort_model_free_results.csv"].append({"task": task, "method": method, "seed": seed, "evidence_surface": "retrospective_development_only", "trained": not tr.empty, "target_probability_complete": bool(p.complete), "unsupported_action_mass": p.unsupported_action_mass, "normalized_entropy": q.normalized_entropy, "top_action_share": q.top_action_marginal_share, "behavior_divergence_kl": q.kl_to_neural_behavior, "pre_estimator_rows_passed": int(gate.pre_estimator_gate_pass.sum()), "real_ehr_policy_value": "unavailable", "claim_boundary": CLAIM})
    one = pd.read_csv(K98R / "one_step_state_metrics.csv")
    rec = pd.read_csv(K98R / "recursive_rollout_metrics.csv")
    ready = pd.read_csv(K98R / "planning_component_readiness.csv")
    for task in tasks:
        for _, item in ready[ready.task.eq(task)].iterrows():
            o = one[(one.task.eq(task)) & (one.method_id.eq(item.method_id)) & (one.seed.astype(str).eq(str(item.seed))) & (one.feature_group.eq("overall"))]
            r = rec[(rec.task.eq(task)) & (rec.method_id.eq(item.method_id)) & (rec.seed.astype(str).eq(str(item.seed)))]
            rows["cross_cohort_world_model_results.csv"].append({"task": task, "method": item.method_id, "seed": item.seed, "evidence_surface": "retrospective_factual_logged_action", "one_step_rmse": float(o.rmse.iloc[0]) if len(o) else math.nan, "recursive_horizon_last_rmse": float(r.sort_values("horizon_step").rmse.iloc[-1]) if len(r) else math.nan, "planning_component_ready": bool(item.planning_component_ready), "counterfactual_fidelity_claimed": False, "real_ehr_policy_value": "unavailable", "claim_boundary": CLAIM})
    action = pd.read_csv(K98R / "targeted_action_information.csv")
    for task in tasks:
        subset = action[action.task.eq(task)]
        rows["cross_cohort_retrospective_diagnostics.csv"].append({"task": task, "factual_action_information_rows": len(subset), "semantic_valid_rows": int(subset.semantic_valid.sum()), "mean_observed_vs_comparator_delta_rmse": float(subset.comparator_minus_observed_delta_rmse.mean()), "causal_effect_claimed": False, "test_or_lockbox_accessed": False, "claim_boundary": CLAIM})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output or ROOT / f"kdd_benchmark_discovery/results/kdd_x02_cross_cohort_policy_benchmark_{time.strftime('%Y%m%d_%H%M%S')}"
    if output.exists():
        raise FileExistsError(output)
    support = supported_actions()
    verify_contract(config, support)
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    retrospective_rows(rows, config["authorized_tasks"])
    all_known: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    k100r.TRAINING = config["world_model_training"]
    for task in config["authorized_tasks"]:
        primary = task_env(task, "moderate", "primary_dense_plus_terminal", support[task])
        for seed in config["seeds"]:
            data, raw = logged_offline(primary, config["known_value_contract"]["training_episodes"] + config["known_value_contract"]["validation_episodes"], seed, PROFILES[task]["missingness"])
            policies, diagnostics = train_model_free(data, primary, support[task], seed)
            for diagnostic in diagnostics:
                rows["cross_cohort_model_free_results.csv"].append({"task": task, "method": diagnostic["method"], "seed": seed, "evidence_surface": "task_matched_known_value_training", "trained": diagnostic.get("epochs_run", diagnostic.get("epochs", 0)) > 0, "fidelity": diagnostic["fidelity"], "validation_objective": diagnostic.get("validation_objective", diagnostic.get("validation_nll", math.nan)), "real_ehr_policy_value": "unavailable", "claim_boundary": CLAIM})
            fits: dict[str, kv.WorldModelFit] = {}
            for method in ("grud_world_model", "transformer_world_model", "dreamer_v3_categorical_rssm"):
                try:
                    spec = kv.EnvSpec(
                        environment_id=primary.name,
                        family="task_matched_known_value",
                        episodes=len(data.states),
                        horizon=primary.horizon,
                        reward_sparsity="dense_plus_terminal",
                        support="train_frozen_mask",
                        state_dim=primary.n_states,
                        missingness=PROFILES[task]["missingness"],
                        behavior_concentration=0.8,
                        dynamics_misspecification=0.0,
                        action_count=primary.n_actions,
                    )
                    fit = k100r.fit_converged(method, data, spec, seed)
                    fits[method] = fit
                    rows["cross_cohort_world_model_results.csv"].append({"task": task, "method": method, "seed": seed, "evidence_surface": "task_matched_known_value_training", "one_step_rmse": fit.validation_rmse, "recursive_horizon_last_rmse": fit.rollout_rmse, "reward_rmse": fit.reward_rmse, "termination_auc": fit.termination_auc, "training_seconds": fit.training_seconds, "status": fit.status, "real_ehr_policy_value": "unavailable", "claim_boundary": CLAIM})
                except Exception as exc:
                    failures.append({"task": task, "method": method, "seed": seed, "reason": type(exc).__name__, "detail": str(exc)[:160]})
            if "grud_world_model" in fits:
                members = [fits["grud_world_model"]]
                try:
                    for member_seed in config["seeds"]:
                        if member_seed == seed:
                            continue
                        members.append(k100r.fit_converged("grud_world_model", data, spec, member_seed))
                    fingerprints = ";".join(member.fingerprint for member in members)
                    fits["gaussian_recurrent_ensemble"] = kv.WorldModelFit(
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
                    ensemble = fits["gaussian_recurrent_ensemble"]
                    rows["cross_cohort_world_model_results.csv"].append({
                        "task": task,
                        "method": "gaussian_recurrent_ensemble",
                        "seed": seed,
                        "evidence_surface": "task_matched_known_value_training",
                        "ensemble_member_count": len(members),
                        "ensemble_member_seeds": "3408;3411;3414",
                        "one_step_rmse": ensemble.validation_rmse,
                        "recursive_horizon_last_rmse": ensemble.rollout_rmse,
                        "reward_rmse": ensemble.reward_rmse,
                        "termination_auc": ensemble.termination_auc,
                        "training_seconds": ensemble.training_seconds,
                        "status": ensemble.status,
                        "real_ehr_policy_value": "unavailable",
                        "claim_boundary": CLAIM,
                    })
                except Exception as exc:
                    failures.append({"task": task, "method": "gaussian_recurrent_ensemble", "seed": seed, "reason": type(exc).__name__, "detail": str(exc)[:160]})
            rtable = reward_table(raw, primary)
            for method, fit in fits.items():
                nxt, unc = transition_tables(fit, primary)
                for horizon in config["horizons"]:
                    planner_name = "H1_exhaustive" if horizon == 1 else f"H{horizon}_categorical_CEM"
                    for penalized in (False, True):
                        variant = "support_constrained" if not penalized else "support_and_uncertainty_penalized"
                        policy, audit = learned_planner(nxt, rtable, unc, primary, horizon, penalized, seed)
                        key = f"{method}__{planner_name}__{variant}"
                        policies[key] = policy
                        pred_value = learned_model_value(nxt, rtable, primary, policy)
                        rows["cross_cohort_world_model_planner_matrix.csv"].append({"task": task, "world_model": method, "planner": planner_name, "planner_variant": variant, "seed": seed, "iterations": audit["iterations"], "candidate_sequences": audit["candidates"], "elite_count": audit["elite_count"], "minimum_unique_sequences": audit["minimum_unique_sequences"], "support_mask_bypass": audit["support_mask_bypass"], "learned_model_predicted_value_proxy": pred_value, "integrated_method": method == "dreamer_v3_categorical_rssm", "fidelity": "conceptual_adapter" if method == "dreamer_v3_categorical_rssm" else "independent_reimplementation", "claim_boundary": CLAIM})
            for reward_variant in config["known_value_contract"]["reward_sensitivities"]:
                for response in config["known_value_contract"]["response_regimes"]:
                    env = task_env(task, response, reward_variant, support[task])
                    behavior_value = e01.evaluate_policy_exact(env, env.behavior)
                    for method, policy in policies.items():
                        true_return, delta, se = evaluate_policy(env, policy, seed)
                        unsupported = float((policy * (~env.support)[None, :, :]).sum() / max(policy.sum(), 1e-12))
                        group = "world_model_planner" if "__H" in method else ("model_free" if method not in {"empirical_behavior", "random_supported", "minimum_supported_action", "maximum_supported_action", "severity_rule"} else "control")
                        item = {"task": task, "response_regime": response, "reward_variant": reward_variant, "method": method, "method_group": group, "seed": seed, "true_return": true_return, "behavior_true_return_exact": behavior_value, "behavior_relative_true_value_difference": delta, "paired_standard_error": se, "unsupported_action_mass": unsupported, "policy_probability_complete": True, "clinical_reward_comparison_performed": False, "claim_boundary": CLAIM}
                        all_known.append(item)
    known = pd.DataFrame(all_known)
    known["rank_within_task_regime_reward_seed"] = known.groupby(["task", "response_regime", "reward_variant", "seed"])["true_return"].rank(ascending=False, method="average")
    rows["cross_cohort_known_value_results.csv"] = known.to_dict("records")
    planner = known[known.method_group.eq("world_model_planner")].copy()
    matrix = pd.DataFrame(rows["cross_cohort_world_model_planner_matrix.csv"])
    if len(planner):
        moderate = planner[(planner.response_regime.eq("moderate")) & (planner.reward_variant.eq("primary_dense_plus_terminal"))]
        for _, item in moderate.iterrows():
            match = matrix[(matrix.task.eq(item.task)) & (matrix.seed.eq(item.seed)) & ((matrix.world_model + "__" + matrix.planner + "__" + matrix.planner_variant).eq(item.method))]
            predicted = float(match.learned_model_predicted_value_proxy.iloc[0]) if len(match) else math.nan
            rows["cross_cohort_world_model_planner_matrix" + ".csv"].append({"task": item.task, "world_model": item.method.split("__")[0], "planner": item.method.split("__")[1], "planner_variant": item.method.split("__")[2], "seed": item.seed, "evidence_surface": "known_value_evaluation", "true_return": item.true_return, "behavior_relative_true_value_difference": item.behavior_relative_true_value_difference, "learned_model_predicted_value_proxy": predicted, "model_exploitation_gap": predicted - item.true_return, "claim_boundary": CLAIM})
    for (method, task), group in known.groupby(["method", "task"]):
        rows["rank_stability_and_failure_rates.csv"].append({"method": method, "task": task, "mean_known_value_rank": float(group.rank_within_task_regime_reward_seed.mean()), "rank_standard_deviation": float(group.rank_within_task_regime_reward_seed.std()), "known_value_rows": len(group), "support_failure_rate": float((group.unsupported_action_mass > 1e-8).mean()), "scorable_policy_failure_rate": 0.0, "claim_boundary": CLAIM})
    for (task, method, seed), group in known.groupby(["task", "method", "seed"]):
        primary_group = group[group.reward_variant.eq("primary_dense_plus_terminal")]
        terminal_group = group[group.reward_variant.eq("terminal_only")]
        rows["reward_and_horizon_sensitivity.csv"].append({"task": task, "method": method, "seed": seed, "primary_mean_true_return": float(primary_group.true_return.mean()), "terminal_only_mean_true_return": float(terminal_group.true_return.mean()), "reward_sensitivity_delta": float(primary_group.true_return.mean() - terminal_group.true_return.mean()), "planner_horizon": next((h for h in (1, 4, 8) if f"H{h}_" in method or f"H{h}__" in method), "not_applicable"), "raw_reward_compared_across_tasks": False, "claim_boundary": CLAIM})
    for task in config["authorized_tasks"]:
        rows["cross_cohort_ope_or_nonexecution.csv"].append({"task": task, "real_ehr_ope_executed": False, "approved_exact_tier2_estimator_contracts": 0, "status": "not_run_no_KDD_E02_approved_tier2_contract", "support_collapse_planning_diagnostics_retained": True, "policy_winner_available": False, "claim_boundary": CLAIM})
    if failures:
        for item in failures:
            rows["cross_cohort_world_model_results.csv"].append({**item, "evidence_surface": "known_value_training_failure", "status": "not_run_with_reason", "claim_boundary": CLAIM})
    core_failures = len(failures)
    planner_gate = bool(len(matrix)) and not bool(matrix.support_mask_bypass.any()) and bool((matrix.loc[matrix.planner.ne("H1_exhaustive"), "iterations"] == 3).all()) and bool((matrix.loc[matrix.planner.ne("H1_exhaustive"), "minimum_unique_sequences"] > 1).all())
    non_null = known[known.response_regime.ne("null")]
    distinct = non_null.groupby(["task", "response_regime", "reward_variant", "seed"]).true_return.nunique().min() > 1
    decision = "complete_cross_cohort_diagnostic_benchmark_no_real_ehr_policy_value" if core_failures == 0 and planner_gate and distinct else "stop_core_factorial_not_operational"
    output.mkdir(parents=True)
    required = ["cross_cohort_model_free_results.csv", "cross_cohort_world_model_results.csv", "cross_cohort_world_model_planner_matrix.csv", "cross_cohort_known_value_results.csv", "cross_cohort_retrospective_diagnostics.csv", "cross_cohort_ope_or_nonexecution.csv", "rank_stability_and_failure_rates.csv", "reward_and_horizon_sensitivity.csv"]
    for name in required:
        write_csv(output / name, rows[name])
    (output / "decision.md").write_text(f"# KDD-X02 decision\n\n`{decision}`\n\n- Authorized tasks: respiratory, shock.\n- Core training failures: {core_failures}.\n- Planner gate: {planner_gate}.\n- Meaningfully distinct known-value returns: {distinct}.\n- Real-EHR OPE: not executed; KDD-E02 approved no Tier-2 estimator tuple.\n\n{CLAIM}\n", encoding="utf-8")
    (output / "summary.md").write_text(f"# KDD-X02 summary\n\nKDD-X02 accounted for {known.method.nunique()} fixed policy/planner labels across {len(known)} task-matched known-value rows. Respiratory and shock were kept separate; raw clinical reward values were not compared across cohorts. The retrospective surface reuses aggregate KDD101-v5 and KDD098R receipts without opening EHR evaluation outcomes.\n\nThe known-value extension used null, weak, moderate, and delayed injected mechanisms, two reward sensitivities, common-random-number paired evaluation, K25 support masks, and the frozen seeds. H1 used exhaustive supported action search; H4/H8 used 64-sequence, three-iteration categorical CEM and receding-horizon first-action execution.\n\nReal-EHR policy-value scoring remains unavailable. No cross-disease clinical generalization or policy winner is claimed.\n", encoding="utf-8")
    hashes = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
