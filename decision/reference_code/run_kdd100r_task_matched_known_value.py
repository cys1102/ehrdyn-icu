from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import resource
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from kdd_benchmark_discovery import run_kdd100_complete_known_value as core
from kdd_benchmark_discovery.kdd069_rssm_models import DreamerV3CategoricalRSSM


CLAIM_BOUNDARY = (
    "Task-matched procedural known-truth validation only. Aggregate EHR contracts shape "
    "simulator regimes; no patient row or trajectory is used or reconstructed. No real-EHR "
    "causal response, policy value, treatment benefit, clinical utility, deployment, or "
    "autonomous-decision claim."
)


@dataclass(frozen=True)
class MatchedProfile:
    task: str
    action_count: int
    supported_actions: int
    horizon: int
    missingness: float
    terminal_prevalence: float
    censoring_fraction: float
    top_action_share: float
    entropy_normalized: float
    reward_scale: float


PROFILES: dict[str, MatchedProfile] = {}
REGIMES: dict[str, dict[str, object]] = {}
TRAINING: dict[str, float | int] = {}
GENERIC_MODEL_FREE_POLICIES = core.model_free_policies
AUTHORIZED_MODEL_FREE = {
    "behavior_cloning",
    "discrete_bcq",
    "discrete_cql",
    "soft_spibb",
    "decision_transformer_adapter",
    "random_supported",
    "severity_rule",
}


class TaskMatchedEnvironment(core.KnownValueEnvironment):
    """Procedural simulator parameterized only by aggregate task contracts."""

    def __init__(self, spec: core.EnvSpec, seed: int) -> None:
        super().__init__(spec, seed)
        regime = REGIMES[spec.environment_id]
        profile = PROFILES[str(regime["task"])]
        self.profile = profile
        self.response_strength = float(regime["response_strength"])
        self.reward_regime = str(regime["reward_regime"])
        self.circularity_control = bool(regime["circularity_control"])
        self.dynamics_stress = bool(regime["dynamics_misspecification"])
        self.denominator_stress = bool(regime["denominator_misspecification"])
        self.termination_stress = bool(regime["termination_miscalibration"])
        self.observation_stress = bool(regime["observation_dependent_measurement"])
        self.supported[:] = False
        self.supported[: profile.supported_actions] = True
        # The action response is known by construction and is exactly zero in null regimes.
        self.b *= self.response_strength / max(float(np.linalg.norm(self.b, axis=1).mean()), 1e-8)
        self.reward_weight *= profile.reward_scale
        self.behavior_weight *= max(0.25, 2.0 * (1.0 - profile.entropy_normalized))
        self.behavior_weight[:, 0] += math.log(max(profile.top_action_share, 1e-4) * profile.action_count)

    def transition_mean(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        value = state @ self.a.T + self.b[action]
        value[:, 1] += 0.18 * state[:, 0]  # delayed response surface
        if self.dynamics_stress:
            value += 0.24 * np.square(np.tanh(state)) * np.sign(self.b[action] + 1e-6)
        return np.tanh(value)

    def _physiology_reward(self, next_state: np.ndarray) -> np.ndarray:
        value = next_state @ self.reward_weight - 0.05 * np.square(next_state).mean(axis=1)
        return np.clip(value, -self.profile.reward_scale, self.profile.reward_scale)

    def reward(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray, step: int) -> np.ndarray:
        physiology = self._physiology_reward(next_state)
        terminal = np.where(next_state[:, 0] < 0.0, self.profile.reward_scale, -self.profile.reward_scale)
        if self.reward_regime == "dense_physiology":
            value = physiology
        elif self.reward_regime == "sparse_terminal":
            value = terminal if step == self.spec.horizon - 1 else np.zeros(len(state))
        else:
            value = 0.35 * physiology + (0.65 * terminal if step == self.spec.horizon - 1 else 0.0)
        return value

    def observed_reward(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray, step: int) -> np.ndarray:
        value = self.reward(state, action, next_state, step)
        if self.circularity_control:
            # Deliberately invalid training reward: mechanically contains the current action.
            value = value + 0.35 * self.profile.reward_scale * action / max(self.spec.action_count - 1, 1)
        return value

    def terminal_probability(self, next_state: np.ndarray, step: int) -> np.ndarray:
        cumulative = np.clip(self.profile.terminal_prevalence, 1e-4, 0.95)
        base_hazard = 1.0 - (1.0 - cumulative) ** (1.0 / self.spec.horizon)
        logit = math.log(base_hazard / (1.0 - base_hazard)) + 0.55 * np.abs(next_state[:, 0])
        if self.termination_stress:
            logit += 0.8 * next_state[:, 1]
        return 1.0 / (1.0 + np.exp(-logit))

    def behavior(self, state: np.ndarray) -> np.ndarray:
        logits = state @ self.behavior_weight
        logits[:, ~self.supported] = -30.0
        return core._softmax(logits)

    def generate(self, seed: int) -> core.OfflineData:
        rng = np.random.default_rng(seed + 19)
        n, h, d, k = self.spec.episodes, self.spec.horizon, self.spec.state_dim, self.spec.action_count
        states = np.zeros((n, h, d), dtype=np.float32)
        next_states = np.zeros_like(states)
        actions = np.zeros((n, h), dtype=np.int16)
        rewards = np.zeros((n, h), dtype=np.float32)
        terminal = np.zeros((n, h), dtype=bool)
        behavior = np.zeros((n, h, k), dtype=np.float32)
        initial = rng.normal(0.0, 0.55, size=(n, d)).astype(np.float32)
        current = initial.copy()
        alive = np.ones(n, dtype=bool)
        censor_step = np.where(
            rng.random(n) < self.profile.censoring_fraction,
            rng.integers(1, h, size=n),
            h,
        )
        for step in range(h):
            states[:, step] = current
            probability = self.behavior(current)
            behavior[:, step] = probability
            action = np.asarray([rng.choice(k, p=row) for row in probability], dtype=np.int16)
            action[~alive] = 0
            actions[:, step] = action
            following = self.transition_mean(current, action) + rng.normal(0.0, 0.06, size=current.shape)
            following = np.clip(following, -2.5, 2.5).astype(np.float32)
            next_states[:, step] = following
            rewards[:, step] = np.where(alive, self.observed_reward(current, action, following, step), 0.0)
            ended = alive & (rng.random(n) < self.terminal_probability(following, step))
            censored = alive & (censor_step == step)
            if step == h - 1:
                ended |= alive & ~censored
            terminal[:, step] = ended
            alive &= ~(ended | censored)
            current = following
        if self.observation_stress:
            probability_observed = np.clip(
                1.0 - self.profile.missingness + 0.10 * np.abs(states[..., :1]), 0.01, 0.99
            )
            masks = rng.random(states.shape) < probability_observed
        else:
            masks = rng.random(states.shape) >= self.profile.missingness
        observed = np.where(masks, states, 0.0).astype(np.float32)
        deltas = np.zeros_like(states)
        for step in range(1, h):
            deltas[:, step] = np.where(masks[:, step - 1], 1.0, deltas[:, step - 1] + 1.0)
        return core.OfflineData(states, observed, masks, deltas, actions, rewards, next_states, terminal, behavior, initial)


def fit_converged(name: str, data: core.OfflineData, spec: core.EnvSpec, seed: int) -> core.WorldModelFit:
    """Train the KDD098R core architecture with the preregistered early-stop contract."""
    # The generic runner requests two extra ensemble labels per outer seed. Map those labels
    # onto the other two preregistered seeds so the ensemble uses only 3408/3411/3414.
    seed_map = {
        3509: 3411, 3610: 3414,
        3512: 3408, 3613: 3414,
        3515: 3408, 3616: 3411,
    }
    seed = seed_map.get(seed, seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(data.states))
    split = max(1, int(len(order) * 0.75))
    train, val = order[:split], order[split:]
    model = core._make_model(name, spec.state_dim, spec.action_count)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(TRAINING["learning_rate"]), weight_decay=1e-4)
    onehot = np.eye(spec.action_count, dtype=np.float32)[data.actions]
    dataset = TensorDataset(
        torch.from_numpy(data.observed[train]),
        torch.from_numpy(data.masks[train].astype(np.float32)),
        torch.from_numpy(data.deltas[train]),
        torch.from_numpy(onehot[train]),
        torch.from_numpy(data.next_states[train]),
    )
    loader = DataLoader(dataset, batch_size=int(TRAINING["batch_size"]), shuffle=True, generator=torch.Generator().manual_seed(seed))
    best_state: dict[str, torch.Tensor] | None = None
    best = float("inf")
    best_epoch = 0
    stale = 0
    start = time.perf_counter()
    for epoch in range(1, int(TRAINING["max_epochs"]) + 1):
        model.train()
        for values, masks, deltas, action, target in loader:
            output = model(values, masks, deltas, action)
            scale = torch.exp(output.log_scale)
            loss = (output.log_scale + 0.5 * torch.square((target - output.mean) / scale)).mean()
            loss = loss + 0.01 * output.auxiliary_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            output = model(
                torch.from_numpy(data.observed[val]),
                torch.from_numpy(data.masks[val].astype(np.float32)),
                torch.from_numpy(data.deltas[val]),
                torch.from_numpy(onehot[val]),
            )
            error = output.mean.numpy() - data.next_states[val]
            score = 0.7 * float(np.sqrt(np.mean(np.square(error)))) + 0.3 * float(np.mean(np.abs(error)))
        relative = (best - score) / max(abs(best), 1e-8) if np.isfinite(best) else np.inf
        if relative >= float(TRAINING["minimum_relative_validation_improvement"]):
            best, best_epoch, stale = score, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        elif epoch >= int(TRAINING["min_epochs"]):
            stale += 1
        if epoch >= int(TRAINING["min_epochs"]) and stale >= int(TRAINING["patience"]):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    training_seconds = time.perf_counter() - start
    model.eval()
    with torch.inference_mode():
        output = model(
            torch.from_numpy(data.observed[val]),
            torch.from_numpy(data.masks[val].astype(np.float32)),
            torch.from_numpy(data.deltas[val]),
            torch.from_numpy(onehot[val]),
        )
        prediction = output.mean.numpy()
        scale = np.exp(output.log_scale.numpy())
        if isinstance(model, DreamerV3CategoricalRSSM):
            recursive = model.rollout(
                torch.from_numpy(data.observed[val, 0]),
                torch.from_numpy(data.masks[val, 0].astype(np.float32)),
                torch.from_numpy(data.deltas[val, 0]),
                torch.from_numpy(onehot[val]),
            ).mean.numpy()
        else:
            current = data.observed[val, 0].copy()
            history, mask_history, delta_history, recursive_steps = [], [], [], []
            for step in range(spec.horizon):
                history.append(current.copy())
                mask_history.append(np.ones_like(current, dtype=np.float32))
                delta_history.append(np.zeros_like(current, dtype=np.float32))
                rec = model(
                    torch.from_numpy(np.stack(history, axis=1)),
                    torch.from_numpy(np.stack(mask_history, axis=1)),
                    torch.from_numpy(np.stack(delta_history, axis=1)),
                    torch.from_numpy(onehot[val, : step + 1]),
                ).mean[:, -1].numpy()
                recursive_steps.append(rec)
                current = rec
            recursive = np.stack(recursive_steps, axis=1)
    error = prediction - data.next_states[val]
    z = np.abs(error) / np.maximum(scale, 1e-5)
    reward_model = Ridge(alpha=1.0).fit(
        np.concatenate([data.states[train].reshape(-1, spec.state_dim), onehot[train].reshape(-1, spec.action_count)], axis=1),
        data.rewards[train].reshape(-1),
    )
    reward_pred = reward_model.predict(
        np.concatenate([prediction.reshape(-1, spec.state_dim), onehot[val].reshape(-1, spec.action_count)], axis=1)
    ).reshape(len(val), spec.horizon)
    term_x = np.concatenate([data.next_states[train].reshape(-1, spec.state_dim), onehot[train].reshape(-1, spec.action_count)], axis=1)
    term_y = data.terminal[train].reshape(-1).astype(int)
    if len(np.unique(term_y)) < 2:
        term_auc = float("nan")
    else:
        term_model = LogisticRegression(max_iter=100, class_weight="balanced", random_state=seed).fit(term_x, term_y)
        term_prob = term_model.predict_proba(
            np.concatenate([prediction.reshape(-1, spec.state_dim), onehot[val].reshape(-1, spec.action_count)], axis=1)
        )[:, 1]
        term_auc = float(roc_auc_score(data.terminal[val].reshape(-1), term_prob))
    return core.WorldModelFit(
        name=name,
        seed=seed,
        model=model,
        validation_rmse=float(np.sqrt(np.mean(np.square(error)))),
        validation_mae=float(np.mean(np.abs(error))),
        nll=float(np.mean(np.log(np.maximum(scale, 1e-5)) + 0.5 * np.square(error / np.maximum(scale, 1e-5)))),
        coverage90=float(np.mean(z <= 1.6448536)),
        rollout_rmse=float(np.sqrt(np.mean(np.square(recursive - data.next_states[val])))),
        reward_rmse=float(np.sqrt(np.mean(np.square(reward_pred - data.rewards[val])))),
        termination_auc=term_auc,
        uncertainty_ece=float(abs(np.mean(z <= 1.6448536) - 0.90)),
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        training_seconds=training_seconds,
        peak_memory_mb=float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
        status=f"converged_protocol_best_epoch_{best_epoch}_stopped_epoch_{epoch}",
        fingerprint=core._state_fingerprint(model),
    )


def authorized_model_free_policies(
    env: TaskMatchedEnvironment,
    data: core.OfflineData,
    behavior_policy,
):
    policies = GENERIC_MODEL_FREE_POLICIES(env, data, behavior_policy)
    output = {}
    for name, policy in policies.items():
        if name not in AUTHORIZED_MODEL_FREE:
            continue

        def support_constrained(state, selected=policy):
            probability = selected(state).copy()
            probability[:, ~env.supported] = 0.0
            total = probability.sum(axis=1, keepdims=True)
            fallback = np.broadcast_to(env.supported.astype(float), probability.shape)
            fallback = fallback / fallback.sum(axis=1, keepdims=True)
            return np.where(total > 0.0, probability / np.maximum(total, 1e-12), fallback)

        output[name] = support_constrained
    return output


def _build_specs(config: dict[str, object]) -> tuple[core.EnvSpec, ...]:
    profiles = config["task_profiles"]
    responses = config["response_regimes"]
    rewards = {"null": "dense_physiology", "weak": "sparse_terminal", "moderate": "terminal_plus_physiology"}
    specs: list[core.EnvSpec] = []
    for task, raw in profiles.items():
        profile = MatchedProfile(
            task=task,
            action_count=int(raw["action_count"]),
            supported_actions=int(raw["supported_actions"]),
            horizon=int(raw["horizon"]),
            missingness=float(raw["observation_missingness"]),
            terminal_prevalence=float(raw["terminal_prevalence"]),
            censoring_fraction=float(raw["censoring_fraction"]),
            top_action_share=float(raw["behavior_top_action_share"]),
            entropy_normalized=float(raw["behavior_entropy_normalized"]),
            reward_scale=float(raw["reward_scale"]),
        )
        PROFILES[task] = profile
        for response, strength in responses.items():
            environment_id = f"{task}_K{profile.action_count}_{response}"
            REGIMES[environment_id] = {
                "task": task,
                "response_regime": response,
                "response_strength": float(strength),
                "reward_regime": rewards[response],
                "circularity_control": response == "null",
                "dynamics_misspecification": response == "moderate",
                "low_support": profile.supported_actions < profile.action_count,
                "denominator_misspecification": response == "weak",
                "termination_miscalibration": response == "moderate",
                "observation_dependent_measurement": response == "null",
            }
            specs.append(core.EnvSpec(
                environment_id=environment_id,
                family=f"task_matched_{task}",
                episodes=128,
                horizon=profile.horizon,
                reward_sparsity=rewards[response],
                support="low" if profile.supported_actions < profile.action_count else "high",
                state_dim=8,
                missingness=profile.missingness,
                behavior_concentration=max(0.25, 2.0 * (1.0 - profile.entropy_normalized)),
                dynamics_misspecification=0.24 if response == "moderate" else 0.0,
                action_count=profile.action_count,
            ))
    return tuple(specs)


def _postprocess(output: Path, config: dict[str, object], generic_manifest_before: str) -> None:
    contracts = pd.read_csv(output / "known_value_environment_contracts.csv")
    contracts["task"] = contracts.environment_id.map(lambda value: REGIMES[value]["task"])
    contracts["response_regime"] = contracts.environment_id.map(lambda value: REGIMES[value]["response_regime"])
    contracts["reward_regime"] = contracts.environment_id.map(lambda value: REGIMES[value]["reward_regime"])
    contracts["supported_actions"] = contracts.task.map(lambda value: PROFILES[value].supported_actions)
    contracts["terminal_prevalence_target"] = contracts.task.map(lambda value: PROFILES[value].terminal_prevalence)
    contracts["censoring_fraction_target"] = contracts.task.map(lambda value: PROFILES[value].censoring_fraction)
    contracts["source"] = "procedural_generator_parameterized_by_aggregate_contracts_only"
    contracts["claim_boundary"] = CLAIM_BOUNDARY
    contracts.to_csv(output / "task_matched_environment_contracts.csv", index=False)

    aggregate = pd.read_csv(output / "offline_data_regimes.csv")
    aggregate["task"] = aggregate.environment_id.map(lambda value: REGIMES[value]["task"])
    aggregate["aggregate_source"] = "KDD097_and_KDD099R-A_frozen_aggregate_receipts"
    aggregate["patient_rows_accessed"] = False
    aggregate["trajectory_reconstruction"] = False
    aggregate["match_scope"] = "action_count_horizon_reward_terminal_censoring_missingness_behavior_support_delay_only"
    optional_rows = []
    for stress in config["optional_compact_stress"]:
        optional_rows.append({
            "environment_id": stress["profile"],
            "offline_episodes": 0,
            "decision_horizon": np.nan,
            "reward_sparsity": "not_assigned",
            "action_support": "optional_compact_stress_contract_only",
            "state_dimension": np.nan,
            "observation_missingness": np.nan,
            "behavior_concentration": np.nan,
            "dynamics_misspecification": np.nan,
            "action_count": stress["action_count"],
            "frozen_before_execution": True,
            "task": stress["profile"],
            "aggregate_source": "KDD100R_optional_compact_stress_definition",
            "patient_rows_accessed": False,
            "trajectory_reconstruction": False,
            "match_scope": "optional_stress_not_in_primary_factorial",
            "status": "not_run_with_reason",
            "reason": "Optional compact stress is not an invented real-EHR policy authorization and was not needed to evaluate the K25 primary contract.",
        })
    aggregate = pd.concat([aggregate, pd.DataFrame(optional_rows)], ignore_index=True)
    aggregate.to_csv(output / "aggregate_match_receipts.csv", index=False)

    response_rows, reward_rows = [], []
    for environment_id, regime in REGIMES.items():
        profile = PROFILES[str(regime["task"])]
        response_rows.append({"environment_id": environment_id, **regime, "known_true_response": True, "real_ehr_authorization": False})
        reward_rows.append({
            "environment_id": environment_id,
            "task": regime["task"],
            "reward_regime": regime["reward_regime"],
            "reward_scale": profile.reward_scale,
            "terminal_prevalence_target": profile.terminal_prevalence,
            "censoring_fraction_target": profile.censoring_fraction,
            "circularity_negative_control": regime["circularity_control"],
            "true_environment_reward_excludes_current_action": True,
            "offline_training_reward_contains_action_when_negative_control": regime["circularity_control"],
        })
    pd.DataFrame(response_rows).to_csv(output / "known_action_response_regimes.csv", index=False)
    pd.DataFrame(reward_rows).to_csv(output / "reward_and_termination_regimes.csv", index=False)

    component = pd.read_csv(output / "world_model_component_metrics.csv")
    component.to_csv(output / "predictive_policy_bridge_components.csv", index=False)
    ope = pd.read_csv(output / "ope_accuracy_and_coverage.csv")
    ope.to_csv(output / "ope_accuracy_bias_coverage.csv", index=False)

    regret = pd.read_csv(output / "policy_regret.csv")
    primary = ope[ope.stress_surface == "primary"].copy()
    behavior = primary[primary.method == "behavior_cloning"][["environment_id", "seed", "estimator", "true_value", "estimated_value"]]
    behavior = behavior.rename(columns={"true_value": "behavior_true", "estimated_value": "behavior_estimate"})
    joined = primary.merge(behavior, on=["environment_id", "seed", "estimator"], how="left")
    selected = joined.loc[joined.groupby(["environment_id", "seed", "estimator"]).estimated_value.idxmax()].copy()
    oracle = regret.groupby(["environment_id", "seed"], as_index=False).true_return.max().rename(columns={"true_return": "oracle_true"})
    selected = selected.merge(oracle, on=["environment_id", "seed"], how="left")
    selected["selected_policy_regret"] = selected.oracle_true - selected.true_value
    selected["false_improvement"] = (selected.estimated_value > selected.behavior_estimate) & (selected.true_value <= selected.behavior_true)
    selected.to_csv(output / "false_improvement_and_selected_regret.csv", index=False)

    term_rows = []
    for environment_id, regime in REGIMES.items():
        term_rows.append({
            "environment_id": environment_id,
            "task": regime["task"],
            "stress_active": regime["termination_miscalibration"],
            "calibration_shift": 0.8 if regime["termination_miscalibration"] else 0.0,
            "evaluation": "known_hazard_vs_train_fit_termination_head",
            "real_ehr_inference": "prohibited",
        })
    pd.DataFrame(term_rows).to_csv(output / "termination_miscalibration_stress.csv", index=False)

    gates = config["frozen_gates"]
    summary = pd.read_csv(output / "estimator_and_guardrail_pass_fail.csv")
    selected_summary = selected.groupby("estimator", as_index=False).agg(
        false_improvement_rate=("false_improvement", "mean"),
        mean_selected_regret=("selected_policy_regret", "mean"),
    )
    scale = max(float(primary.true_value.std()), 1e-8)
    selected_summary["normalized_selected_regret"] = selected_summary.mean_selected_regret / scale
    summary = summary.merge(selected_summary, on="estimator", how="left")
    rank = pd.read_csv(output / "policy_rank_recovery.csv")
    rank["task"] = rank.environment_id.map(lambda value: REGIMES[value]["task"])
    rank["response_regime"] = rank.environment_id.map(lambda value: REGIMES[value]["response_regime"])
    coverage = rank.groupby("estimator").task.nunique().div(len(PROFILES)).rename("task_profile_coverage")
    response_tau = rank.groupby(["estimator", "response_regime"]).kendall_tau.mean().groupby("estimator").min().rename("minimum_response_regime_kendall_tau")
    summary = summary.merge(coverage, on="estimator", how="left").merge(response_tau, on="estimator", how="left")
    action_rows = pd.concat([
        pd.read_csv(output / "world_model_planner_true_returns.csv")[["method", "unsupported_action_rate"]],
        pd.read_csv(output / "model_free_true_returns.csv")[["method", "unsupported_action_rate"]],
    ], ignore_index=True)
    unsupported = float(action_rows.unsupported_action_rate.max()) if len(action_rows) else 1.0
    summary["false_improvement_pass"] = summary.false_improvement_rate <= float(gates["false_improvement_rate_max"])
    summary["selected_regret_pass"] = summary.normalized_selected_regret <= float(gates["normalized_selected_policy_regret_max"])
    summary["task_profile_coverage_pass"] = summary.task_profile_coverage >= float(gates["minimum_task_profile_coverage"])
    summary["response_regime_rank_pass"] = summary.minimum_response_regime_kendall_tau >= float(gates["minimum_response_regime_kendall_tau"])
    summary["unsupported_action_rate_max"] = unsupported
    summary["unsupported_action_pass"] = unsupported <= float(gates["unsupported_action_rate_max"])
    summary["approved_before_real_ehr"] = summary[["value_accuracy_pass", "coverage_pass", "rank_pass", "false_improvement_pass", "selected_regret_pass", "task_profile_coverage_pass", "response_regime_rank_pass", "unsupported_action_pass"]].all(axis=1)
    summary["decision"] = np.where(summary.approved_before_real_ehr, "approved_synthetic_known_value_only", "not_approved_failed_known_value_gate")
    summary["gate_source"] = "immutable_KDD100R_config_frozen_before_execution"
    summary.to_csv(output / "estimator_and_guardrail_pass_fail.csv", index=False)

    approved = summary.loc[summary.approved_before_real_ehr, "estimator"].tolist()
    decision = "task_matched_known_value_pipeline_complete"
    policy_value = ";".join(approved) if approved else str(config["no_estimator_decision"])
    pd.DataFrame([{
        "experiment_id": "KDD100R",
        "decision": decision,
        "task_profiles": len(PROFILES),
        "environment_regimes": len(REGIMES),
        "seeds": len(config["seeds"]),
        "approved_ope_estimators": ";".join(approved) if approved else "none",
        "real_ehr_policy_value_decision": policy_value,
        "patient_level_ehr_accessed": False,
        "patient_trajectory_reconstructed": False,
        "generic_kdd100_v4_preserved": generic_manifest_before == str(config["upstream_sha256"]["kdd100_v4_artifact_manifest"]),
        "real_ehr_causal_action_response_validated": False,
        "clinical_utility_validated": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }]).to_csv(output / "decision.csv", index=False)

    multiplicity = pd.read_csv(output / "multiplicity_and_uncertainty_receipt.csv")
    multiplicity["estimator_gate_frozen_config"] = "configs/kdd100r_task_matched_known_value.json"
    multiplicity["false_improvement_and_selected_regret_in_gate"] = True
    multiplicity.to_csv(output / "multiplicity_and_uncertainty_receipt.csv", index=False)

    resource_frame = pd.read_csv(output / "resource_metrics.csv")
    resource_frame["synthetic_only"] = True
    resource_frame.to_csv(output / "resource_metrics.csv", index=False)

    privacy = pd.read_csv(output / "privacy_audit.csv")
    privacy = pd.concat([privacy, pd.DataFrame([
        {"check": "aggregate_contract_source_only", "status": "pass", "value": 1, "detail": "Only frozen aggregate KDD097/KDD099R-A values were transcribed into immutable config."},
        {"check": "patient_trajectory_reconstruction", "status": "pass", "value": 0, "detail": "All trajectories were generated independently by the procedural simulator."},
        {"check": "generic_kdd100_v4_manifest_preserved", "status": "pass" if generic_manifest_before == str(config["upstream_sha256"]["kdd100_v4_artifact_manifest"]) else "fail", "value": generic_manifest_before, "detail": "Generic KDD100-v4 artifact manifest is an immutable upstream preflight."},
    ])], ignore_index=True)
    privacy.to_csv(output / "privacy_audit.csv", index=False)

    report = f"""# KDD100R task-matched known-value validation

## Decision

`{decision}`

Real-EHR policy-value decision: `{policy_value}`.

KDD100R used independently generated procedural trajectories. No patient-level EHR row, exact timestamp, identifier, prediction, checkpoint, or reconstructed patient trajectory was read or exported. The environment parameters match only the aggregate task contracts frozen in KDD097/KDD099R-A.

## Scope

- Task profiles: sepsis, respiratory, shock; K=25 in every primary regime.
- Known response: null, weak, and moderate.
- Rewards: dense physiology, sparse terminal, and terminal-plus-physiology.
- Stress surfaces: action-reward circularity negative control, low support, denominator misspecification, nonlinear dynamics misspecification, termination miscalibration, and observation-dependent measurement.
- Core: GRU-D, causal Transformer, categorical RSSM, matched Gaussian ensemble; support-constrained MPC/CEM adapter, uncertainty-penalized adapter, compatible Dreamer actor adapter, model-free core, and frozen OPE estimators.

## Frozen gate interpretation

The KDD100-v4 accuracy, coverage, and rank gates were retained and extended with the preregistered false-improvement and selected-policy-regret gates. No gate was weakened to approve an estimator. If none pass, later real-EHR policy experiments are restricted to training, support, collapse, and exploitation diagnostics and may not report a policy-value comparison.

## Claim boundary

{CLAIM_BOUNDARY}
"""
    (output / "kdd100r_report.md").write_text(report, encoding="utf-8")
    (output / "kdd100_report.md").unlink()
    core._write_hashes(output)


def _tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.iterdir()):
        if item.is_file():
            digest.update(item.name.encode())
            digest.update(item.read_bytes())
    return digest.hexdigest()


def run(config_path: Path, *, smoke: bool = False) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not bool(config["synthetic_only"]):
        raise ValueError("KDD100R requires synthetic_only=true")
    output = Path(config["output_directory"] + ("_smoke" if smoke else ""))
    generic = Path("kdd_benchmark_discovery/results/kdd100_complete_known_value_pipeline_20260714_v4")
    manifest = generic / "artifact_hashes.json"
    generic_manifest_before = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if generic_manifest_before != str(config["upstream_sha256"]["kdd100_v4_artifact_manifest"]):
        raise RuntimeError("KDD100-v4 immutable artifact manifest hash drift")
    global TRAINING
    TRAINING = dict(config["training_contract"])
    specs = _build_specs(config)
    if smoke:
        TRAINING["max_epochs"] = 8
        TRAINING["min_epochs"] = 2
        TRAINING["patience"] = 2
        specs = tuple(copy.copy(spec) for spec in specs[:1])
        specs = tuple(core.EnvSpec(**{**spec.__dict__, "episodes": 24}) for spec in specs)
    core.SPECS = specs
    core.SEEDS = (int(config["seeds"][0]),) if smoke else tuple(int(value) for value in config["seeds"])
    core.BOOTSTRAPS = 20 if smoke else int(config["ope_contract"]["bootstrap_replicates"])
    core.CLAIM_BOUNDARY = CLAIM_BOUNDARY
    core.WORLD_MODELS = ("grud_world_model", "transformer_world_model", "dreamer_v3_categorical_rssm")
    core.KnownValueEnvironment = TaskMatchedEnvironment
    core.fit_world_model = fit_converged
    core.model_free_policies = authorized_model_free_policies
    core.run(output)
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != generic_manifest_before:
        raise RuntimeError("KDD100-v4 changed during KDD100R")
    _postprocess(output, config, generic_manifest_before)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KDD100R task-matched known-value validation")
    parser.add_argument("--config", type=Path, default=Path("configs/kdd100r_task_matched_known_value.json"))
    parser.add_argument("--smoke", action="store_true", help="Run one tiny synthetic regime for implementation validation")
    args = parser.parse_args()
    run(args.config, smoke=args.smoke)


if __name__ == "__main__":
    main()
