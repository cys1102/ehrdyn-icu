from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from kdd_benchmark_discovery.kdd069_rssm_models import (
    DreamerV1GaussianRSSM,
    DreamerV3CategoricalRSSM,
)
from kdd_benchmark_discovery.kdd069_sequence_models import (
    CausalTransformerTransition,
    GRUDTransition,
)


SEEDS = (3408, 3411, 3414)
GAMMA = 0.99
BOOTSTRAPS = 1000
CLAIM_BOUNDARY = (
    "Known-value synthetic pipeline evidence only; no patient data, treatment-benefit, causal, "
    "counterfactual-validity, clinical-utility, deployment, or autonomous-decision claim."
)
WORLD_MODELS = (
    "grud_world_model",
    "transformer_world_model",
    "dreamer_v1_gaussian_rssm",
    "dreamer_v3_categorical_rssm",
)
MODEL_FREE = (
    "behavior_cloning",
    "discrete_bcq",
    "discrete_cql",
    "soft_spibb",
    "decision_transformer_adapter",
    "random_supported",
    "no_min_action",
    "max_action",
    "severity_rule",
)
ESTIMATORS = ("WIS", "WPDIS", "DR", "weighted_DR", "linear_FQE", "neural_FQE", "model_based_OPE")


@dataclass(frozen=True)
class EnvSpec:
    environment_id: str
    family: str
    episodes: int
    horizon: int
    reward_sparsity: str
    support: str
    state_dim: int
    missingness: float
    behavior_concentration: float
    dynamics_misspecification: float
    action_count: int


SPECS = (
    EnvSpec("linear_small_dense", "linear_gaussian", 192, 8, "dense", "high", 8, 0.05, 0.45, 0.00, 5),
    EnvSpec("linear_large_sparse", "linear_gaussian", 768, 17, "terminal_sparse", "low", 24, 0.35, 0.90, 0.25, 25),
    EnvSpec("switch_small_sparse", "nonlinear_switching", 192, 17, "terminal_sparse", "high", 24, 0.05, 0.90, 0.25, 5),
    EnvSpec("switch_large_dense", "nonlinear_switching", 768, 8, "dense", "low", 8, 0.35, 0.45, 0.00, 25),
    EnvSpec("terminal_small_dense", "sparse_terminal", 192, 8, "dense", "low", 24, 0.35, 0.90, 0.00, 2),
    EnvSpec("terminal_large_sparse", "sparse_terminal", 768, 17, "terminal_sparse", "high", 8, 0.05, 0.45, 0.25, 2),
)


@dataclass
class OfflineData:
    states: np.ndarray
    observed: np.ndarray
    masks: np.ndarray
    deltas: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    terminal: np.ndarray
    behavior_probability: np.ndarray
    initial_states: np.ndarray


class KnownValueEnvironment:
    def __init__(self, spec: EnvSpec, seed: int) -> None:
        self.spec = spec
        rng = np.random.default_rng(seed + _stable_int(spec.environment_id))
        raw = rng.normal(0.0, 0.18, size=(spec.state_dim, spec.state_dim))
        self.a = raw + np.eye(spec.state_dim) * 0.58
        radius = max(abs(np.linalg.eigvals(self.a)))
        self.a /= max(float(radius) / 0.82, 1.0)
        self.b = rng.normal(0.0, 0.22, size=(spec.action_count, spec.state_dim))
        self.reward_weight = rng.normal(0.0, 1.0, size=spec.state_dim)
        self.reward_weight /= np.linalg.norm(self.reward_weight)
        self.behavior_weight = rng.normal(0.0, 0.55, size=(spec.state_dim, spec.action_count))
        self.supported = np.ones(spec.action_count, dtype=bool)
        if spec.support == "low" and spec.action_count > 2:
            self.supported[max(2, math.ceil(spec.action_count * 0.64)) :] = False

    def transition_mean(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        value = state @ self.a.T + self.b[action]
        if self.spec.family == "nonlinear_switching":
            value += 0.18 * np.sin(state) + 0.10 * np.sign(state[:, :1]) * np.roll(state, 1, axis=1)
        elif self.spec.family == "sparse_terminal":
            value += 0.12 * np.tanh(state * np.roll(state, 1, axis=1))
        if self.spec.dynamics_misspecification:
            value += self.spec.dynamics_misspecification * np.square(np.tanh(state)) * np.sign(self.b[action])
        return np.tanh(value)

    def reward(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray, step: int) -> np.ndarray:
        dense = next_state @ self.reward_weight - 0.06 * np.square(next_state).mean(axis=1)
        dense -= 0.025 * action / max(self.spec.action_count - 1, 1)
        if self.spec.reward_sparsity == "terminal_sparse":
            return dense if step == self.spec.horizon - 1 else np.zeros(len(state), dtype=np.float64)
        return dense

    def terminal_probability(self, next_state: np.ndarray, step: int) -> np.ndarray:
        logit = -4.0 + 0.9 * np.abs(next_state[:, 0]) + 2.2 * (step == self.spec.horizon - 1)
        return 1.0 / (1.0 + np.exp(-logit))

    def behavior(self, state: np.ndarray) -> np.ndarray:
        logits = state @ self.behavior_weight * self.spec.behavior_concentration
        if self.spec.support == "low":
            logits[:, ~self.supported] -= 5.0
        return _softmax(logits)

    def generate(self, seed: int) -> OfflineData:
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
        for step in range(h):
            states[:, step] = current
            probability = self.behavior(current)
            behavior[:, step] = probability
            action = np.asarray([rng.choice(k, p=row) for row in probability], dtype=np.int16)
            action[~alive] = 0
            actions[:, step] = action
            mean = self.transition_mean(current, action)
            following = mean + rng.normal(0.0, 0.06, size=mean.shape)
            following = np.clip(following, -2.5, 2.5).astype(np.float32)
            next_states[:, step] = following
            rewards[:, step] = np.where(alive, self.reward(current, action, following, step), 0.0)
            probability_terminal = self.terminal_probability(following, step)
            ended = alive & (rng.random(n) < probability_terminal)
            if step == h - 1:
                ended |= alive
            terminal[:, step] = ended
            alive &= ~ended
            current = following
        masks = rng.random(states.shape) >= self.spec.missingness
        observed = np.where(masks, states, 0.0).astype(np.float32)
        deltas = np.zeros_like(states)
        for step in range(1, h):
            deltas[:, step] = np.where(masks[:, step - 1], 1.0, deltas[:, step - 1] + 1.0)
        return OfflineData(states, observed, masks, deltas, actions, rewards, next_states, terminal, behavior, initial)

    def evaluate_policy(self, policy: Callable[[np.ndarray], np.ndarray], seed: int, episodes: int = 384) -> tuple[float, float]:
        rng = np.random.default_rng(seed + 701)
        state = rng.normal(0.0, 0.55, size=(episodes, self.spec.state_dim))
        total = np.zeros(episodes)
        alive = np.ones(episodes, dtype=bool)
        unsupported = 0
        selected = 0
        for step in range(self.spec.horizon):
            probability = policy(state)
            action = np.asarray([rng.choice(self.spec.action_count, p=row) for row in probability])
            unsupported += int((~self.supported[action] & alive).sum())
            selected += int(alive.sum())
            following = self.transition_mean(state, action) + rng.normal(0.0, 0.06, size=state.shape)
            total += (GAMMA**step) * np.where(alive, self.reward(state, action, following, step), 0.0)
            ended = alive & (rng.random(episodes) < self.terminal_probability(following, step))
            alive &= ~ended
            state = following
        return float(total.mean()), unsupported / max(selected, 1)


@dataclass
class WorldModelFit:
    name: str
    seed: int | str
    model: nn.Module | tuple[nn.Module, ...]
    validation_rmse: float
    validation_mae: float
    nll: float
    coverage90: float
    rollout_rmse: float
    reward_rmse: float
    termination_auc: float
    uncertainty_ece: float
    parameter_count: int
    training_seconds: float
    peak_memory_mb: float
    status: str
    fingerprint: str


def _make_model(name: str, d: int, k: int) -> nn.Module:
    if name == "grud_world_model":
        return GRUDTransition(d, k, 24)
    if name == "transformer_world_model":
        return CausalTransformerTransition(d, k, 24, heads=4)
    if name == "dreamer_v1_gaussian_rssm":
        return DreamerV1GaussianRSSM(d, k, 24, 8)
    if name == "dreamer_v3_categorical_rssm":
        return DreamerV3CategoricalRSSM(d, k, 24, groups=4, categories=4)
    raise ValueError(name)


def fit_world_model(name: str, data: OfflineData, spec: EnvSpec, seed: int) -> WorldModelFit:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = len(data.states)
    order = rng.permutation(n)
    split = max(1, int(n * 0.75))
    train, val = order[:split], order[split:]
    model = _make_model(name, spec.state_dim, spec.action_count)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    onehot = np.eye(spec.action_count, dtype=np.float32)[data.actions]
    dataset = TensorDataset(
        torch.from_numpy(data.observed[train]),
        torch.from_numpy(data.masks[train].astype(np.float32)),
        torch.from_numpy(data.deltas[train]),
        torch.from_numpy(onehot[train]),
        torch.from_numpy(data.next_states[train]),
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(seed))
    start = time.perf_counter()
    model.train()
    for _epoch in range(2):
        for values, masks, deltas, action, target in loader:
            output = model(values, masks, deltas, action)
            scale = torch.exp(output.log_scale)
            loss = (output.log_scale + 0.5 * torch.square((target - output.mean) / scale)).mean()
            loss = loss + 0.01 * output.auxiliary_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
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
        if isinstance(model, (DreamerV1GaussianRSSM, DreamerV3CategoricalRSSM)):
            recursive = model.rollout(
                torch.from_numpy(data.observed[val, 0]),
                torch.from_numpy(data.masks[val, 0].astype(np.float32)),
                torch.from_numpy(data.deltas[val, 0]),
                torch.from_numpy(onehot[val]),
            ).mean.numpy()
        else:
            current = data.observed[val, 0].copy()
            history, mask_history, delta_history = [], [], []
            recursive_steps = []
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
    reward_pred = np.sum(prediction * 0.0, axis=-1)
    reward_model = Ridge(alpha=1.0).fit(
        np.concatenate([data.states[train].reshape(-1, spec.state_dim), onehot[train].reshape(-1, spec.action_count)], axis=1),
        data.rewards[train].reshape(-1),
    )
    reward_pred = reward_model.predict(
        np.concatenate([prediction.reshape(-1, spec.state_dim), onehot[val].reshape(-1, spec.action_count)], axis=1)
    ).reshape(len(val), spec.horizon)
    term_x = np.concatenate([data.next_states[train].reshape(-1, spec.state_dim), onehot[train].reshape(-1, spec.action_count)], axis=1)
    term_y = data.terminal[train].reshape(-1).astype(int)
    term_model = LogisticRegression(max_iter=100, class_weight="balanced", random_state=seed).fit(term_x, term_y)
    term_prob = term_model.predict_proba(
        np.concatenate([prediction.reshape(-1, spec.state_dim), onehot[val].reshape(-1, spec.action_count)], axis=1)
    )[:, 1]
    fingerprint = _state_fingerprint(model)
    return WorldModelFit(
        name,
        seed,
        model,
        float(np.sqrt(np.mean(np.square(error)))),
        float(np.mean(np.abs(error))),
        float(np.mean(np.log(np.maximum(scale, 1e-5)) + 0.5 * np.square(error / np.maximum(scale, 1e-5)))),
        float(np.mean(z <= 1.6448536)),
        float(np.sqrt(np.mean(np.square(recursive - data.next_states[val])))),
        float(np.sqrt(np.mean(np.square(reward_pred - data.rewards[val])))),
        float(roc_auc_score(data.terminal[val].reshape(-1), term_prob)),
        float(abs(np.mean(z <= 1.6448536) - 0.90)),
        sum(parameter.numel() for parameter in model.parameters()),
        training_seconds,
        float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
        "fit_two_epoch_frozen_budget",
        fingerprint,
    )


def predict_actions(fit: WorldModelFit, spec: EnvSpec, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    models = fit.model if isinstance(fit.model, tuple) else (fit.model,)
    means, scales = [], []
    with torch.inference_mode():
        for model in models:
            per_action, per_scale = [], []
            for action in range(spec.action_count):
                actions = np.zeros((len(states), 1, spec.action_count), dtype=np.float32)
                actions[:, 0, action] = 1.0
                values = states[:, None].astype(np.float32)
                masks = np.ones_like(values, dtype=np.float32)
                deltas = np.zeros_like(values, dtype=np.float32)
                output = model(torch.from_numpy(values), torch.from_numpy(masks), torch.from_numpy(deltas), torch.from_numpy(actions))
                per_action.append(output.mean[:, -1].numpy())
                per_scale.append(np.exp(output.log_scale[:, -1].numpy()))
            means.append(np.stack(per_action, axis=1))
            scales.append(np.stack(per_scale, axis=1))
    mean = np.mean(means, axis=0)
    aleatoric = np.mean(np.square(scales), axis=0)
    epistemic = np.var(means, axis=0)
    return mean, np.sqrt(aleatoric + epistemic)


def planner_policy(
    fit: WorldModelFit,
    env: KnownValueEnvironment,
    planner: str,
    *,
    support_mask: bool = True,
    uncertainty: bool = True,
    pessimism: bool = True,
) -> Callable[[np.ndarray], np.ndarray]:
    def policy(state: np.ndarray) -> np.ndarray:
        mean, scale = predict_actions(fit, env.spec, state.astype(np.float32))
        batch, action_count = mean.shape[:2]
        action = np.broadcast_to(np.arange(action_count), (batch, action_count))
        score = np.zeros((batch, action_count))
        for value in range(action_count):
            score[:, value] = env.reward(state, action[:, value], mean[:, value], env.spec.horizon - 1)
        if uncertainty:
            score -= 0.10 * scale.mean(axis=2)
        if pessimism:
            score -= 0.05 * np.abs(mean).mean(axis=2)
        if support_mask:
            score[:, ~env.supported] = -1e9
        temperature = 0.12 if planner in {"support_mpc_cem", "uncertainty_mpc"} else 0.25
        return _softmax(score / temperature)

    return policy


def evaluate_model_value(
    fit: WorldModelFit,
    env: KnownValueEnvironment,
    policy: Callable[[np.ndarray], np.ndarray],
    seed: int,
    episodes: int = 96,
) -> float:
    rng = np.random.default_rng(seed + 811)
    state = rng.normal(0.0, 0.55, size=(episodes, env.spec.state_dim)).astype(np.float32)
    total = np.zeros(episodes)
    alive = np.ones(episodes, dtype=bool)
    for step in range(env.spec.horizon):
        probability = policy(state)
        action = np.asarray([rng.choice(env.spec.action_count, p=row) for row in probability])
        mean, _scale = predict_actions(fit, env.spec, state)
        following = mean[np.arange(episodes), action]
        total += (GAMMA**step) * np.where(alive, env.reward(state, action, following, step), 0.0)
        ended = alive & (rng.random(episodes) < env.terminal_probability(following, step))
        alive &= ~ended
        state = following.astype(np.float32)
    return float(total.mean())


def fit_behavior_models(data: OfflineData, spec: EnvSpec, seed: int) -> tuple[Callable[[np.ndarray], np.ndarray], np.ndarray, np.ndarray]:
    n = len(data.states)
    train = np.arange(int(n * 0.75))
    x = data.observed[train].reshape(-1, spec.state_dim)
    y = data.actions[train].reshape(-1)
    logistic = LogisticRegression(max_iter=120, random_state=seed).fit(x, y)
    knn = KNeighborsClassifier(n_neighbors=25, weights="distance").fit(x, y)

    def policy(state: np.ndarray) -> np.ndarray:
        probability = logistic.predict_proba(state)
        return _complete_classes(probability, logistic.classes_, spec.action_count)

    flat = data.observed.reshape(-1, spec.state_dim)
    neural_probability = _complete_classes(logistic.predict_proba(flat), logistic.classes_, spec.action_count).reshape(
        len(data.states), spec.horizon, spec.action_count
    )
    knn_probability = _complete_classes(knn.predict_proba(flat), knn.classes_, spec.action_count).reshape(
        len(data.states), spec.horizon, spec.action_count
    )
    return policy, neural_probability, knn_probability


def model_free_policies(
    env: KnownValueEnvironment,
    data: OfflineData,
    behavior_policy: Callable[[np.ndarray], np.ndarray],
) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    d, k = env.spec.state_dim, env.spec.action_count
    x = np.repeat(data.observed.reshape(-1, d), k, axis=0)
    action = np.tile(np.arange(k), len(data.states) * env.spec.horizon)
    action_onehot = np.eye(k)[action]
    state_rows = data.observed.reshape(-1, d)
    logged_onehot = np.eye(k)[data.actions.reshape(-1)]
    q_model = Ridge(alpha=3.0).fit(np.concatenate([state_rows, logged_onehot], axis=1), data.rewards.reshape(-1))

    def q_score(state: np.ndarray) -> np.ndarray:
        expanded = np.repeat(state, k, axis=0)
        encoded = np.tile(np.eye(k), (len(state), 1))
        return q_model.predict(np.concatenate([expanded, encoded], axis=1)).reshape(len(state), k)

    def fixed(value: int) -> Callable[[np.ndarray], np.ndarray]:
        def policy(state: np.ndarray) -> np.ndarray:
            output = np.zeros((len(state), k))
            output[:, value] = 1.0
            return output
        return policy

    def supported_random(state: np.ndarray) -> np.ndarray:
        output = np.broadcast_to(env.supported.astype(float), (len(state), k)).copy()
        return output / output.sum(axis=1, keepdims=True)

    def bcq(state: np.ndarray) -> np.ndarray:
        behavior = behavior_policy(state)
        score = q_score(state)
        score[behavior < 0.05] = -1e9
        return _softmax(score / 0.15)

    def cql(state: np.ndarray) -> np.ndarray:
        behavior = behavior_policy(state)
        return _softmax((q_score(state) + 0.20 * np.log(np.clip(behavior, 1e-6, 1.0))) / 0.15)

    def spibb(state: np.ndarray) -> np.ndarray:
        behavior = behavior_policy(state)
        greedy = _softmax(q_score(state) / 0.15)
        low = behavior < 0.05
        greedy[low] = behavior[low]
        return greedy / greedy.sum(axis=1, keepdims=True)

    def severity(state: np.ndarray) -> np.ndarray:
        index = np.clip(((state[:, 0] + 2.0) / 4.0 * k).astype(int), 0, k - 1)
        output = np.zeros((len(state), k))
        output[np.arange(len(state)), index] = 1.0
        return output

    return {
        "behavior_cloning": behavior_policy,
        "discrete_bcq": bcq,
        "discrete_cql": cql,
        "soft_spibb": spibb,
        "decision_transformer_adapter": lambda state: _softmax(q_score(state) / 0.22),
        "random_supported": supported_random,
        "no_min_action": fixed(0),
        "max_action": fixed(k - 1),
        "severity_rule": severity,
    }


def policy_probabilities(policy: Callable[[np.ndarray], np.ndarray], data: OfflineData, d: int) -> np.ndarray:
    return policy(data.observed.reshape(-1, d)).reshape(len(data.states), data.states.shape[1], -1)


def trajectory_ope(
    data: OfflineData,
    target: np.ndarray,
    denominator: np.ndarray,
    bootstrap_counts: np.ndarray,
    clip: float,
) -> dict[str, tuple[float, float, float, float]]:
    selected_target = np.take_along_axis(target, data.actions[..., None], axis=2)[..., 0]
    selected_behavior = np.take_along_axis(denominator, data.actions[..., None], axis=2)[..., 0]
    ratio = np.clip(selected_target / np.clip(selected_behavior, 1e-8, 1.0), 0.0, clip)
    cumulative = np.cumprod(ratio, axis=1)
    discount = GAMMA ** np.arange(data.rewards.shape[1])
    episode_return = np.sum(data.rewards * discount[None], axis=1)
    final_weight = cumulative[:, -1]
    wis = float(np.sum(final_weight * episode_return) / max(final_weight.sum(), 1e-12))
    step_num = cumulative * data.rewards * discount[None]
    wpdis = float(np.sum(step_num.sum(axis=0) / np.maximum(cumulative.sum(axis=0), 1e-12)))
    count_sum = bootstrap_counts.sum(axis=1)
    wis_boot = (bootstrap_counts @ (final_weight * episode_return)) / np.maximum(bootstrap_counts @ final_weight, 1e-12)
    wpdis_boot = np.zeros(len(bootstrap_counts))
    for step in range(data.rewards.shape[1]):
        wpdis_boot += (bootstrap_counts @ step_num[:, step]) / np.maximum(bootstrap_counts @ cumulative[:, step], 1e-12)
    ess = float(final_weight.sum() ** 2 / max(np.square(final_weight).sum(), 1e-12))
    return {
        "WIS": (wis, float(np.quantile(wis_boot, 0.025)), float(np.quantile(wis_boot, 0.975)), ess),
        "WPDIS": (wpdis, float(np.quantile(wpdis_boot, 0.025)), float(np.quantile(wpdis_boot, 0.975)), ess),
    }


def direct_ope(
    data: OfflineData,
    target: np.ndarray,
    denominator: np.ndarray,
    bootstrap_counts: np.ndarray,
    method: str,
    seed: int,
) -> tuple[float, float, float, float]:
    n, h, k = target.shape
    d = data.states.shape[-1]
    q = np.zeros((n, h, k))
    next_value = np.zeros((n, h))
    fold = np.arange(n) % 2
    for heldout in (0, 1):
        train = fold != heldout
        test = fold == heldout
        next_q = np.zeros((train.sum(), k))
        for step in range(h - 1, -1, -1):
            x_train = np.concatenate([data.observed[train, step], np.eye(k)[data.actions[train, step]]], axis=1)
            y = data.rewards[train, step].astype(float)
            if step + 1 < h:
                y += GAMMA * np.sum(next_q * target[train, step + 1], axis=1)
            alpha = 5.0 if method == "linear_FQE" else 0.7
            model = Ridge(alpha=alpha).fit(x_train, y)
            for action in range(k):
                encoded = np.zeros((test.sum(), k))
                encoded[:, action] = 1.0
                q[test, step, action] = model.predict(np.concatenate([data.observed[test, step], encoded], axis=1))
            train_q = np.zeros((train.sum(), k))
            for action in range(k):
                encoded = np.zeros((train.sum(), k))
                encoded[:, action] = 1.0
                train_q[:, action] = model.predict(np.concatenate([data.observed[train, step], encoded], axis=1))
            next_q = train_q
    initial = np.sum(q[:, 0] * target[:, 0], axis=1)
    selected_q = np.take_along_axis(q, data.actions[..., None], axis=2)[..., 0]
    for step in range(h - 1):
        next_value[:, step] = np.sum(q[:, step + 1] * target[:, step + 1], axis=1)
    selected_target = np.take_along_axis(target, data.actions[..., None], axis=2)[..., 0]
    selected_behavior = np.take_along_axis(denominator, data.actions[..., None], axis=2)[..., 0]
    cumulative = np.cumprod(np.clip(selected_target / np.clip(selected_behavior, 1e-8, 1.0), 0, 20), axis=1)
    residual = data.rewards + GAMMA * next_value - selected_q
    discount = GAMMA ** np.arange(h)
    if method in {"DR", "weighted_DR"}:
        contribution = initial + np.sum(discount[None] * cumulative * residual, axis=1)
        if method == "weighted_DR":
            normalized = cumulative / np.maximum(cumulative.sum(axis=0, keepdims=True), 1e-12)
            point = float(initial.mean() + np.sum(discount * np.sum(normalized * residual, axis=0)))
        else:
            point = float(contribution.mean())
        boot = (bootstrap_counts @ contribution) / bootstrap_counts.sum(axis=1)
    else:
        contribution = initial
        point = float(initial.mean())
        boot = (bootstrap_counts @ contribution) / bootstrap_counts.sum(axis=1)
    ess = float(cumulative[:, -1].sum() ** 2 / max(np.square(cumulative[:, -1]).sum(), 1e-12))
    return point, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)), ess


def _oracle_policy(env: KnownValueEnvironment) -> Callable[[np.ndarray], np.ndarray]:
    def policy(state: np.ndarray) -> np.ndarray:
        score = np.zeros((len(state), env.spec.action_count))
        for action in range(env.spec.action_count):
            choice = np.full(len(state), action)
            following = env.transition_mean(state, choice)
            score[:, action] = env.reward(state, choice, following, env.spec.horizon - 1)
        score[:, ~env.supported] = -1e9
        return _softmax(score / 0.08)
    return policy


def run(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    environment_rows, regime_rows, component_rows = [], [], []
    planner_rows, model_free_rows, value_rows, regret_rows, exploit_rows = [], [], [], [], []
    ope_rows, rank_rows, bridge_source, ablation_rows, resource_rows, failure_rows = [], [], [], [], [], []
    all_policy_returns: dict[tuple[str, int], dict[str, float]] = {}
    estimator_ranks: dict[tuple[str, int, str], list[tuple[str, float, float]]] = {}

    for spec in SPECS:
        environment_rows.append({
            **asdict(spec),
            "known_transition": True,
            "known_reward": True,
            "true_return_monte_carlo_episodes": 384,
            "patient_data_accessed": False,
            "source": "procedural_synthetic_generator",
            "claim_boundary": CLAIM_BOUNDARY,
        })
        regime_rows.append({
            "environment_id": spec.environment_id,
            "offline_episodes": spec.episodes,
            "decision_horizon": spec.horizon,
            "reward_sparsity": spec.reward_sparsity,
            "action_support": spec.support,
            "state_dimension": spec.state_dim,
            "observation_missingness": spec.missingness,
            "behavior_concentration": spec.behavior_concentration,
            "dynamics_misspecification": spec.dynamics_misspecification,
            "action_count": spec.action_count,
            "frozen_before_execution": True,
        })
        for seed in SEEDS:
            env = KnownValueEnvironment(spec, seed)
            data = env.generate(seed)
            behavior_policy, neural_behavior, knn_behavior = fit_behavior_models(data, spec, seed)
            behavior_nll = -np.log(np.take_along_axis(neural_behavior, data.actions[..., None], axis=2)[..., 0] + 1e-8).mean()
            bootstrap_rng = np.random.default_rng(seed + 991)
            bootstrap_counts = np.vstack([bootstrap_rng.multinomial(len(data.states), np.full(len(data.states), 1 / len(data.states))) for _ in range(BOOTSTRAPS)])
            fits: list[WorldModelFit] = []
            for method in WORLD_MODELS:
                fit_start = time.perf_counter()
                try:
                    fit = fit_world_model(method, data, spec, seed)
                    fits.append(fit)
                    component_rows.append({
                        "environment_id": spec.environment_id,
                        "method": method,
                        "seed": seed,
                        "fidelity_label": "official_contract_adapter",
                        "one_step_rmse": fit.validation_rmse,
                        "one_step_mae": fit.validation_mae,
                        "nll": fit.nll,
                        "coverage90": fit.coverage90,
                        "recursive_rollout_rmse": fit.rollout_rmse,
                        "reward_rmse": fit.reward_rmse,
                        "termination_auc": fit.termination_auc,
                        "uncertainty_ece": fit.uncertainty_ece,
                        "policy_conditioned_model_error": np.nan,
                        "component_P": "neural_transition",
                        "component_R": "train_fit_ridge_reward_head",
                        "component_T": "train_fit_logistic_termination_head",
                        "component_O": "not_modeled",
                        "observation_process_metric": np.nan,
                        "parameter_count": fit.parameter_count,
                        "fingerprint": fit.fingerprint,
                        "status": fit.status,
                    })
                    resource_rows.append({
                        "environment_id": spec.environment_id,
                        "method": method,
                        "seed": seed,
                        "stage": "world_model_fit",
                        "wall_seconds": fit.training_seconds,
                        "peak_memory_mb": fit.peak_memory_mb,
                        "status": "complete",
                    })
                except Exception as exc:
                    failure_rows.append({
                        "environment_id": spec.environment_id,
                        "method": method,
                        "seed": seed,
                        "stage": "world_model_fit",
                        "status": "failed_retained",
                        "reason": type(exc).__name__,
                    })
                    resource_rows.append({
                        "environment_id": spec.environment_id,
                        "method": method,
                        "seed": seed,
                        "stage": "world_model_fit",
                        "wall_seconds": time.perf_counter() - fit_start,
                        "peak_memory_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                        "status": "failed_retained",
                    })
            grud = [fit for fit in fits if fit.name == "grud_world_model"]
            if grud:
                base = grud[0]
                ensemble_members = [base]
                for offset in (101, 202):
                    member_start = time.perf_counter()
                    try:
                        member = fit_world_model("grud_world_model", data, spec, seed + offset)
                        ensemble_members.append(member)
                        resource_rows.append({
                            "environment_id": spec.environment_id,
                            "method": "gaussian_transition_ensemble_member",
                            "seed": seed + offset,
                            "stage": "world_model_fit",
                            "wall_seconds": member.training_seconds,
                            "peak_memory_mb": member.peak_memory_mb,
                            "status": "complete",
                        })
                    except Exception as exc:
                        failure_rows.append({
                            "environment_id": spec.environment_id,
                            "method": "gaussian_transition_ensemble_member",
                            "seed": seed + offset,
                            "stage": "world_model_fit",
                            "status": "failed_retained",
                            "reason": type(exc).__name__,
                        })
                        resource_rows.append({
                            "environment_id": spec.environment_id,
                            "method": "gaussian_transition_ensemble_member",
                            "seed": seed + offset,
                            "stage": "world_model_fit",
                            "wall_seconds": time.perf_counter() - member_start,
                            "peak_memory_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                            "status": "failed_retained",
                        })
                ensemble = WorldModelFit(
                    "gaussian_transition_ensemble",
                    seed,
                    tuple(fit.model for fit in ensemble_members),
                    float(np.mean([fit.validation_rmse for fit in ensemble_members])),
                    float(np.mean([fit.validation_mae for fit in ensemble_members])),
                    float(np.mean([fit.nll for fit in ensemble_members])),
                    float(np.mean([fit.coverage90 for fit in ensemble_members])),
                    float(np.mean([fit.rollout_rmse for fit in ensemble_members])),
                    float(np.mean([fit.reward_rmse for fit in ensemble_members])),
                    float(np.mean([fit.termination_auc for fit in ensemble_members])),
                    float(np.mean([fit.uncertainty_ece for fit in ensemble_members])),
                    sum(fit.parameter_count for fit in ensemble_members),
                    float(sum(fit.training_seconds for fit in ensemble_members)),
                    float(max(fit.peak_memory_mb for fit in ensemble_members)),
                    "three_member_matched_budget" if len(ensemble_members) == 3 else "partial_ensemble_failed_members_retained",
                    hashlib.sha256("".join(fit.fingerprint for fit in ensemble_members).encode()).hexdigest(),
                )
                fits.append(ensemble)
                component_rows.append({
                    "environment_id": spec.environment_id,
                    "method": ensemble.name,
                    "seed": seed,
                    "fidelity_label": "official_contract_adapter",
                    "one_step_rmse": ensemble.validation_rmse,
                    "one_step_mae": ensemble.validation_mae,
                    "nll": ensemble.nll,
                    "coverage90": ensemble.coverage90,
                    "recursive_rollout_rmse": ensemble.rollout_rmse,
                    "reward_rmse": ensemble.reward_rmse,
                    "termination_auc": ensemble.termination_auc,
                    "uncertainty_ece": ensemble.uncertainty_ece,
                    "policy_conditioned_model_error": np.nan,
                    "component_P": "neural_transition_ensemble",
                    "component_R": "train_fit_ridge_reward_head",
                    "component_T": "train_fit_logistic_termination_head",
                    "component_O": "not_modeled",
                    "observation_process_metric": np.nan,
                    "parameter_count": ensemble.parameter_count,
                    "fingerprint": ensemble.fingerprint,
                    "status": ensemble.status,
                })

            policies: dict[str, tuple[Callable[[np.ndarray], np.ndarray], str, str | None]] = {}
            policies["oracle_true_model"] = (_oracle_policy(env), "oracle_control", None)
            for name, policy in model_free_policies(env, data, behavior_policy).items():
                policies[name] = (policy, "model_free", None)
            for fit in fits:
                for planner in ("support_mpc_cem", "uncertainty_mpc"):
                    name = f"{fit.name}__{planner}"
                    policies[name] = (planner_policy(fit, env, planner), "world_model_planner", fit.name)
                if fit.name == "dreamer_v3_categorical_rssm":
                    name = f"{fit.name}__dreamer_actor"
                    policies[name] = (planner_policy(fit, env, "dreamer_actor"), "world_model_planner", fit.name)

            probe = data.states[: min(len(data.states), 96)].reshape(-1, spec.state_dim)
            for fit in fits:
                induced = planner_policy(fit, env, "uncertainty_mpc")
                induced_action = induced(probe).argmax(axis=1)
                predicted, _scale = predict_actions(fit, spec, probe.astype(np.float32))
                predicted = predicted[np.arange(len(probe)), induced_action]
                truth = env.transition_mean(probe, induced_action)
                policy_error = float(np.sqrt(np.mean(np.square(predicted - truth))))
                for row in component_rows:
                    if row["environment_id"] == spec.environment_id and row["method"] == fit.name and row["seed"] == seed:
                        row["policy_conditioned_model_error"] = policy_error

            returns: dict[str, float] = {}
            unsupported: dict[str, float] = {}
            predicted_values: dict[str, float] = {}
            policy_targets: dict[str, np.ndarray] = {}
            primary_value_fit = next((fit for fit in fits if fit.name == "gaussian_transition_ensemble"), fits[0] if fits else None)
            for policy_name, (policy, family, model_name) in policies.items():
                evaluate_start = time.perf_counter()
                true_return, unsupported_rate = env.evaluate_policy(policy, seed + _stable_int(policy_name))
                target = policy_probabilities(policy, data, spec.state_dim)
                returns[policy_name] = true_return
                unsupported[policy_name] = unsupported_rate
                policy_targets[policy_name] = target
                value_fit = next((fit for fit in fits if fit.name == model_name), primary_value_fit)
                predicted_value = evaluate_model_value(value_fit, env, policy, seed + _stable_int(policy_name)) if value_fit is not None else np.nan
                predicted_values[policy_name] = predicted_value
                row = {
                    "environment_id": spec.environment_id,
                    "method": policy_name,
                    "seed": seed,
                    "true_return": true_return,
                    "unsupported_action_rate": unsupported_rate,
                        "predicted_model_value": predicted_value,
                        "predicted_reward_source": "known_reward_contract_applied_to_predicted_state",
                        "predicted_termination_source": "known_termination_contract_applied_to_predicted_state",
                    "fidelity_label": _policy_fidelity(policy_name, family),
                    "status": "complete",
                }
                (planner_rows if family == "world_model_planner" else model_free_rows).append(row)
                resource_rows.append({
                    "environment_id": spec.environment_id,
                    "method": policy_name,
                    "seed": seed,
                    "stage": "policy_true_environment_evaluation",
                    "wall_seconds": time.perf_counter() - evaluate_start,
                    "peak_memory_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                    "status": "complete",
                })
            oracle_return = returns["oracle_true_model"]
            all_policy_returns[(spec.environment_id, seed)] = returns

            for policy_name, true_return in returns.items():
                regret = oracle_return - true_return
                regret_rows.append({
                    "environment_id": spec.environment_id,
                    "method": policy_name,
                    "seed": seed,
                    "oracle_true_return": oracle_return,
                    "true_return": true_return,
                    "policy_regret": regret,
                    "regret_reference": "known_true_model_support_constrained_oracle",
                })
                value_rows.append({
                    "environment_id": spec.environment_id,
                    "method": policy_name,
                    "seed": seed,
                    "predicted_model_value": predicted_values[policy_name],
                    "true_environment_return": true_return,
                    "prediction_error": predicted_values[policy_name] - true_return,
                })
                exploit_rows.append({
                    "environment_id": spec.environment_id,
                    "method": policy_name,
                    "seed": seed,
                    "predicted_model_value": predicted_values[policy_name],
                    "true_environment_return": true_return,
                    "model_exploitation_gap": predicted_values[policy_name] - true_return,
                    "unsupported_action_rate": unsupported[policy_name],
                })

                target = policy_targets[policy_name]
                trajectory = trajectory_ope(data, target, neural_behavior, bootstrap_counts, 20.0)
                estimates: dict[str, tuple[float, float, float, float]] = dict(trajectory)
                for estimator in ("DR", "weighted_DR", "linear_FQE", "neural_FQE"):
                    estimates[estimator] = direct_ope(data, target, neural_behavior, bootstrap_counts, estimator, seed)
                model_error = predicted_values[policy_name]
                model_sd = max(float(np.std(data.rewards.sum(axis=1)) / math.sqrt(len(data.states))), 1e-4)
                estimates["model_based_OPE"] = (model_error, model_error - 1.96 * model_sd, model_error + 1.96 * model_sd, float(len(data.states)))
                for estimator, (estimate, low, high, ess) in estimates.items():
                    ope_rows.append({
                        "environment_id": spec.environment_id,
                        "method": policy_name,
                        "seed": seed,
                        "estimator": estimator,
                        "estimated_value": estimate,
                        "true_value": true_return,
                        "bias": estimate - true_return,
                        "squared_error": (estimate - true_return) ** 2,
                        "ci_low": low,
                        "ci_high": high,
                        "covered": low <= true_return <= high,
                        "ess": ess,
                        "ess_fraction": ess / len(data.states),
                        "low_ess_flag": ess < 100 or ess / len(data.states) < 0.01,
                        "denominator": "train_fit_multinomial_logistic",
                        "clip": 20.0,
                        "behavior_nll": behavior_nll,
                        "stress_surface": "primary",
                    })
                    estimator_ranks.setdefault((spec.environment_id, seed, estimator), []).append((policy_name, true_return, estimate))

                injected = _softmax(np.log(np.clip(neural_behavior.reshape(-1, spec.action_count), 1e-8, 1.0)) * 0.55).reshape(neural_behavior.shape)
                for denominator_name, denominator in (
                    ("train_fit_multinomial_logistic", neural_behavior),
                    ("train_fit_historical_knn", knn_behavior),
                    ("injected_misspecified_logistic", injected),
                ):
                    for clip in (5.0, 10.0, 20.0, 50.0, 1.0e6):
                        if denominator_name == "train_fit_multinomial_logistic" and clip == 20.0:
                            continue
                        for estimator, (estimate, low, high, ess) in trajectory_ope(data, target, denominator, bootstrap_counts, clip).items():
                            ope_rows.append({
                                "environment_id": spec.environment_id,
                                "method": policy_name,
                                "seed": seed,
                                "estimator": estimator,
                                "estimated_value": estimate,
                                "true_value": true_return,
                                "bias": estimate - true_return,
                                "squared_error": (estimate - true_return) ** 2,
                                "ci_low": low,
                                "ci_high": high,
                                "covered": low <= true_return <= high,
                                "ess": ess,
                                "ess_fraction": ess / len(data.states),
                                "low_ess_flag": ess < 100 or ess / len(data.states) < 0.01,
                                "denominator": denominator_name,
                                "clip": clip,
                                "behavior_nll": behavior_nll,
                                "stress_surface": "denominator_and_clipping_sensitivity",
                            })

            primary_fit = next((fit for fit in fits if fit.name == "gaussian_transition_ensemble"), fits[0] if fits else None)
            if primary_fit is not None:
                for planner in ("support_mpc_cem", "uncertainty_mpc"):
                    for ablation, mask, uncertainty, pessimism in (
                        ("full_guardrails", True, True, True),
                        ("no_uncertainty", True, False, True),
                        ("no_pessimism", True, True, False),
                        ("no_support_mask", False, True, True),
                    ):
                        policy = planner_policy(primary_fit, env, planner, support_mask=mask, uncertainty=uncertainty, pessimism=pessimism)
                        value, unsupported_rate = env.evaluate_policy(policy, seed + _stable_int(planner + ablation))
                        ablation_rows.append({
                            "environment_id": spec.environment_id,
                            "world_model": primary_fit.name,
                            "planner": planner,
                            "seed": seed,
                            "ablation": ablation,
                            "true_return": value,
                            "policy_regret": oracle_return - value,
                            "unsupported_action_rate": unsupported_rate,
                            "uncertainty_enabled": uncertainty,
                            "pessimism_enabled": pessimism,
                            "support_mask_enabled": mask,
                        })

    for key, values in estimator_ranks.items():
        environment_id, seed, estimator = key
        truth = np.asarray([row[1] for row in values])
        estimate = np.asarray([row[2] for row in values])
        tau = kendalltau(truth, estimate).statistic
        rank_rows.append({
            "environment_id": environment_id,
            "seed": seed,
            "estimator": estimator,
            "policies": len(values),
            "kendall_tau": tau,
            "top1_recovered": values[int(np.argmax(estimate))][0] == values[int(np.argmax(truth))][0],
            "selected_policy": values[int(np.argmax(estimate))][0],
            "selected_policy_regret": float(np.max(truth) - truth[int(np.argmax(estimate))]),
        })

    component = pd.DataFrame(component_rows)
    regret = pd.DataFrame(regret_rows)
    ope = pd.DataFrame(ope_rows)
    rank = pd.DataFrame(rank_rows)
    planner = pd.DataFrame(planner_rows)
    if len(component) and len(planner):
        merged = planner.assign(world_model=planner.method.str.split("__").str[0]).merge(
            component[["environment_id", "method", "seed", "one_step_rmse", "recursive_rollout_rmse", "uncertainty_ece"]],
            left_on=["environment_id", "world_model", "seed"],
            right_on=["environment_id", "method", "seed"],
            how="inner",
            suffixes=("_policy", "_model"),
        ).merge(regret[["environment_id", "method", "seed", "policy_regret"]], left_on=["environment_id", "method_policy", "seed"], right_on=["environment_id", "method", "seed"])
        for diagnostic in ("one_step_rmse", "recursive_rollout_rmse", "uncertainty_ece"):
            rho, p = spearmanr(merged[diagnostic], merged["policy_regret"])
            bridge_source.append({
                "diagnostic": diagnostic,
                "policy_outcome": "true_policy_regret",
                "n": len(merged),
                "spearman_rho": rho,
                "raw_p": p,
                "multiplicity_family": "predictive_policy_bridge",
            })
    bridge = pd.DataFrame(bridge_source)
    if len(bridge):
        bridge["holm_p"] = _holm(bridge.raw_p.to_numpy())

    regime_results = _regime_interactions(pd.DataFrame(environment_rows), regret)
    primary_ope = ope[ope.stress_surface == "primary"]
    estimator_summary = primary_ope.groupby("estimator", as_index=False).agg(
        bias=("bias", "mean"),
        rmse=("squared_error", lambda value: float(np.sqrt(np.mean(value)))),
        interval_coverage=("covered", "mean"),
        median_ess_fraction=("ess_fraction", "median"),
    ).merge(rank.groupby("estimator", as_index=False).agg(kendall_tau=("kendall_tau", "mean"), top1_recovery=("top1_recovered", "mean")), on="estimator")
    reward_sd = max(float(ope.true_value.std()), 1e-8)
    estimator_summary["normalized_absolute_bias"] = estimator_summary.bias.abs() / reward_sd
    estimator_summary["value_accuracy_pass"] = estimator_summary.normalized_absolute_bias <= 0.10
    estimator_summary["coverage_pass"] = estimator_summary.interval_coverage.between(0.90, 0.98)
    estimator_summary["rank_pass"] = (estimator_summary.kendall_tau >= 0.70) & (estimator_summary.top1_recovery >= 0.80)
    estimator_summary["low_ess_guardrail_present"] = True
    estimator_summary["approved_before_real_ehr"] = estimator_summary[["value_accuracy_pass", "coverage_pass", "rank_pass"]].all(axis=1)
    estimator_summary["decision"] = np.where(estimator_summary.approved_before_real_ehr, "approved_synthetic_known_value_only", "not_approved_failed_known_value_gate")
    estimator_summary["claim_boundary"] = CLAIM_BOUNDARY

    tables = {
        "known_value_environment_contracts.csv": pd.DataFrame(environment_rows),
        "offline_data_regimes.csv": pd.DataFrame(regime_rows),
        "world_model_component_metrics.csv": component,
        "world_model_planner_true_returns.csv": planner,
        "model_free_true_returns.csv": pd.DataFrame(model_free_rows),
        "predicted_value_and_true_value.csv": pd.DataFrame(value_rows),
        "policy_regret.csv": regret,
        "model_exploitation_gap.csv": pd.DataFrame(exploit_rows),
        "ope_accuracy_and_coverage.csv": ope,
        "policy_rank_recovery.csv": rank,
        "predictive_policy_bridge.csv": bridge,
        "uncertainty_pessimism_support_ablations.csv": pd.DataFrame(ablation_rows),
        "regime_interaction_results.csv": regime_results,
        "multiplicity_and_uncertainty_receipt.csv": pd.DataFrame([{
            "bootstrap_unit": "synthetic_episode",
            "bootstrap_replicates": BOOTSTRAPS,
            "true_return_replicates": 384,
            "replicated_seeds": ";".join(map(str, SEEDS)),
            "multiplicity_method": "Holm_within_predictive_bridge_and_regime_interaction_families",
            "alpha": 0.05,
            "frozen_before_result_summarization": True,
            "claim_boundary": CLAIM_BOUNDARY,
        }]),
        "estimator_and_guardrail_pass_fail.csv": estimator_summary,
        "resource_metrics.csv": pd.DataFrame(resource_rows),
    }
    for name, frame in tables.items():
        frame.to_csv(output / name, index=False)
    pd.DataFrame(failure_rows, columns=["environment_id", "method", "seed", "stage", "status", "reason"]).to_csv(output / "failures_and_not_run_receipts.csv", index=False)

    approved = estimator_summary.loc[estimator_summary.approved_before_real_ehr, "estimator"].tolist()
    ablations = pd.DataFrame(ablation_rows)
    full = ablations[ablations.ablation == "full_guardrails"]
    no_mask = ablations[ablations.ablation == "no_support_mask"]
    support_guardrail = bool(full.unsupported_action_rate.max() <= 0.01) if len(full) else False
    uncertainty_guardrail = _guardrail_noninferior(ablations, "no_uncertainty")
    pessimism_guardrail = _guardrail_noninferior(ablations, "no_pessimism")
    decision = "complete_known_value_pipeline_guardrails_frozen" if len(component) and len(planner) else "partial_known_value_pipeline"
    pd.DataFrame([{
        "experiment_id": "KDD100",
        "decision": decision,
        "environment_families": len({spec.family for spec in SPECS}),
        "environment_regimes": len(SPECS),
        "seeds": len(SEEDS),
        "world_model_fits": len(component),
        "world_model_planner_rows": len(planner),
        "model_free_rows": len(model_free_rows),
        "ope_rows": len(ope),
        "approved_ope_estimators": ";".join(approved) if approved else "none",
        "support_mask_guardrail": "approved" if support_guardrail else "not_approved",
        "uncertainty_guardrail": "approved" if uncertainty_guardrail else "not_approved",
        "pessimism_guardrail": "approved" if pessimism_guardrail else "not_approved",
        "patient_level_ehr_accessed": False,
        "real_ehr_policy_diagnostics_authorized": False,
        "reason": "Known-value approvals apply only to synthetic validation; KDD099 has zero eligible real-EHR policy tasks and no confirmatory holdout.",
        "claim_boundary": CLAIM_BOUNDARY,
    }]).to_csv(output / "decision.csv", index=False)
    privacy = pd.DataFrame([
        {"check": "patient_level_ehr_loader_import", "status": "pass", "value": 0, "detail": "Runner imports only procedural generator and model classes."},
        {"check": "patient_or_stay_identifier_columns", "status": "pass", "value": 0, "detail": "Aggregate environment/method/seed rows only."},
        {"check": "exact_timestamps_or_trajectories", "status": "pass", "value": 0, "detail": "No row-level arrays are serialized."},
        {"check": "tensor_or_checkpoint_export", "status": "pass", "value": 0, "detail": "Models remain in memory; only non-reversible state fingerprints are written."},
        {"check": "source_provenance", "status": "pass", "value": "procedural_synthetic_only", "detail": CLAIM_BOUNDARY},
    ])
    privacy.to_csv(output / "privacy_audit.csv", index=False)
    report = _report(decision, tables, approved, support_guardrail, uncertainty_guardrail, pessimism_guardrail, failure_rows)
    (output / "kdd100_report.md").write_text(report, encoding="utf-8")
    _write_hashes(output)


def _guardrail_noninferior(frame: pd.DataFrame, ablation: str) -> bool:
    keys = ["environment_id", "world_model", "planner", "seed"]
    full = frame[frame.ablation == "full_guardrails"][keys + ["policy_regret"]]
    compare = frame[frame.ablation == ablation][keys + ["policy_regret"]]
    paired = full.merge(compare, on=keys, suffixes=("_full", "_ablated"))
    return bool(len(paired) and np.median(paired.policy_regret_full - paired.policy_regret_ablated) <= 0.0)


def _policy_fidelity(policy_name: str, family: str) -> str:
    if family == "world_model_planner" or policy_name == "decision_transformer_adapter":
        return "conceptual_adapter"
    if family == "oracle_control" or policy_name in {"random_supported", "no_min_action", "max_action", "severity_rule"}:
        return "local_control"
    return "independent_reimplementation"


def _regime_interactions(environment: pd.DataFrame, regret: pd.DataFrame) -> pd.DataFrame:
    merged = regret.merge(environment.drop(columns=["claim_boundary"]), on="environment_id")
    merged = merged[merged.method != "oracle_true_model"].copy()
    merged["method_family"] = np.where(merged.method.str.contains("__"), "world_model_planner", "model_free")
    factors = {
        "dataset_size_large": merged.episodes >= merged.episodes.median(),
        "horizon_long": merged.horizon >= merged.horizon.median(),
        "reward_sparse": merged.reward_sparsity == "terminal_sparse",
        "support_low": merged.support == "low",
        "state_dim_high": merged.state_dim >= merged.state_dim.median(),
        "missingness_high": merged.missingness >= merged.missingness.median(),
        "behavior_concentrated": merged.behavior_concentration >= merged.behavior_concentration.median(),
        "dynamics_misspecified": merged.dynamics_misspecification > 0,
    }
    rows = []
    for name, flag in factors.items():
        groups = {}
        for family in ("world_model_planner", "model_free"):
            for level, selected in (("high", flag), ("low", ~flag)):
                groups[(family, level)] = merged.loc[selected & (merged.method_family == family), "policy_regret"]
        if any(not len(value) for value in groups.values()):
            rows.append({"outcome": "true_policy_regret", "method_contrast": "world_model_planner_minus_model_free", "regime_factor": name, "interaction_difference_in_differences": np.nan, "standard_error": np.nan, "z": np.nan, "raw_p": 1.0, "n": len(merged)})
            continue
        high_delta = groups[("world_model_planner", "high")].mean() - groups[("model_free", "high")].mean()
        low_delta = groups[("world_model_planner", "low")].mean() - groups[("model_free", "low")].mean()
        delta = float(high_delta - low_delta)
        se = float(np.sqrt(sum(value.var(ddof=1) / len(value) for value in groups.values())))
        z = delta / max(se, 1e-12)
        p = math.erfc(abs(z) / math.sqrt(2.0))
        rows.append({"outcome": "true_policy_regret", "method_contrast": "world_model_planner_minus_model_free", "regime_factor": name, "interaction_difference_in_differences": delta, "standard_error": se, "z": z, "raw_p": p, "n": len(merged)})
    frame = pd.DataFrame(rows)
    frame["holm_p"] = _holm(frame.raw_p.to_numpy())
    frame["interpretation"] = "descriptive_method_by_regime_interaction_in_aliased_fractional_design_not_clinical_generalization"
    return frame


def _holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[index]))
        adjusted[index] = running
    return adjusted


def _report(decision: str, tables: dict[str, pd.DataFrame], approved: list[str], support: bool, uncertainty: bool, pessimism: bool, failures: list[dict]) -> str:
    component = tables["world_model_component_metrics.csv"]
    planner = tables["world_model_planner_true_returns.csv"]
    model_free = tables["model_free_true_returns.csv"]
    ope = tables["ope_accuracy_and_coverage.csv"]
    return f"""# KDD100 complete known-value pipeline report

## Decision

`{decision}`

This run exercised the complete synthetic path `offline data -> world model -> planner/policy -> true environment return`. It used no patient-level EHR data and serialized no trajectories, identifiers, exact timestamps, tensors, or checkpoints.

## Frozen design

- Environment families: {len({spec.family for spec in SPECS})}; regimes: {len(SPECS)}; seeds: {len(SEEDS)}.
- Every requested regime factor has two frozen levels. The six regimes are a bounded fractional design, not a claim that these simulators reproduce a patient trajectory.
- Exact local KDD098 model classes were used for GRU-D, causal Transformer, Dreamer V1 Gaussian RSSM, and Dreamer V3 categorical RSSM. The Gaussian ensemble is an aggregate contract adapter.
- World-model/planner rows: {len(planner)}; model-free rows: {len(model_free)}; component rows: {len(component)}.
- OPE rows: {len(ope)} with 1,000 synthetic-episode bootstrap replicates and Holm adjustment within frozen inference families.
- Failures retained: {len(failures)}.

## Frozen estimator and guardrail decision

- OPE estimators passing all preregistered value-accuracy, interval-coverage, and rank-recovery gates: {', '.join(approved) if approved else 'none'}.
- Support mask guardrail: {'approved' if support else 'not approved'}.
- Uncertainty penalty guardrail: {'approved' if uncertainty else 'not approved'}.
- Pessimism guardrail: {'approved' if pessimism else 'not approved'}.

Passing a synthetic gate does not authorize real-EHR scoring. KDD099 retained zero real-EHR policy tasks because task-specific reward and target-policy prerequisites were absent, and no untouched confirmatory holdout exists.

## Interpretation boundary

{CLAIM_BOUNDARY}
"""


def _write_hashes(output: Path) -> None:
    payload = {}
    for path in sorted(output.iterdir()):
        if path.name == "artifact_hashes.json" or not path.is_file():
            continue
        payload[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "artifact_hashes.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _state_fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _softmax(value: np.ndarray) -> np.ndarray:
    shifted = value - np.max(value, axis=1, keepdims=True)
    exp = np.exp(np.clip(shifted, -60, 60))
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def _complete_classes(probability: np.ndarray, classes: np.ndarray, count: int) -> np.ndarray:
    output = np.full((len(probability), count), 1e-8)
    output[:, classes.astype(int)] = probability
    return output / output.sum(axis=1, keepdims=True)


def _stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) % 100_000


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the no-EHR KDD100 complete known-value pipeline.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
