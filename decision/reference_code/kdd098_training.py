from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import resource
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .kdd069_model_types import TransitionOutput, parameter_count
from .kdd069_rssm_models import DreamerV1GaussianRSSM, DreamerV3CategoricalRSSM
from .kdd069_sequence_models import CausalTransformerTransition, GRUDTransition
from .kdd098_data import TaskSequences, deterministic_training_indices


METHOD_TO_MODEL = {
    "grud_world_model": "grud",
    "transformer_world_model": "transformer",
    "dreamer_v1_gaussian_rssm": "dreamer_v1",
    "dreamer_v3_categorical_rssm": "dreamer_v3",
}


@dataclass(slots=True)
class Surface:
    task: str
    method: str
    seed: int | str
    one_mean: np.ndarray
    one_scale: np.ndarray | None
    recursive_mean: np.ndarray
    recursive_scale: np.ndarray | None
    no_action_one: np.ndarray
    no_action_recursive: np.ndarray
    shuffled_one: np.ndarray
    shuffled_recursive: np.ndarray
    termination_probability: np.ndarray
    parameter_count: int
    training_seconds: float
    inference_seconds: float
    peak_memory_mb: float
    convergence_status: str
    checkpoint_sha256: str


@dataclass(slots=True)
class FitReceipt:
    task: str
    method: str
    seed: int
    selected_epoch: int
    validation_rmse: float
    epoch_validation_rmse: str
    train_fit_episodes: int
    train_calibration_episodes: int
    validation_episodes: int
    checkpoint_sha256: str


def fit_neural_world_model(
    task: TaskSequences,
    method: str,
    seed: int,
    budget: dict[str, object],
    checkpoint_root: Path,
) -> tuple[Surface, FitReceipt]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    fit_indices, calibration_indices = deterministic_training_indices(
        task, int(budget["max_train_episodes_per_task"]), 3408
    )
    validation_indices = np.flatnonzero(task.episodes("validation"))
    model = _make_model(method, task.states.shape[-1], task.action_dim, budget)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(budget["learning_rate"]),
        weight_decay=float(budget["weight_decay"]),
    )
    dataset = TensorDataset(
        torch.from_numpy(task.states[fit_indices]),
        torch.from_numpy(task.state_masks[fit_indices].astype(np.float32)),
        torch.from_numpy(task.deltas[fit_indices]),
        torch.from_numpy(task.actions[fit_indices]),
        torch.from_numpy(task.targets[fit_indices]),
        torch.from_numpy(task.target_masks[fit_indices].astype(np.float32)),
        torch.from_numpy(task.valid_steps[fit_indices].astype(np.float32)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(budget["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_rmse = math.inf
    best_epoch = 0
    epoch_scores: list[float] = []
    train_start = time.perf_counter()
    for epoch in range(1, int(budget["epochs"]) + 1):
        model.train()
        for states, masks, deltas, actions, targets, target_masks, valid in loader:
            optimizer.zero_grad()
            output = model(states, masks, deltas, actions)
            observed = target_masks * valid.unsqueeze(-1)
            scale = torch.exp(output.log_scale)
            nll = output.log_scale + 0.5 * torch.square((targets - output.mean) / scale)
            loss = (nll * observed).sum() / torch.clamp(observed.sum(), min=1.0)
            loss = loss + 0.01 * output.auxiliary_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(budget["gradient_clip_norm"]))
            optimizer.step()
        score = _validation_rmse(model, task, validation_indices, int(budget["batch_size"]))
        epoch_scores.append(score)
        if score < best_rmse - 1.0e-12:
            best_rmse = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError(f"no finite checkpoint selected for {task.task}/{method}/{seed}")
    model.load_state_dict(best_state)
    training_seconds = time.perf_counter() - train_start
    scale_factor = _scale_calibration(model, task, calibration_indices, int(budget["batch_size"]))
    termination = fit_termination_head(task, fit_indices, seed)

    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "experiment_id": "KDD098",
        "task": task.task,
        "method": method,
        "seed": seed,
        "selected_epoch": best_epoch,
        "model_state": model.state_dict(),
        "scale_factor": torch.from_numpy(scale_factor),
        "termination_weight": torch.from_numpy(termination[0]),
        "termination_bias": float(termination[1]),
        "preprocessing_sha256": task.preprocessing_sha256,
    }
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    payload = buffer.getvalue()
    checkpoint_sha = hashlib.sha256(payload).hexdigest()
    checkpoint_path = checkpoint_root / f"{task.task}__{method}__seed{seed}.pt"
    checkpoint_path.write_bytes(payload)

    inference_start = time.perf_counter()
    observed_actions = task.actions[validation_indices]
    no_actions = np.zeros_like(observed_actions)
    no_actions[..., 0] = task.valid_steps[validation_indices]
    shuffled = _within_role_shuffle(observed_actions, task.valid_steps[validation_indices], seed)
    one_mean, one_scale = _one_step(model, task, validation_indices, observed_actions, scale_factor, int(budget["batch_size"]))
    recursive_mean, recursive_scale = _recursive(model, task, validation_indices, observed_actions, scale_factor, int(budget["batch_size"]))
    no_one, _ = _one_step(model, task, validation_indices, no_actions, scale_factor, int(budget["batch_size"]))
    no_recursive, _ = _recursive(model, task, validation_indices, no_actions, scale_factor, int(budget["batch_size"]))
    shuffled_one, _ = _one_step(model, task, validation_indices, shuffled, scale_factor, int(budget["batch_size"]))
    shuffled_recursive, _ = _recursive(model, task, validation_indices, shuffled, scale_factor, int(budget["batch_size"]))
    terminal_probability = termination_probability(one_mean, observed_actions, task.valid_steps[validation_indices], termination)
    inference_seconds = time.perf_counter() - inference_start
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
    surface = Surface(
        task.task,
        method,
        seed,
        one_mean,
        one_scale,
        recursive_mean,
        recursive_scale,
        no_one,
        no_recursive,
        shuffled_one,
        shuffled_recursive,
        terminal_probability,
        parameter_count(model) + len(termination[0]) + 1,
        training_seconds,
        inference_seconds,
        peak,
        "selected_at_frozen_budget_boundary" if best_epoch == int(budget["epochs"]) else "selected_before_frozen_budget_boundary",
        checkpoint_sha,
    )
    receipt = FitReceipt(
        task.task,
        method,
        seed,
        best_epoch,
        best_rmse,
        ";".join(f"{value:.8f}" for value in epoch_scores),
        len(fit_indices),
        len(calibration_indices),
        len(validation_indices),
        checkpoint_sha,
    )
    return surface, receipt


def persistence_surface(task: TaskSequences) -> Surface:
    indices = np.flatnonzero(task.episodes("validation"))
    states = task.states[indices]
    valid = task.valid_steps[indices]
    one = states.copy()
    recursive = np.repeat(states[:, :1], states.shape[1], axis=1)
    actions = task.actions[indices]
    termination = fit_termination_head(task, np.flatnonzero(task.episodes("train")), 3408)
    terminal_probability = termination_probability(one, actions, valid, termination)
    return Surface(
        task.task,
        "persistence_locf",
        3408,
        one,
        None,
        recursive,
        None,
        one.copy(),
        recursive.copy(),
        one.copy(),
        recursive.copy(),
        terminal_probability,
        len(termination[0]) + 1,
        0.0,
        0.0,
        float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
        "deterministic_control",
        "not_applicable_control",
    )


def fit_hgb_surface(task: TaskSequences, seed: int, budget: dict[str, object]) -> Surface:
    from sklearn.ensemble import HistGradientBoostingRegressor

    start = time.perf_counter()
    fit_indices, calibration_indices = deterministic_training_indices(
        task, int(budget["max_train_episodes_per_task"]), 3408
    )
    x, residual, observed = _tabular_training_rows(task, fit_indices)
    rng = np.random.default_rng(seed)
    maximum = int(budget["hgb_max_train_rows"])
    if len(x) > maximum:
        keep = np.sort(rng.choice(len(x), size=maximum, replace=False))
        x, residual, observed = x[keep], residual[keep], observed[keep]
    models: list[HistGradientBoostingRegressor | None] = []
    for feature in range(task.states.shape[-1]):
        keep = observed[:, feature]
        if not keep.any():
            models.append(None)
            continue
        model = HistGradientBoostingRegressor(
            max_iter=int(budget["hgb_max_iter"]),
            learning_rate=0.06,
            max_leaf_nodes=int(budget["hgb_max_leaf_nodes"]),
            l2_regularization=1.0e-3,
            random_state=seed + feature,
        )
        model.fit(x[keep], residual[keep, feature])
        models.append(model)
    training_seconds = time.perf_counter() - start
    validation = np.flatnonzero(task.episodes("validation"))
    observed_actions = task.actions[validation]
    no_actions = np.zeros_like(observed_actions)
    no_actions[..., 0] = task.valid_steps[validation]
    shuffled = _within_role_shuffle(observed_actions, task.valid_steps[validation], seed)
    inference_start = time.perf_counter()
    one = _hgb_one(task, validation, observed_actions, models)
    recursive = _hgb_recursive(task, validation, observed_actions, models)
    no_one = _hgb_one(task, validation, no_actions, models)
    no_recursive = _hgb_recursive(task, validation, no_actions, models)
    shuffled_one = _hgb_one(task, validation, shuffled, models)
    shuffled_recursive = _hgb_recursive(task, validation, shuffled, models)
    cal_one = _hgb_one(task, calibration_indices, task.actions[calibration_indices], models)
    cal_mask = task.target_masks[calibration_indices] & task.valid_steps[calibration_indices, :, None]
    scale = np.ones(task.states.shape[-1], dtype=np.float32)
    for feature in range(len(scale)):
        keep = cal_mask[..., feature]
        if keep.any():
            scale[feature] = max(float(np.sqrt(np.mean(np.square(task.targets[calibration_indices, :, feature][keep] - cal_one[:, :, feature][keep])))), 1.0e-3)
    one_scale = np.broadcast_to(scale, one.shape).copy()
    recursive_scale = np.broadcast_to(scale, recursive.shape).copy()
    termination = fit_termination_head(task, fit_indices, seed)
    terminal_probability = termination_probability(one, observed_actions, task.valid_steps[validation], termination)
    inference_seconds = time.perf_counter() - inference_start
    return Surface(
        task.task,
        "hgb_residual",
        seed,
        one,
        one_scale,
        recursive,
        recursive_scale,
        no_one,
        no_recursive,
        shuffled_one,
        shuffled_recursive,
        terminal_probability,
        len(models) * int(budget["hgb_max_iter"]) * int(budget["hgb_max_leaf_nodes"]),
        training_seconds,
        inference_seconds,
        float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
        "sanity_control_complete",
        "not_applicable_sanity_control",
    )


def gaussian_ensemble_surface(task: TaskSequences, members: list[Surface], checkpoint_root: Path) -> Surface:
    if len(members) != 3:
        raise RuntimeError("Gaussian ensemble requires the three frozen GRU-D seeds")
    def combine(name: str) -> tuple[np.ndarray, np.ndarray]:
        means = np.stack([getattr(member, name) for member in members])
        member_scales = [member.one_scale if name == "one_mean" else member.recursive_scale for member in members]
        scales = np.stack([np.maximum(value, 1.0e-4) for value in member_scales])
        mean = means.mean(axis=0)
        variance = (np.square(scales) + np.square(means)).mean(axis=0) - np.square(mean)
        return mean.astype(np.float32), np.sqrt(np.maximum(variance, 1.0e-6)).astype(np.float32)
    one, one_scale = combine("one_mean")
    recursive, recursive_scale = combine("recursive_mean")
    bundle = {
        "experiment_id": "KDD098",
        "task": task.task,
        "method": "gaussian_transition_ensemble",
        "seeds": [3408, 3411, 3414],
        "member_checkpoint_sha256": [member.checkpoint_sha256 for member in members],
        "aggregation": "equal_weight_moment_matched_gaussian",
        "preprocessing_sha256": task.preprocessing_sha256,
    }
    payload = (json.dumps(bundle, sort_keys=True, separators=(",", ":")) + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    (checkpoint_root / f"{task.task}__gaussian_transition_ensemble__bundle.json").write_bytes(payload)
    return Surface(
        task.task,
        "gaussian_transition_ensemble",
        "3408;3411;3414",
        one,
        one_scale,
        recursive,
        recursive_scale,
        np.mean(np.stack([member.no_action_one for member in members]), axis=0).astype(np.float32),
        np.mean(np.stack([member.no_action_recursive for member in members]), axis=0).astype(np.float32),
        np.mean(np.stack([member.shuffled_one for member in members]), axis=0).astype(np.float32),
        np.mean(np.stack([member.shuffled_recursive for member in members]), axis=0).astype(np.float32),
        np.mean(np.stack([member.termination_probability for member in members]), axis=0).astype(np.float32),
        sum(member.parameter_count for member in members),
        sum(member.training_seconds for member in members),
        sum(member.inference_seconds for member in members),
        max(member.peak_memory_mb for member in members),
        "derived_from_three_frozen_grud_members",
        digest,
    )


def _make_model(method: str, feature_dim: int, action_dim: int, budget: dict[str, object]) -> nn.Module:
    hidden = int(budget["hidden_dim"])
    latent = int(budget["latent_dim"])
    kind = METHOD_TO_MODEL[method]
    if kind == "grud":
        return GRUDTransition(feature_dim, action_dim, hidden)
    if kind == "transformer":
        return CausalTransformerTransition(feature_dim, action_dim, hidden)
    if kind == "dreamer_v1":
        return DreamerV1GaussianRSSM(feature_dim, action_dim, hidden, latent)
    if kind == "dreamer_v3":
        return DreamerV3CategoricalRSSM(feature_dim, action_dim, hidden, max(2, latent // 4), 4)
    raise ValueError(method)


def _validation_rmse(model: nn.Module, task: TaskSequences, indices: np.ndarray, batch_size: int) -> float:
    mean, _scale = _one_step(model, task, indices, task.actions[indices], None, batch_size)
    mask = task.target_masks[indices] & task.valid_steps[indices, :, None]
    return float(np.sqrt(np.mean(np.square(task.targets[indices][mask] - mean[mask]))))


def _scale_calibration(model: nn.Module, task: TaskSequences, indices: np.ndarray, batch_size: int) -> np.ndarray:
    mean, scale = _one_step(model, task, indices, task.actions[indices], None, batch_size)
    mask = task.target_masks[indices] & task.valid_steps[indices, :, None]
    factor = np.ones(task.states.shape[-1], dtype=np.float32)
    for feature in range(len(factor)):
        keep = mask[..., feature]
        if keep.any():
            standardized = (task.targets[indices, :, feature][keep] - mean[:, :, feature][keep]) / np.maximum(scale[:, :, feature][keep], 1.0e-4)
            factor[feature] = np.clip(float(np.sqrt(np.mean(np.square(standardized)))), 0.25, 4.0)
    return factor


def _one_step(model, task, indices, actions, factor, batch_size):
    means: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            output = model(
                torch.from_numpy(task.states[batch]),
                torch.from_numpy(task.state_masks[batch].astype(np.float32)),
                torch.from_numpy(task.deltas[batch]),
                torch.from_numpy(actions[start : start + len(batch)]),
            )
            means.append(output.mean.cpu().numpy())
            scales.append(np.exp(output.log_scale.cpu().numpy()))
    mean = np.concatenate(means).astype(np.float32)
    scale = np.concatenate(scales).astype(np.float32)
    if factor is not None:
        scale *= factor.reshape(1, 1, -1)
    return mean, scale


def _recursive(model, task, indices, actions, factor, batch_size):
    means: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            action = torch.from_numpy(actions[start : start + len(batch)])
            initial = torch.from_numpy(task.states[batch, 0])
            initial_mask = torch.from_numpy(task.state_masks[batch, 0].astype(np.float32))
            initial_delta = torch.from_numpy(task.deltas[batch, 0])
            if isinstance(model, (DreamerV1GaussianRSSM, DreamerV3CategoricalRSSM)):
                output = model.rollout(initial, initial_mask, initial_delta, action)
                mean = output.mean
                scale = torch.exp(output.log_scale)
            else:
                values = initial[:, None]
                masks = initial_mask[:, None]
                deltas = initial_delta[:, None]
                step_means: list[torch.Tensor] = []
                step_scales: list[torch.Tensor] = []
                for step in range(action.shape[1]):
                    output = model(values, masks, deltas, action[:, : step + 1])
                    current = output.mean[:, -1]
                    step_means.append(current)
                    step_scales.append(torch.exp(output.log_scale[:, -1]))
                    values = torch.cat([values, current[:, None]], dim=1)
                    masks = torch.cat([masks, torch.zeros_like(current[:, None])], dim=1)
                    deltas = torch.cat([deltas, deltas[:, -1:] + 1.0], dim=1)
                mean = torch.stack(step_means, dim=1)
                scale = torch.stack(step_scales, dim=1)
            means.append(mean.cpu().numpy())
            scales.append(scale.cpu().numpy())
    mean_array = np.concatenate(means).astype(np.float32)
    scale_array = np.concatenate(scales).astype(np.float32) * factor.reshape(1, 1, -1)
    return mean_array, scale_array


def _within_role_shuffle(actions: np.ndarray, valid: np.ndarray, seed: int) -> np.ndarray:
    output = actions.copy()
    positions = np.argwhere(valid)
    order = np.random.default_rng(seed).permutation(len(positions))
    values = actions[positions[:, 0], positions[:, 1]][order]
    output[positions[:, 0], positions[:, 1]] = values
    return output


def fit_termination_head(task: TaskSequences, indices: np.ndarray, seed: int) -> tuple[np.ndarray, float]:
    states = task.targets[indices]
    actions = task.actions[indices]
    valid = task.valid_steps[indices]
    horizon = np.broadcast_to(np.linspace(0.0, 1.0, states.shape[1], dtype=np.float32), valid.shape)
    x = np.concatenate([states, actions, horizon[..., None]], axis=-1)[valid]
    y = task.terminal[indices][valid]
    if len(x) > 100_000:
        keep = np.sort(np.random.default_rng(seed).choice(len(x), size=100_000, replace=False))
        x, y = x[keep], y[keep]
    weight = np.zeros(x.shape[1], dtype=np.float64)
    bias = float(np.log((y.mean() + 1.0e-4) / (1.0 - y.mean() + 1.0e-4)))
    positive_weight = max(float((len(y) - y.sum()) / max(y.sum(), 1.0)), 1.0)
    for _step in range(120):
        logits = np.clip(x @ weight + bias, -20.0, 20.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        sample_weight = np.where(y > 0.5, positive_weight, 1.0)
        error = (probability - y) * sample_weight
        weight -= 0.05 * ((x.T @ error) / sample_weight.sum() + 1.0e-4 * weight)
        bias -= 0.05 * float(error.sum() / sample_weight.sum())
    return weight.astype(np.float32), float(bias)


def termination_probability(predicted_state, actions, valid, head):
    horizon = np.broadcast_to(np.linspace(0.0, 1.0, predicted_state.shape[1], dtype=np.float32), valid.shape)
    x = np.concatenate([predicted_state, actions, horizon[..., None]], axis=-1)
    logits = np.clip(x @ head[0] + head[1], -20.0, 20.0)
    probability = 1.0 / (1.0 + np.exp(-logits))
    return np.where(valid, probability, np.nan).astype(np.float32)


def _tabular_training_rows(task: TaskSequences, indices: np.ndarray):
    valid = task.valid_steps[indices]
    horizon = np.broadcast_to(np.linspace(0.0, 1.0, task.states.shape[1], dtype=np.float32), valid.shape)
    x = np.concatenate([task.states[indices], task.state_masks[indices].astype(np.float32), task.deltas[indices], task.actions[indices], horizon[..., None]], axis=-1)[valid]
    residual = (task.targets[indices] - task.states[indices])[valid]
    observed = task.target_masks[indices][valid]
    return x.astype(np.float32), residual.astype(np.float32), observed


def _hgb_predict(models, inputs):
    output = np.zeros((len(inputs), len(models)), dtype=np.float32)
    for feature, model in enumerate(models):
        if model is not None:
            output[:, feature] = model.predict(inputs).astype(np.float32)
    return output


def _hgb_one(task, indices, actions, models):
    valid = task.valid_steps[indices]
    horizon = np.broadcast_to(np.linspace(0.0, 1.0, task.states.shape[1], dtype=np.float32), valid.shape)
    x = np.concatenate([task.states[indices], task.state_masks[indices].astype(np.float32), task.deltas[indices], actions, horizon[..., None]], axis=-1)
    residual = _hgb_predict(models, x.reshape(-1, x.shape[-1])).reshape(task.states[indices].shape)
    return (task.states[indices] + residual).astype(np.float32)


def _hgb_recursive(task, indices, actions, models):
    current = task.states[indices, 0].copy()
    mask = task.state_masks[indices, 0].astype(np.float32)
    delta = task.deltas[indices, 0].copy()
    output: list[np.ndarray] = []
    for step in range(actions.shape[1]):
        horizon = np.full((len(indices), 1), step / max(actions.shape[1] - 1, 1), dtype=np.float32)
        x = np.concatenate([current, mask, delta, actions[:, step], horizon], axis=-1)
        current = current + _hgb_predict(models, x)
        output.append(current.copy())
        mask.fill(0.0)
        delta += 1.0
    return np.stack(output, axis=1).astype(np.float32)
