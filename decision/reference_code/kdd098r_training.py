from __future__ import annotations

import copy
import hashlib
import io
import math
import resource
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .kdd069_model_types import parameter_count
from .kdd098_data import TaskSequences, deterministic_training_indices
from .kdd098_training import (
    Surface,
    _make_model,
    _one_step,
    _recursive,
    _scale_calibration,
    fit_termination_head,
    termination_probability,
)


@dataclass(slots=True)
class RFitReceipt:
    task: str
    method: str
    seed: int
    selected_epoch: int
    trained_epochs: int
    epoch_rmse: list[float]
    epoch_mae: list[float]
    epoch_composite: list[float]
    cap_hit: bool
    improving_at_cap: bool
    early_stop: bool
    fit_episodes: int
    calibration_episodes: int
    validation_episodes: int
    checkpoint_opportunities: int
    checkpoint_sha256: str


@dataclass(slots=True)
class RSurface:
    surface: Surface
    previous_action_one: np.ndarray
    matched_shuffled_one: np.ndarray
    matched_shuffled_recursive: np.ndarray
    termination_native: np.ndarray
    termination_recalibrated: np.ndarray
    termination_recalibration_slope: float
    termination_recalibration_intercept: float


def fit_world_model(
    task: TaskSequences,
    method: str,
    seed: int,
    budget: dict[str, object],
    epoch_cap: int,
    checkpoint_root: Path,
) -> tuple[RSurface, RFitReceipt]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    fit_idx, cal_idx = deterministic_training_indices(task, int(budget["max_train_episodes_per_task"]), 3408)
    val_idx = np.flatnonzero(task.episodes("validation"))
    model = _make_model(method, task.states.shape[-1], task.action_dim, budget)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(budget["learning_rate"]), weight_decay=float(budget["weight_decay"]))
    dataset = TensorDataset(
        torch.from_numpy(task.states[fit_idx]),
        torch.from_numpy(task.state_masks[fit_idx].astype(np.float32)),
        torch.from_numpy(task.deltas[fit_idx]),
        torch.from_numpy(task.actions[fit_idx]),
        torch.from_numpy(task.targets[fit_idx]),
        torch.from_numpy(task.target_masks[fit_idx].astype(np.float32)),
        torch.from_numpy(task.valid_steps[fit_idx].astype(np.float32)),
    )
    loader = DataLoader(dataset, batch_size=int(budget["batch_size"]), shuffle=True, generator=torch.Generator().manual_seed(seed))
    best_state = None
    best_score = math.inf
    best_epoch = 0
    stale = 0
    rmses: list[float] = []
    maes: list[float] = []
    composites: list[float] = []
    started = time.perf_counter()
    min_epochs = int(budget["min_epochs"])
    patience = int(budget["patience"])
    relative_floor = float(budget["minimum_relative_validation_improvement"])
    for epoch in range(1, epoch_cap + 1):
        model.train()
        for states, masks, deltas, actions, targets, target_masks, valid in loader:
            optimizer.zero_grad()
            prediction = model(states, masks, deltas, actions)
            observed = target_masks * valid.unsqueeze(-1)
            scale = torch.exp(prediction.log_scale)
            nll = prediction.log_scale + 0.5 * torch.square((targets - prediction.mean) / scale)
            loss = (nll * observed).sum() / torch.clamp(observed.sum(), min=1.0) + 0.01 * prediction.auxiliary_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(budget["gradient_clip_norm"]))
            optimizer.step()
        rmse, mae = _validation_errors(model, task, val_idx, int(budget["batch_size"]))
        composite = 0.7 * rmse + 0.3 * mae
        rmses.append(rmse); maes.append(mae); composites.append(composite)
        relative = (best_score - composite) / max(abs(best_score), 1.0e-12) if math.isfinite(best_score) else math.inf
        if relative >= relative_floor:
            best_score = composite
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch >= min_epochs and stale >= patience:
            break
    if best_state is None:
        raise RuntimeError(f"no finite KDD098R checkpoint for {task.task}/{method}/{seed}")
    trained_epochs = len(composites)
    cap_hit = trained_epochs == epoch_cap
    improving = cap_hit and trained_epochs > 1 and ((composites[-2] - composites[-1]) / max(abs(composites[-2]), 1.0e-12) >= relative_floor)
    model.load_state_dict(best_state)
    scale_factor = _scale_calibration(model, task, cal_idx, int(budget["batch_size"]))
    term_head = fit_termination_head(task, fit_idx, seed)
    calibration_state, _ = _one_step(model, task, cal_idx, task.actions[cal_idx], scale_factor, int(budget["batch_size"]))
    calibration_native = termination_probability(calibration_state, task.actions[cal_idx], task.valid_steps[cal_idx], term_head)
    slope, intercept = _fit_platt(task.terminal[cal_idx][task.valid_steps[cal_idx]], calibration_native[task.valid_steps[cal_idx]])

    val_actions = task.actions[val_idx]
    no_actions = np.zeros_like(val_actions); no_actions[..., 0] = task.valid_steps[val_idx]
    matched_actions = matched_action_shuffle(task, val_idx, seed)
    previous_actions = previous_action_array(val_actions, task.valid_steps[val_idx])
    one, one_scale = _one_step(model, task, val_idx, val_actions, scale_factor, int(budget["batch_size"]))
    recursive, recursive_scale = _recursive(model, task, val_idx, val_actions, scale_factor, int(budget["batch_size"]))
    no_one, _ = _one_step(model, task, val_idx, no_actions, scale_factor, int(budget["batch_size"]))
    no_recursive, _ = _recursive(model, task, val_idx, no_actions, scale_factor, int(budget["batch_size"]))
    matched_one, _ = _one_step(model, task, val_idx, matched_actions, scale_factor, int(budget["batch_size"]))
    matched_recursive, _ = _recursive(model, task, val_idx, matched_actions, scale_factor, int(budget["batch_size"]))
    previous_one, _ = _one_step(model, task, val_idx, previous_actions, scale_factor, int(budget["batch_size"]))
    native = termination_probability(one, val_actions, task.valid_steps[val_idx], term_head)
    recalibrated = _apply_platt(native, slope, intercept)

    checkpoint = {
        "experiment_id": "KDD098R", "task": task.task, "method": method, "seed": seed,
        "selected_epoch": best_epoch, "trained_epochs": trained_epochs, "epoch_cap": epoch_cap,
        "model_state": model.state_dict(), "scale_factor": torch.from_numpy(scale_factor),
        "termination_weight": torch.from_numpy(term_head[0]), "termination_bias": term_head[1],
        "termination_recalibration_slope": slope, "termination_recalibration_intercept": intercept,
        "preprocessing_sha256": task.preprocessing_sha256,
    }
    payload_io = io.BytesIO(); torch.save(checkpoint, payload_io); payload = payload_io.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    (checkpoint_root / f"{task.task}__{method}__seed{seed}.pt").write_bytes(payload)
    elapsed = time.perf_counter() - started
    surface = Surface(
        task.task, method, seed, one, one_scale, recursive, recursive_scale,
        no_one, no_recursive, matched_one, matched_recursive, native,
        parameter_count(model) + len(term_head[0]) + 3, elapsed, 0.0,
        float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
        "cap_hit_still_improving" if improving else ("cap_hit_plateau" if cap_hit else "early_stopped"), digest,
    )
    wrapped = RSurface(surface, previous_one, matched_one, matched_recursive, native, recalibrated, slope, intercept)
    receipt = RFitReceipt(
        task.task, method, seed, best_epoch, trained_epochs, rmses, maes, composites, cap_hit, improving,
        trained_epochs < epoch_cap, len(fit_idx), len(cal_idx), len(val_idx), trained_epochs, digest,
    )
    return wrapped, receipt


def matched_action_shuffle(task: TaskSequences, indices: np.ndarray, seed: int) -> np.ndarray:
    actions = task.actions[indices]
    valid = task.valid_steps[indices]
    train = np.flatnonzero(task.episodes("train"))
    severity_train = np.nanmean(np.abs(task.states[train, :, :8]), axis=-1)
    severity_train = severity_train[task.valid_steps[train]]
    cuts = np.quantile(severity_train, [1 / 3, 2 / 3]) if len(severity_train) else np.array([0.0, 1.0])
    severity = np.digitize(np.nanmean(np.abs(task.states[indices, :, :8]), axis=-1), cuts)
    counts = np.bincount(task.action_classes[train][task.valid_steps[train]].clip(min=0), minlength=task.action_dim)
    propensity = counts[task.action_classes[indices].clip(min=0)] / max(counts.sum(), 1)
    prop_bucket = np.digitize(propensity, [0.01, 0.05])
    output = actions.copy()
    rng = np.random.default_rng(seed + 1901)
    for step in range(actions.shape[1]):
        for sev in range(3):
            for prop in range(3):
                rows = np.flatnonzero(valid[:, step] & (severity[:, step] == sev) & (prop_bucket[:, step] == prop))
                if len(rows) > 1:
                    output[rows, step] = actions[rng.permutation(rows), step]
    return output


def previous_action_array(actions: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = actions.copy()
    output[:, 1:] = actions[:, :-1]
    return output


def _validation_errors(model, task: TaskSequences, indices: np.ndarray, batch_size: int) -> tuple[float, float]:
    mean, _ = _one_step(model, task, indices, task.actions[indices], None, batch_size)
    mask = task.target_masks[indices] & task.valid_steps[indices, :, None]
    error = task.targets[indices][mask] - mean[mask]
    return float(np.sqrt(np.mean(np.square(error)))), float(np.mean(np.abs(error)))


def _fit_platt(labels: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    y = np.asarray(labels, dtype=np.float64)
    p = np.clip(np.asarray(probability, dtype=np.float64), 1.0e-5, 1.0 - 1.0e-5)
    x = np.log(p / (1.0 - p))
    slope, intercept = 1.0, 0.0
    for _ in range(200):
        q = 1.0 / (1.0 + np.exp(-np.clip(slope * x + intercept, -20.0, 20.0)))
        error = q - y
        slope -= 0.02 * float(np.mean(error * x))
        intercept -= 0.02 * float(np.mean(error))
    return float(slope), float(intercept)


def _apply_platt(probability: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    p = np.clip(probability, 1.0e-5, 1.0 - 1.0e-5)
    logits = np.log(p / (1.0 - p))
    output = 1.0 / (1.0 + np.exp(-np.clip(slope * logits + intercept, -20.0, 20.0)))
    return output.astype(np.float32)
