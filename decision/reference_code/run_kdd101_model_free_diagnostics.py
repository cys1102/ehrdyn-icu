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
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .kdd087a_behavior import _apply_temperature, _knn, _neural, _temperature
from .kdd098_data import TaskSequences, build_kdd098_task_sequences
from .kdd098r_metrics import reward_values
from .run_kdd099_policy_contract import behavior_states


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/kdd101_model_free_diagnostics_v1.json"
MIMIC = Path("<authorized-mimiciv-root>")
KDD095 = ROOT / "kdd_benchmark_discovery/results/kdd095_rich_first_contract_20260714_v3/baseline_catalog.csv"
KDD097 = ROOT / "results/kdd097_rich_task_materialization_20260714_v2"
KDD099RA = ROOT / "kdd_benchmark_discovery/results/kdd099r_a_reward_provenance_20260714_v2"
KDD100R = ROOT / "kdd_benchmark_discovery/results/kdd100r_task_matched_known_value_20260714_v2"
KDD099RB = ROOT / "kdd_benchmark_discovery/results/kdd099r_b_authorization_synthesis_20260714_v2"
CLAIM = (
    "Development-only retrospective support, collapse, divergence, and stability diagnostics. "
    "Real-EHR policy value and policy-winner comparison are unavailable. No treatment-benefit, causal, "
    "counterfactual-validity, clinical-utility, deployment, or autonomous-decision claim is supported."
)


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecisionTransformerAdapter(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden: int, max_steps: int):
        super().__init__()
        self.input = nn.Linear(input_dim + 2, hidden)
        self.position = nn.Parameter(torch.zeros(1, max_steps, hidden))
        layer = nn.TransformerEncoderLayer(hidden, 4, hidden * 2, 0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.head = nn.Linear(hidden, action_dim)

    def forward(self, x: torch.Tensor, rtg: torch.Tensor, reward_known: torch.Tensor) -> torch.Tensor:
        token = self.input(torch.cat([x, rtg[..., None], reward_known[..., None]], dim=-1)) + self.position[:, : x.shape[1]]
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool), diagonal=1)
        return self.head(self.encoder(token, mask=mask))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KDD101 development-only model-free diagnostics")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mimiciv-root", type=Path, default=MIMIC)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    args = parser.parse_args()
    config = load_config(args.config)
    output = args.output or ROOT / config["output_directory"]
    if output.exists():
        raise FileExistsError(output)
    verify_contract(config)
    old_receipts = historical_receipt_hashes()

    tasks = [task for task in build_kdd098_task_sequences(args.mimiciv_root, args.chunksize) if task.task in config["authorized_tasks"]]
    if {task.task for task in tasks} != set(config["authorized_tasks"]):
        raise RuntimeError("authorized KDD101 task inventory drift")

    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    probabilities: dict[tuple[str, str, int], dict[str, Any]] = {}
    for task in tasks:
        denominator = fit_behavior_denominators(task, config)
        support = supported_actions(task)
        severity = severity_contract(task, support)
        for seed in config["seeds"]:
            learned = train_learned_policies(task, denominator, support, config, seed, rows)
            controls = control_probabilities(task, denominator, support, severity, config, seed)
            for method, result in {**learned, **controls}.items():
                probabilities[(task.task, method, seed)] = result
                audit_policy(task, method, seed, result, denominator, support, config, rows)

    add_method_coverage(config, rows)
    add_seed_and_cross_cohort(probabilities, config, rows)
    add_blocked_ope_receipts(probabilities, config, rows)
    add_empty_stress_receipt(rows)
    add_privacy_and_decision(probabilities, config, rows)
    if not rows["failures_and_not_run_receipts.csv"]:
        rows["failures_and_not_run_receipts.csv"].append({
            "experiment_id": "KDD101", "task": "all", "method": "all", "seed": "all",
            "status": "no_failures_or_not_run_events", "reason": "all_prespecified_core_rows_completed",
            "replacement_after_results": False, "claim_boundary": CLAIM,
        })

    output.mkdir(parents=True)
    required = [
        "model_free_method_coverage.csv", "policy_training_and_convergence.csv", "target_probability_completeness.csv",
        "support_overlap_ratio_ess.csv", "policy_entropy_collapse_divergence.csv", "post_training_scoring_decisions.csv",
        "denominator_specific_ope.csv", "reward_clipping_horizon_sensitivity.csv", "core_task_model_free_results.csv",
        "low_action_signal_stress_results.csv", "cross_cohort_model_free_robustness.csv", "seed_stability.csv",
        "resource_metrics.csv", "failures_and_not_run_receipts.csv", "privacy_audit.csv", "decision.csv",
    ]
    for name in required:
        write_csv(output / name, rows[name])
    (output / "kdd101_report.md").write_text(report(rows, config), encoding="utf-8")
    write_hashes(output)
    for historical_path, historical_hash in old_receipts.items():
        if tree_hash(Path(historical_path)) != historical_hash:
            raise RuntimeError(f"historical KDD101 receipt changed during additive run: {historical_path}")


def load_config(path: Path) -> dict[str, Any]:
    overlay = json.loads(path.read_text(encoding="utf-8"))
    if "extends" not in overlay:
        return overlay
    base_path = ROOT / overlay.pop("extends")
    expected = overlay.pop("base_config_sha256")
    if sha256(base_path) != expected:
        raise RuntimeError("KDD101 base config hash drift")
    base = load_config(base_path)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key].update(value)
        else:
            base[key] = value
    return base


def verify_contract(config: dict[str, Any]) -> None:
    paths = {"KDD097": KDD097, "KDD099R-A": KDD099RA, "KDD100R": KDD100R, "KDD099R-B": KDD099RB}
    for name, path in paths.items():
        manifest = path / "artifact_hashes.json"
        if not manifest.exists() or sha256(manifest) != config["upstream_manifest_sha256"][name]:
            raise RuntimeError(f"{name} manifest hash drift")
    authorized = list(csv.DictReader((KDD099RB / "kdd101_authorized_tasks.csv").open(encoding="utf-8")))
    active = {r["task"] for r in authorized if r["KDD101_model_free_training_authorized"] == "True"}
    if active != set(config["authorized_tasks"]):
        raise RuntimeError("KDD099R-B authorization drift")
    decision = next(csv.DictReader((KDD099RB / "decision.csv").open(encoding="utf-8")))
    if decision["decision"] != "diagnostic_pretraining_authorized_real_ehr_policy_value_scoring_blocked":
        raise RuntimeError("KDD099R-B decision does not authorize diagnostic training")
    approved = list(csv.DictReader((KDD100R / "estimator_and_guardrail_pass_fail.csv").open(encoding="utf-8")))
    if any(r.get("approved_for_real_ehr_policy_value") == "True" for r in approved):
        raise RuntimeError("config says no estimator approved but KDD100R disagrees")
    if config["estimator_disposition"] != "no_real_ehr_policy_value_estimator_approved":
        raise RuntimeError("estimator disposition weakened")


def historical_receipt_hashes() -> dict[str, str]:
    roots = sorted((ROOT / "kdd_benchmark_discovery/results").glob("kdd101_*"))
    return {str(p): tree_hash(p) for p in roots if p.is_dir()}


def supported_actions(task: TaskSequences) -> np.ndarray:
    train = np.flatnonzero(task.episodes("train"))
    mask = np.zeros(task.action_dim, dtype=bool)
    classes = task.action_classes[train][task.valid_steps[train] & task.strong_support[train]]
    mask[np.unique(classes[classes >= 0]).astype(int)] = True
    if not mask.any():
        raise RuntimeError(f"no train-frozen supported action for {task.task}")
    return mask


def fit_behavior_denominators(task: TaskSequences, config: dict[str, Any]) -> dict[str, Any]:
    contract = config["behavior_contract"]
    state = behavior_states(task).reshape(-1, behavior_states(task).shape[-1])
    action = task.action_classes.reshape(-1)
    episode = np.repeat(np.arange(len(task.roles)), task.valid_steps.shape[1])
    valid = task.valid_steps.reshape(-1) & (action >= 0)
    train_ep = np.flatnonzero(task.episodes("train")); val_ep = np.flatnonzero(task.episodes("validation"))
    rng = np.random.default_rng(3408 + sum(map(ord, task.task)))
    order = rng.permutation(train_ep); ncal = max(1, int(round(0.1 * len(order))))
    cal_ep, fit_ep = order[:ncal], order[ncal:]
    fit = capped_indices(np.flatnonzero(valid & np.isin(episode, fit_ep)), int(contract["maximum_fit_windows"]), rng)
    cal = capped_indices(np.flatnonzero(valid & np.isin(episode, cal_ep)), int(contract["maximum_calibration_windows"]), rng)
    val = episode_preserving_indices(task, val_ep, int(contract["maximum_validation_windows"]))
    mean, std = state[fit].mean(0), state[fit].std(0); std[std < 1e-5] = 1.0
    xfit = ((state[fit] - mean) / std).astype(np.float32); xcal = ((state[cal] - mean) / std).astype(np.float32)
    xval = ((state[val] - mean) / std).astype(np.float32); yfit = action[fit].astype(np.int64); ycal = action[cal].astype(np.int64)
    knn_cal, knn_val = _knn(xfit, yfit, xcal, xval, task.action_dim, 3408)
    neural_cal, neural_val = _neural(xfit, yfit, xcal, xval, task.action_dim, 3408)
    return {
        "x_train": xfit, "y_train": yfit, "flat_train": fit, "x_val": xval, "y_val": action[val].astype(np.int64), "flat_val": val,
        "val_episode": episode[val], "val_step": val % task.valid_steps.shape[1], "mean": mean, "std": std,
        "historical_knn": _apply_temperature(knn_val, _temperature(knn_cal, ycal)),
        "neural_classifier": _apply_temperature(neural_val, _temperature(neural_cal, ycal)),
    }


def capped_indices(indices: np.ndarray, maximum: int, rng: np.random.Generator) -> np.ndarray:
    return np.sort(rng.choice(indices, maximum, replace=False)) if len(indices) > maximum else np.asarray(indices)


def episode_preserving_indices(task: TaskSequences, episodes: np.ndarray, maximum: int) -> np.ndarray:
    chosen: list[int] = []
    steps = task.valid_steps.shape[1]
    for ep in episodes:
        local = (ep * steps + np.flatnonzero(task.valid_steps[ep])).tolist()
        if chosen and len(chosen) + len(local) > maximum:
            break
        chosen.extend(local)
    return np.asarray(chosen, dtype=int)


def severity_contract(task: TaskSequences, support: np.ndarray) -> dict[str, Any]:
    train = np.flatnonzero(task.episodes("train")); severity = np.nanmean(np.abs(task.states[train, :, :8]), axis=-1)[task.valid_steps[train]]
    actions = np.flatnonzero(support)
    return {"cuts": np.quantile(severity, [1 / 3, 2 / 3]), "actions": np.asarray([actions[0], actions[len(actions) // 2], actions[-1]])}


def reward_surface(task: TaskSequences, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    candidate = config["reward_contract"][task.task]
    all_ep = np.arange(len(task.roles))
    truth, _estimate, available = reward_values(task, all_ep, task.targets, task.targets, candidate)
    return truth, available


def train_learned_policies(task: TaskSequences, denom: dict[str, Any], support: np.ndarray, config: dict[str, Any], seed: int, rows: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    torch.manual_seed(seed); np.random.seed(seed)
    train_cfg = config["training"]; maximum = int(train_cfg["maximum_train_windows"])
    rng = np.random.default_rng(seed + sum(map(ord, task.task)))
    take = capped_indices(np.arange(len(denom["x_train"])), maximum, rng)
    xtr, ytr = denom["x_train"][take], denom["y_train"][take]
    xval, yval = denom["x_val"], denom["y_val"]
    reward, reward_ok = reward_surface(task, config)
    flat_reward = reward.reshape(-1); flat_reward_ok = reward_ok.reshape(-1)
    train_global = np.flatnonzero(task.valid_steps.reshape(-1) & np.repeat(task.episodes("train"), task.valid_steps.shape[1]))
    reward_by_train = {int(i): float(flat_reward[i]) for i in train_global if flat_reward_ok[i]}
    results: dict[str, dict[str, Any]] = {}

    bc, bc_meta = fit_classifier(xtr, ytr, xval, yval, task.action_dim, support, train_cfg, seed)
    results["behavior_cloning"] = make_result(policy_probs(bc, xval, support), bc_meta, "independent_reimplementation")
    append_training(rows, task, "behavior_cloning", seed, bc_meta, "independent_reimplementation", len(xtr))

    global_for_take = denom["flat_train"][take]
    reward_keep = np.asarray([int(i) in reward_by_train for i in global_for_take])
    if reward_keep.sum() < 100:
        for method in ("discrete_bcq", "discrete_cql", "soft_spibb"):
            rows["failures_and_not_run_receipts.csv"].append(failure(task.task, method, seed, "insufficient_observed_candidate_reward_transitions"))
    else:
        rtr = np.asarray([reward_by_train.get(int(i), 0.0) for i in global_for_take], dtype=np.float32)
        full_state = behavior_states(task).reshape(-1, behavior_states(task).shape[-1])
        steps = task.valid_steps.shape[1]
        next_global = np.asarray([
            int(i + 1) if int(i) % steps + 1 < steps and task.valid_steps.reshape(-1)[int(i + 1)] else int(i)
            for i in global_for_take
        ])
        xnext = ((full_state[next_global] - denom["mean"]) / denom["std"]).astype(np.float32)
        done = task.terminal.reshape(-1)[global_for_take].astype(np.float32)
        val_global = denom["flat_val"]
        val_keep = flat_reward_ok[val_global]
        val_next_global = np.asarray([
            int(i + 1) if int(i) % steps + 1 < steps and task.valid_steps.reshape(-1)[int(i + 1)] else int(i)
            for i in val_global
        ])
        val_next = ((full_state[val_next_global] - denom["mean"]) / denom["std"]).astype(np.float32)
        val_reward = flat_reward[val_global].astype(np.float32)
        val_done = task.terminal.reshape(-1)[val_global].astype(np.float32)
        for method in ("discrete_bcq", "discrete_cql", "soft_spibb"):
            q, meta = fit_q(
                xtr[reward_keep], ytr[reward_keep], rtr[reward_keep], xnext[reward_keep], done[reward_keep],
                xval[val_keep], denom["y_val"][val_keep], val_reward[val_keep], val_next[val_keep], val_done[val_keep],
                task.action_dim, support, train_cfg, seed, method,
            )
            with torch.no_grad(): qval = q(torch.from_numpy(xval)).numpy()
            behavior = results["behavior_cloning"]["probabilities"]
            probs = q_policy(qval, behavior, support, method, train_cfg)
            fidelity = "independent_reimplementation" if method in {"discrete_bcq", "discrete_cql"} else "conceptual_adapter"
            results[method] = make_result(probs, meta, fidelity)
            append_training(rows, task, method, seed, meta, fidelity, int(reward_keep.sum()))

    dt_result = fit_decision_transformer(task, denom, support, config, seed)
    if dt_result is None:
        rows["failures_and_not_run_receipts.csv"].append(failure(task.task, "decision_transformer", seed, "insufficient_complete_reward_conditioned_steps"))
    else:
        results["decision_transformer"] = dt_result
        append_training(rows, task, "decision_transformer", seed, dt_result["meta"], "official_contract_adapter", dt_result["meta"]["training_examples"])
    return results


def fit_classifier(xtr, ytr, xval, yval, action_dim, support, cfg, seed):
    model = MLP(xtr.shape[1], action_dim, int(cfg["hidden_dim"])); optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    return fit_loop(model, optimizer, TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr)), xval, yval, support, cfg, seed, "classifier")


def fit_q(xtr, ytr, reward, xnext, done, xval, yval, val_reward, val_next, val_done, action_dim, support, cfg, seed, method):
    if min(len(xtr), len(xval)) < 100:
        raise RuntimeError(f"insufficient observed-reward TD transitions for {method}")
    model = MLP(xtr.shape[1], action_dim, int(cfg["hidden_dim"])); target = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    dataset = TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr), torch.from_numpy(reward), torch.from_numpy(xnext), torch.from_numpy(done))
    loader = DataLoader(dataset, batch_size=int(cfg["batch_size"]), shuffle=True, generator=torch.Generator().manual_seed(seed))
    support_t = torch.from_numpy(support); best, best_loss, stale, best_epoch = None, math.inf, 0, 0; start = time.time()
    gamma = float(cfg["discount"])
    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        model.train(); target.eval()
        for xb, ab, rb, nb, db in loader:
            optimizer.zero_grad(); q = model(xb); chosen = q.gather(1, ab[:, None]).squeeze(1)
            with torch.no_grad(): td_target = rb + gamma * (1.0 - db) * target(nb)[:, support_t].max(1).values
            loss = nn.functional.mse_loss(chosen, td_target)
            if method in {"discrete_cql", "soft_spibb"}:
                loss = loss + float(cfg["cql_alpha"]) * (torch.logsumexp(q[:, support_t], 1) - chosen).mean()
            loss.backward(); optimizer.step()
        target.load_state_dict(model.state_dict())
        model.eval()
        with torch.no_grad():
            vq = model(torch.from_numpy(xval)); vchosen = vq.gather(1, torch.from_numpy(yval)[:, None]).squeeze(1)
            vtarget = torch.from_numpy(val_reward) + gamma * (1.0 - torch.from_numpy(val_done)) * target(torch.from_numpy(val_next))[:, support_t].max(1).values
            val_loss = float(nn.functional.mse_loss(vchosen, vtarget).item())
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch, stale, best = val_loss, epoch, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else: stale += 1
        if epoch >= int(cfg["min_epochs"]) and stale >= int(cfg["patience"]): break
    if best is not None: model.load_state_dict(best)
    meta = {"best_epoch": best_epoch, "epochs_run": epoch, "validation_objective": best_loss, "validation_objective_name": "observed_reward_discounted_TD_MSE", "converged": stale >= int(cfg["patience"]), "runtime_seconds": time.time() - start, "checkpoint_sha256": model_hash(model), "training_examples": len(dataset)}
    return model, meta


def fit_loop(model, optimizer, dataset, xval, yval, support, cfg, seed, kind):
    loader = DataLoader(dataset, batch_size=int(cfg["batch_size"]), shuffle=True, generator=torch.Generator().manual_seed(seed))
    best, best_loss, stale, best_epoch = None, math.inf, 0, 0; start = time.time()
    support_t = torch.from_numpy(support)
    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        model.train()
        for batch in loader:
            optimizer.zero_grad(); logits = model(batch[0]); action = batch[1]
            if kind == "classifier": loss = nn.functional.cross_entropy(logits, action)
            else:
                chosen = logits.gather(1, action[:, None]).squeeze(1); loss = nn.functional.mse_loss(chosen, batch[2])
                if kind in {"discrete_cql", "soft_spibb"}: loss = loss + float(cfg["cql_alpha"]) * (torch.logsumexp(logits[:, support_t], 1) - chosen).mean()
            loss.backward(); optimizer.step()
        with torch.no_grad():
            val_logits = model(torch.from_numpy(xval)); val_logits[:, ~support_t] = -1e9
            if kind == "classifier": val_loss = float(nn.functional.cross_entropy(val_logits, torch.from_numpy(yval)).item())
            else: val_loss = float(torch.mean(torch.square(val_logits[:, support_t])).item())
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch, stale = val_loss, epoch, 0; best = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else: stale += 1
        if epoch >= int(cfg["min_epochs"]) and stale >= int(cfg["patience"]): break
    if best is not None: model.load_state_dict(best)
    meta = {"best_epoch": best_epoch, "epochs_run": epoch, "validation_objective": best_loss, "converged": stale >= int(cfg["patience"]), "runtime_seconds": time.time() - start, "checkpoint_sha256": model_hash(model), "training_examples": len(dataset)}
    return model, meta


def policy_probs(model, x, support):
    with torch.no_grad(): logits = model(torch.from_numpy(x)).numpy()
    logits[:, ~support] = -1e9; logits -= logits.max(1, keepdims=True); p = np.exp(logits); return p / p.sum(1, keepdims=True)


def q_policy(q, behavior, support, method, cfg):
    masked = np.where(support[None, :], q, -np.inf)
    greedy = np.zeros_like(q); greedy[np.arange(len(q)), np.argmax(masked, axis=1)] = 1.0
    if method == "discrete_bcq":
        eligible = behavior >= float(cfg["bcq_behavior_threshold"]) * behavior.max(1, keepdims=True); eligible &= support[None, :]
        out = np.zeros_like(q); out[np.arange(len(q)), np.argmax(np.where(eligible, q, -np.inf), axis=1)] = 1.0; return out
    if method == "soft_spibb":
        mix = float(cfg["soft_spibb_mix"]); out = (1 - mix) * behavior + mix * greedy; out[:, ~support] = 0; return out / out.sum(1, keepdims=True)
    return greedy


def fit_decision_transformer(task, denom, support, config, seed):
    cfg = config["training"]; reward, available = reward_surface(task, config); train = np.flatnonzero(task.episodes("train")); val = np.flatnonzero(task.episodes("validation"))
    x = behavior_states(task); rtg = np.zeros_like(reward, dtype=np.float32); known = np.zeros_like(available, dtype=np.float32)
    for ep in train:
        running = 0.0; complete = True
        for step in range(task.valid_steps.shape[1] - 1, -1, -1):
            if not task.valid_steps[ep, step]: continue
            complete = complete and bool(available[ep, step]); running = float(reward[ep, step]) + float(cfg["discount"]) * running if complete else 0.0
            if complete: rtg[ep, step] = running; known[ep, step] = 1.0
    eligible = task.valid_steps[train] & (known[train] > 0)
    if int(eligible.sum()) < 100: return None
    model = DecisionTransformerAdapter(x.shape[-1], task.action_dim, int(cfg["hidden_dim"]), x.shape[1]); opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    dataset = TensorDataset(torch.from_numpy(x[train]), torch.from_numpy(rtg[train]), torch.from_numpy(known[train]), torch.from_numpy(task.action_classes[train].astype(np.int64)), torch.from_numpy(eligible))
    loader = DataLoader(dataset, batch_size=min(int(cfg["batch_size"]), len(dataset)), shuffle=True, generator=torch.Generator().manual_seed(seed))
    best, best_loss, stale, best_epoch = None, math.inf, 0, 0; start = time.time(); support_t = torch.from_numpy(support)
    target_rtg = float(np.quantile(rtg[train][eligible], 0.75))
    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        model.train()
        for xb, rb, kb, ab, mb in loader:
            opt.zero_grad(); logits = model(xb, rb, kb); loss = nn.functional.cross_entropy(logits[mb], ab[mb]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(x[val]); vr = torch.full(task.valid_steps[val].shape, target_rtg); vk = torch.ones_like(vr); logits = model(vx, vr, vk); mask = torch.from_numpy(task.valid_steps[val]); logits[..., ~support_t] = -1e9
            loss = float(nn.functional.cross_entropy(logits[mask], torch.from_numpy(task.action_classes[val].astype(np.int64))[mask]).item())
        if loss < best_loss - 1e-5: best_loss, best_epoch, stale, best = loss, epoch, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else: stale += 1
        if epoch >= int(cfg["min_epochs"]) and stale >= int(cfg["patience"]): break
    if best is not None: model.load_state_dict(best)
    flat_ep = denom["val_episode"]; flat_step = denom["val_step"]
    with torch.no_grad():
        logits = model(torch.from_numpy(x[val]), torch.full(task.valid_steps[val].shape, target_rtg), torch.ones(task.valid_steps[val].shape)).numpy()
    local_map = {int(ep): i for i, ep in enumerate(val)}; chosen = np.asarray([logits[local_map[int(ep)], int(step)] for ep, step in zip(flat_ep, flat_step, strict=True)])
    chosen[:, ~support] = -1e9; chosen -= chosen.max(1, keepdims=True); p = np.exp(chosen); p /= p.sum(1, keepdims=True)
    meta = {"best_epoch": best_epoch, "epochs_run": epoch, "validation_objective": best_loss, "converged": stale >= int(cfg["patience"]), "runtime_seconds": time.time() - start, "checkpoint_sha256": model_hash(model), "training_examples": int(eligible.sum()), "desired_return_source": "train_observed_reward_rtg_q75"}
    return make_result(p, meta, "official_contract_adapter")


def control_probabilities(task, denom, support, severity, config, seed):
    n, k = len(denom["x_val"]), task.action_dim; epsilon = float(config["policy_contract"]["deterministic_control_epsilon"]); supported = np.flatnonzero(support)
    random = np.zeros((n, k)); random[:, supported] = 1 / len(supported)
    def deterministic(actions):
        p = np.zeros((n, k)); p[:, supported] = epsilon / len(supported); p[np.arange(n), actions] += 1 - epsilon; return p
    flat = denom["flat_val"]; state = task.states.reshape(-1, task.states.shape[-1])[flat]; sev = np.nanmean(np.abs(state[:, :8]), axis=1); bucket = np.digitize(sev, severity["cuts"])
    policies = {"supported_random": random, "no_min_action": deterministic(np.full(n, supported[0])), "max_action": deterministic(np.full(n, supported[-1])), "severity_rule": deterministic(severity["actions"][bucket])}
    return {m: make_result(p, {"best_epoch": 0, "epochs_run": 0, "validation_objective": math.nan, "converged": True, "runtime_seconds": 0.0, "checkpoint_sha256": hashlib.sha256((task.task + m).encode()).hexdigest(), "training_examples": 0}, "local_control") for m, p in policies.items()}


def make_result(probabilities, meta, fidelity): return {"probabilities": np.asarray(probabilities, dtype=np.float64), "meta": meta, "fidelity": fidelity}


def append_training(rows, task, method, seed, meta, fidelity, examples):
    rows["policy_training_and_convergence.csv"].append({"experiment_id": "KDD101", "task": task.task, "task_role": "policy_comparison_core", "method": method, "fidelity_label": fidelity, "seed": seed, "fit_role": "train_only", "checkpoint_selection_role": "validation_only", "training_examples": examples, **meta, "checkpoint_exported": False, "status": "trained_development_only", "claim_boundary": CLAIM})
    rows["resource_metrics.csv"].append({"experiment_id": "KDD101", "task": task.task, "method": method, "seed": seed, "runtime_seconds": meta["runtime_seconds"], "device": "cpu", "peak_memory_mb": "not_instrumented", "checkpoint_exported": False, "status": "complete", "claim_boundary": CLAIM})


def audit_policy(task, method, seed, result, denom, support, config, rows):
    p = result["probabilities"]; finite = np.isfinite(p).all(); row_sum = p.sum(1); unsupported = float(p[:, ~support].sum(1).mean()) if (~support).any() else 0.0
    complete = finite and np.max(np.abs(row_sum - 1)) <= 1e-6 and unsupported <= config["policy_contract"]["unsupported_mass_tolerance"]
    marginal = p.mean(0); entropy = float(np.mean(-np.sum(np.where(p > 0, p * np.log(np.clip(p, 1e-12, 1)), 0), 1)))
    normalized_entropy = entropy / math.log(task.action_dim); top_share = float(marginal.max()); used = int((marginal > 1e-4).sum())
    behavior = denom["neural_classifier"]; kl = float(np.mean(np.sum(p * np.log(np.clip(p, 1e-12, 1) / np.clip(behavior, 1e-12, 1)), 1)))
    base = {"experiment_id": "KDD101", "task": task.task, "task_role": "policy_comparison_core", "method": method, "seed": seed, "claim_boundary": CLAIM}
    rows["target_probability_completeness.csv"].append({**base, "validation_steps": len(p), "action_dimension": task.action_dim, "finite": finite, "maximum_sum_to_one_error": float(np.max(np.abs(row_sum - 1))), "unsupported_action_mass": unsupported, "complete": complete, "probabilities_exported": False})
    rows["policy_entropy_collapse_divergence.csv"].append({**base, "entropy": entropy, "normalized_entropy": normalized_entropy, "effective_action_count": math.exp(entropy), "top_action_marginal_share": top_share, "actions_used_probability_gt_1e-4": used, "kl_to_neural_behavior": kl, "collapse_flag_top_share_ge_0p95": top_share >= 0.95, "policy_value_metric": "unavailable"})
    for denominator_name in ("neural_classifier", "historical_knn"):
        d = denom[denominator_name]; logged = denom["y_val"]; target_logged = p[np.arange(len(p)), logged]; denom_logged = d[np.arange(len(d)), logged]
        ratios = target_logged / np.clip(denom_logged, config["behavior_contract"]["denominator_floor"], None)
        low_mass = float(target_logged[denom_logged < config["policy_contract"]["low_denominator_threshold"]].sum() / max(target_logged.sum(), 1e-12))
        for horizon in config["policy_contract"]["horizons"]:
            weights = trajectory_weights(ratios, denom["val_episode"], denom["val_step"], horizon)
            ess = float(weights.sum() ** 2 / np.square(weights).sum()) if len(weights) and np.square(weights).sum() else 0.0
            ess_fraction = ess / len(weights) if len(weights) else 0.0
            overlap_pass = low_mass <= config["policy_contract"]["maximum_low_denominator_target_mass"]
            ratio_pass = bool(len(weights)) and np.isfinite(weights).all() and float(np.quantile(weights, 0.99)) <= config["policy_contract"]["maximum_ratio_q99"]
            ess_pass = ess >= config["policy_contract"]["minimum_ess"] and ess_fraction >= config["policy_contract"]["minimum_ess_fraction"]
            gate = complete and overlap_pass and ratio_pass and ess_pass
            rows["support_overlap_ratio_ess.csv"].append({**base, "denominator": denominator_name, "horizon": horizon, "trajectories": len(weights), "low_denominator_target_mass": low_mass, "ratio_q95": float(np.quantile(weights, .95)) if len(weights) else math.nan, "ratio_q99": float(np.quantile(weights, .99)) if len(weights) else math.nan, "ratio_max": float(weights.max()) if len(weights) else math.nan, "ess": ess, "ess_fraction": ess_fraction, "probability_pass": complete, "overlap_pass": overlap_pass, "ratio_tail_pass": ratio_pass, "ess_pass": ess_pass, "pre_estimator_gate_pass": gate})
        rows["post_training_scoring_decisions.csv"].append({**base, "denominator": denominator_name, "target_probability_complete": complete, "unsupported_action_mass_pass": unsupported <= config["policy_contract"]["unsupported_mass_tolerance"], "overlap_ratio_ess_all_horizons_pass": all(r["pre_estimator_gate_pass"] for r in rows["support_overlap_ratio_ess.csv"] if r["task"] == task.task and r["method"] == method and r["seed"] == seed and r["denominator"] == denominator_name), "KDD100R_approved_estimator_available": False, "real_ehr_policy_value_scoring_authorized": False, "decision": "not_authorized_no_KDD100R_estimator_approved", "claim_boundary": CLAIM})
    rows["core_task_model_free_results.csv"].append({**base, "fidelity_label": result["fidelity"], "training_status": "trained" if result["meta"]["epochs_run"] > 0 else "deterministic_control", "target_probability_complete": complete, "unsupported_action_mass": unsupported, "normalized_entropy": normalized_entropy, "top_action_share": top_share, "behavior_divergence_kl": kl, "seed_checkpoint_sha256": result["meta"]["checkpoint_sha256"], "real_ehr_policy_value": "unavailable_no_approved_estimator", "policy_winner_claimed": False})


def trajectory_weights(ratios, episodes, steps, horizon):
    grouped: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for ep, step, ratio in zip(episodes, steps, ratios, strict=True): grouped[int(ep)].append((int(step), float(ratio)))
    out = []
    for values in grouped.values():
        values.sort(); seq = [v for _s, v in values]
        if len(seq) >= horizon: out.append(float(np.prod(seq[:horizon])))
    return np.asarray(out, dtype=np.float64)


def add_method_coverage(config, rows):
    core = set(config["learned_methods"] + config["control_methods"])
    for source in csv.DictReader(KDD095.open(encoding="utf-8")):
        if source.get("method_id") in core or source.get("method") in core: continue
        method = source.get("method_id") or source.get("method") or source.get("baseline")
        if not method: continue
        rows["model_free_method_coverage.csv"].append({"experiment_id": "KDD101", "method": method, "KDD095_category": source.get("category", source.get("baseline_type", "unknown")), "KDD095_fidelity_label": source.get("fidelity_label", "not_run_with_reason"), "KDD101_status": "not_run_with_reason", "reason": "non_core_KDD095_row_retained_as_coverage_receipt", "trained_or_tested": False, "claim_boundary": CLAIM})
    for method in config["learned_methods"] + config["control_methods"]:
        rows["model_free_method_coverage.csv"].append({"experiment_id": "KDD101", "method": method, "KDD095_category": "model_free_core", "KDD095_fidelity_label": "see_training_registry", "KDD101_status": "executed_or_failure_retained", "reason": "prespecified_core", "trained_or_tested": True, "claim_boundary": CLAIM})


def add_seed_and_cross_cohort(probabilities, config, rows):
    for task in config["authorized_tasks"]:
        for method in config["learned_methods"] + config["control_methods"]:
            available = [(seed, probabilities[(task, method, seed)]["probabilities"]) for seed in config["seeds"] if (task, method, seed) in probabilities]
            for i, (sa, pa) in enumerate(available):
                for sb, pb in available[i + 1:]:
                    rows["seed_stability.csv"].append({"experiment_id": "KDD101", "task": task, "method": method, "seed_a": sa, "seed_b": sb, "mean_probability_total_variation": float(np.mean(np.abs(pa - pb).sum(1) / 2)), "argmax_agreement": float(np.mean(pa.argmax(1) == pb.argmax(1))), "real_ehr_value_stability": "unavailable", "claim_boundary": CLAIM})
            if available:
                entropy = [float(np.mean(-np.sum(p * np.log(np.clip(p, 1e-12, 1)), 1))) for _, p in available]
                rows["cross_cohort_model_free_robustness.csv"].append({"experiment_id": "KDD101", "task": task, "method": method, "seeds_completed": len(available), "entropy_mean_across_seeds": float(np.mean(entropy)), "entropy_std_across_seeds": float(np.std(entropy)), "cross_cohort_policy_rank": "not_computed_no_policy_value", "policy_winner_claimed": False, "claim_boundary": CLAIM})


def add_blocked_ope_receipts(probabilities, config, rows):
    estimators = ["WIS", "WPDIS", "DR", "weighted_DR", "linear_FQE", "neural_FQE"]
    for task, method, seed in probabilities:
        for denominator in config["behavior_contract"]["estimators"]:
            for estimator in estimators:
                rows["denominator_specific_ope.csv"].append({"experiment_id": "KDD101", "task": task, "method": method, "seed": seed, "denominator": denominator, "estimator": estimator, "estimate": "unavailable", "status": "not_run_no_KDD100R_estimator_approved", "gate_relaxed": False, "claim_boundary": CLAIM})
            rows["reward_clipping_horizon_sensitivity.csv"].append({"experiment_id": "KDD101", "task": task, "method": method, "seed": seed, "denominator": denominator, "reward_sensitivity": "not_computed", "clipping_sensitivity": "not_computed", "horizon_sensitivity": "not_computed", "status": "not_run_no_KDD100R_estimator_approved", "claim_boundary": CLAIM})


def add_empty_stress_receipt(rows):
    rows["low_action_signal_stress_results.csv"].append({"experiment_id": "KDD101", "task": "none", "task_role": "low_action_signal_stress_test", "authorized_task_count": 0, "status": "not_run_no_KDD099R_B_authorized_low_action_signal_task", "primary_winner_eligible": False, "claim_boundary": CLAIM})


def add_privacy_and_decision(probabilities, config, rows):
    for artifact in ("patient_or_stay_ids", "exact_timestamps", "row_level_trajectories", "target_probability_rows", "tensors_or_checkpoints", "credentials_or_PHI"):
        rows["privacy_audit.csv"].append({"experiment_id": "KDD101", "artifact_class": artifact, "exported": False, "status": "pass", "claim_boundary": CLAIM})
    trained = len(probabilities)
    rows["decision.csv"].append({"experiment_id": "KDD101", "decision": "complete_diagnostic_model_free_training_no_real_ehr_policy_value", "authorized_tasks": ";".join(config["authorized_tasks"]), "policy_task_role": "policy_comparison_core", "low_action_signal_task_count": 0, "task_method_seed_outputs": trained, "KDD100R_approved_estimator_count": 0, "real_ehr_policy_value_available": False, "primary_policy_winner_available": False, "confirmatory_evidence": False, "claim_boundary": CLAIM})


def failure(task, method, seed, reason): return {"experiment_id": "KDD101", "task": task, "method": method, "seed": seed, "status": "not_run_with_reason", "reason": reason, "replacement_after_results": False, "claim_boundary": CLAIM}


def model_hash(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()): h.update(name.encode()); h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def report(rows, config):
    completed = len(rows["core_task_model_free_results.csv"]); failures = len(rows["failures_and_not_run_receipts.csv"])
    return f"""# KDD101 development-only model-free diagnostics

## Decision

`complete_diagnostic_model_free_training_no_real_ehr_policy_value`

KDD101 produced {completed} aggregate task-method-seed policy diagnostics on the KDD099R-B-authorized sepsis, respiratory, and shock development roles. There were {failures} retained learned-family failure receipts. All three tasks remain in the separate `policy_comparison_core` table; KDD099R-B authorized zero `low_action_signal_stress_test` tasks.

## Frozen contract

- Train-only fitting and validation-only checkpoint selection; seeds {', '.join(map(str, config['seeds']))}.
- KDD097 action/support/split contracts, KDD099 horizons and discount, KDD099R-A benchmark-proxy rewards, and both frozen behavior denominators were retained.
- Unsupported actions have zero target probability. Respiratory's minimum setting is a setting control, not a no-treatment interpretation.
- Learned implementations are labeled adapters or independent reimplementations, never official exact reproductions.

## Scoring disposition

KDD100R approved no OPE estimator. KDD101 therefore did not compute WIS, WPDIS, DR, or FQE and did not substitute another estimator or relax gates. Denominator-specific probability, overlap, ratio-tail, and ESS diagnostics are retained, but real-EHR policy value and policy-winner comparison are unavailable.

## Claim boundary

These are retrospective development-only support, collapse, divergence, and seed-stability diagnostics. They do not establish treatment benefit, causal response, counterfactual validity, clinical utility, deployment readiness, or autonomous decision making.
"""


def write_csv(path, records):
    if not records: raise RuntimeError(f"refusing empty required artifact: {path.name}")
    fields = list(records[0]);
    for record in records:
        for key in record:
            if key not in fields: fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(records)


def write_hashes(output):
    payload = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "artifact_hashes.json"}
    (output / "artifact_hashes.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def tree_hash(path):
    h = hashlib.sha256()
    for file in sorted(p for p in path.rglob("*") if p.is_file()): h.update(str(file.relative_to(path)).encode()); h.update(file.read_bytes())
    return h.hexdigest()


if __name__ == "__main__":
    main()
