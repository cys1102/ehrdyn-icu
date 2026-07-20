from __future__ import annotations

import math
import random
from pathlib import Path
from statistics import fmean, stdev

from .errors import ReleaseContractError
from .public_pomdp import PublicPOMDP, PublicPOMDPConfig, behavior_policy, fit_behavior_cloning


ESTIMATORS = ("IS", "WIS", "CWPDIS", "DR", "WDR", "FQE")


def _fit_nuisance(trajectories, action_count: int, discount: float):
    behavior: dict[int, list[float]] = {}
    q_values: dict[tuple[int, int], list[float]] = {}
    for trajectory in trajectories:
        returns = []
        value = 0.0
        for step in reversed(trajectory):
            value = step.reward + discount * value
            returns.append(value)
        returns.reverse()
        for step, value in zip(trajectory, returns):
            behavior.setdefault(step.context, [1.0] * action_count)[step.action] += 1.0
            q_values.setdefault((step.context, step.action), []).append(value)
    behavior = {key: [value / sum(values) for value in values] for key, values in behavior.items()}
    q = {key: fmean(values) for key, values in q_values.items()}
    return behavior, q


def _target_probability(step, target: str, action_count: int) -> float:
    if target == "behavior":
        return step.behavior_probability
    preferred = step.context % action_count
    return 0.9 if step.action == preferred else 0.1 / max(action_count - 1, 1)


def _estimate(trajectories, target: str, action_count: int, discount: float):
    denominator, q = _fit_nuisance(trajectories, action_count, discount)
    weighted_returns, final_weights, pdis_rows, dr_rows = [], [], [], []
    step_weighted: dict[int, list[tuple[float, float]]] = {}
    for trajectory in trajectories:
        weight, discounted_return, pdis, dr = 1.0, 0.0, 0.0, 0.0
        for time, step in enumerate(trajectory):
            denominator_probability = denominator.get(step.context, [1.0 / action_count] * action_count)[step.action]
            ratio = _target_probability(step, target, action_count) / max(denominator_probability, 1e-12)
            weight *= ratio
            discounted = discount**time
            discounted_return += discounted * step.reward
            pdis += discounted * weight * step.reward
            q_logged = q.get((step.context, step.action), 0.0)
            target_v = sum(_target_probability(type("Row", (), {"context": step.context, "action": action, "behavior_probability": 0.0})(), target, action_count) * q.get((step.context, action), 0.0) for action in range(action_count))
            dr += discounted * (target_v + weight * (step.reward - q_logged))
            step_weighted.setdefault(time, []).append((weight, step.reward))
        weighted_returns.append(weight * discounted_return)
        final_weights.append(weight)
        pdis_rows.append(pdis)
        dr_rows.append(dr)
    total_weight = sum(final_weights)
    cwpdis = 0.0
    for time, values in step_weighted.items():
        denominator_step = sum(weight for weight, _ in values)
        if denominator_step > 0:
            cwpdis += discount**time * sum(weight * reward for weight, reward in values) / denominator_step
    is_value = fmean(weighted_returns)
    wis = sum(weighted_returns) / max(total_weight, 1e-12)
    dr = fmean(dr_rows)
    wdr = 0.5 * dr + 0.5 * cwpdis
    initial_contexts = [trajectory[0].context for trajectory in trajectories if trajectory]
    fqe = fmean(max(q.get((context, action), 0.0) for action in range(action_count)) for context in initial_contexts)
    ess = total_weight**2 / max(sum(value * value for value in final_weights), 1e-12)
    return {"IS": is_value, "WIS": wis, "CWPDIS": cwpdis, "DR": dr, "WDR": wdr, "FQE": fqe}, ess


def _central_interval_width(values: list[float]) -> float:
    ordered = sorted(values)
    lower = ordered[max(0, math.floor(0.05 * (len(ordered) - 1)))]
    upper = ordered[min(len(ordered) - 1, math.ceil(0.95 * (len(ordered) - 1)))]
    return upper - lower


def run_public_ope_smoke(config_path: Path, profile: str, environment_seed: int, datasets: int, episodes: int, bootstrap: int, seed: int) -> dict[str, object]:
    if datasets < 2 or episodes < 16 or bootstrap < 2:
        raise ReleaseContractError("Repeated-OPE smoke requires >=2 datasets, >=16 episodes, and >=2 refits")
    config = PublicPOMDPConfig.load(config_path, profile)
    environment = PublicPOMDP(config, environment_seed)
    behavior = behavior_policy(environment)
    estimates = {target: {name: [] for name in ESTIMATORS} for target in ("behavior", "fixed_context_action")}
    widths = {target: {name: [] for name in ESTIMATORS} for target in estimates}
    ess_values = {target: [] for target in estimates}
    for dataset_index in range(datasets):
        trajectories = [environment.run_episode(seed + dataset_index * 100_000 + episode, behavior)[1] for episode in range(episodes)]
        _ = fit_behavior_cloning(environment, trajectories)
        for target in estimates:
            point, ess = _estimate(trajectories, target, config.action_count, config.discount)
            ess_values[target].append(ess)
            bootstrap_values = {name: [] for name in ESTIMATORS}
            rng = random.Random(seed + 9_000_000 + dataset_index)
            for _replicate in range(bootstrap):
                resample = [trajectories[rng.randrange(len(trajectories))] for _ in trajectories]
                local, _ = _estimate(resample, target, config.action_count, config.discount)
                for name in ESTIMATORS:
                    bootstrap_values[name].append(local[name])
            for name in ESTIMATORS:
                estimates[target][name].append(point[name])
                widths[target][name].append(_central_interval_width(bootstrap_values[name]))
    rows = []
    for target in estimates:
        for name in ESTIMATORS:
            values = estimates[target][name]
            rows.append({
                "target": target,
                "estimator": name,
                "dataset_count": datasets,
                "mean_estimate": fmean(values),
                "between_dataset_sd": stdev(values),
                "mean_full_refit_interval_width": fmean(widths[target][name]),
                "finite_fraction": sum(math.isfinite(value) for value in values) / len(values),
                "mean_ess": fmean(ess_values[target]),
            })
    return {
        "profile": profile,
        "environment_seed": environment_seed,
        "behavior_data_seed_namespace": seed,
        "bootstrap_seed_namespace": seed + 9_000_000,
        "datasets": datasets,
        "episodes_per_dataset": episodes,
        "full_refit_bootstrap_replicates": bootstrap,
        "estimators": list(ESTIMATORS),
        "rows": rows,
        "nuisance_refit_inside_each_bootstrap": True,
        "full_kdd202b_contract": {"datasets": 320, "episodes_per_dataset": 256, "bootstrap_replicates": 500},
        "scope": "bounded public smoke; not the immutable KDD202B evidence run",
    }
