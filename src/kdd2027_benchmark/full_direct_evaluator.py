"""Auditable repaired-POMDP data generation and direct policy evaluation."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

@dataclass(frozen=True)
class SyntheticData:
    observed: np.ndarray
    masks: np.ndarray
    deltas: np.ndarray
    actions: np.ndarray
    next_observed: np.ndarray
    behavior_probability: np.ndarray
    rewards: np.ndarray
    done: np.ndarray
    valid: np.ndarray
    subtypes: np.ndarray

from .full_pomdp_v2 import KDD198EnvironmentV2, independent_dense_values


Policy = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]


def _dense_realization(
    env: KDD198EnvironmentV2,
    streams: dict[str, np.ndarray],
    states: np.ndarray,
    subtypes: np.ndarray,
    pending: np.ndarray,
    action: np.ndarray,
    alive: np.ndarray,
    step: int,
) -> np.ndarray:
    """Return the correction from expected dense table to audited realization."""
    correction = np.zeros(len(states), dtype=float)
    for component_index, target in enumerate(env.contract.dense_targets):
        if not target.primary:
            continue
        expected = env.reward_components["dense_physiology"][states, subtypes, pending, action]
        correction[alive] -= expected[alive]
        available = alive & (
            streams["dense_u"][:, step, component_index]
            < target.availability_or_nonzero_fraction
        )
        indices = np.flatnonzero(available)
        score = env.construction.response_strength * np.asarray(
            [
                0.5
                - env._mismatch(
                    int(states[index]), int(subtypes[index]), int(action[index])
                )
                for index in indices
            ]
        )
        correction[indices] += independent_dense_values(
            streams["dense_sign_u"][indices, step, component_index],
            score,
            target.variance,
            float(env.generator["dense_response_fraction"]),
        )
    return correction


def collect_repaired_dataset(
    env: KDD198EnvironmentV2,
    episodes: int,
    stream_seed: int,
    behavior_family: str,
) -> SyntheticData:
    """Generate logged data under the repaired reward mechanism."""
    streams = env._streams(episodes, stream_seed)
    states = env._draw(
        np.broadcast_to(env.initial_state_probability, (episodes, env.states)),
        streams["state_u"],
    )
    subtypes = env._draw(
        np.broadcast_to(env.subtype_prevalence, (episodes, env.subtypes)),
        streams["subtype_u"],
    )
    marginal = np.asarray(env.contract.target_action_frequency)[env.supported]
    marginal /= marginal.sum()
    pending = env.supported[
        env._draw(
            np.broadcast_to(marginal, (episodes, len(env.supported))),
            streams["prior_u"],
        )
    ]
    previous_mask = np.ones((episodes, env.contract.feature_dim), dtype=bool)
    recency = np.zeros((episodes, env.contract.feature_dim), dtype=np.float32)
    alive = np.ones(episodes, dtype=bool)
    features, horizon, actions_n = (
        env.contract.feature_dim,
        env.horizon,
        env.contract.action_count,
    )
    observed = np.zeros((episodes, horizon, features), np.float32)
    masks = np.zeros((episodes, horizon, features), bool)
    deltas = np.zeros((episodes, horizon, features), np.float32)
    actions = np.zeros((episodes, horizon), np.int16)
    next_observed = np.zeros((episodes, horizon, features), np.float32)
    probabilities = np.zeros((episodes, horizon, actions_n), np.float32)
    rewards = np.zeros((episodes, horizon), np.float32)
    done = np.zeros((episodes, horizon), bool)
    valid = np.zeros((episodes, horizon), bool)
    subtype_rows = np.repeat(subtypes[:, None], horizon, axis=1).astype(np.int16)
    for step in range(horizon):
        obs, mask, current_delta = env._emit(
            states,
            subtypes,
            previous_mask,
            recency,
            pending,
            streams["noise"][:, step],
            streams["mask_u"][:, step],
        )
        probability = np.zeros((episodes, actions_n), float)
        if behavior_family == "ehr_matched":
            bins = env.behavior.context_bin(obs, mask, current_delta, pending, step)
            for index in range(episodes):
                probability[index] = env.behavior.distribution(
                    int(pending[index]), int(bins[index]), actions_n
                )
        elif behavior_family == "smart_like_exploratory":
            probability[:, env.supported] = 1.0 / len(env.supported)
        elif behavior_family == "concentrated_behavior":
            probability[:, env.supported] = (
                1.0 - float(env.generator["concentrated_previous_action_mass"])
            ) * marginal
            probability[np.arange(episodes), pending] += float(
                env.generator["concentrated_previous_action_mass"]
            )
        else:
            raise ValueError(behavior_family)
        action = env._draw(probability, streams["policy_u"][:, step])
        action[~alive] = env.supported[0]
        transition = env.transition[states, subtypes, pending, action]
        next_state = env._draw(transition, streams["transition_u"][:, step])
        immediate = np.zeros(episodes)
        for table in env.reward_components.values():
            immediate += table[states, subtypes, pending, action]
        immediate += _dense_realization(
            env, streams, states, subtypes, pending, action, alive, step
        )
        hazard = 1.0 if step == horizon - 1 else env.contract.termination_hazards[step]
        terminate = alive & (streams["termination_u"][:, step] < hazard)
        terminal = np.zeros(episodes)
        if env.contract.primary_reward_type == "terminal":
            death = streams["outcome_u"][:, step] < env.death_probability(next_state)
            terminal[terminate] = np.where(
                death[terminate],
                env.contract.terminal_reward_minimum,
                env.contract.terminal_reward_maximum,
            )
        next_obs, _, _ = env._emit(
            next_state,
            subtypes,
            mask,
            current_delta,
            action,
            np.zeros_like(streams["noise"][:, step]),
            np.full_like(streams["mask_u"][:, step], 0.5),
        )
        observed[:, step] = obs
        masks[:, step] = mask
        deltas[:, step] = current_delta
        actions[:, step] = action
        next_observed[:, step] = next_obs
        probabilities[:, step] = probability
        rewards[:, step] = immediate + terminal
        done[:, step] = terminate
        valid[:, step] = alive
        alive &= ~terminate
        states, pending, previous_mask, recency = (
            next_state,
            action,
            mask,
            current_delta,
        )
    return SyntheticData(
        observed,
        masks,
        deltas,
        actions,
        next_observed,
        probabilities,
        rewards,
        done,
        valid,
        subtype_rows,
    )


def evaluate_repaired_policy_batch(
    env: KDD198EnvironmentV2,
    policy: Policy,
    episodes: int,
    exogenous_seed: int,
    policy_seed: int,
) -> dict[str, Any]:
    """Evaluate a policy with shared exogenous and separate policy streams."""
    streams = env._streams(episodes, exogenous_seed)
    streams["policy_u"] = np.random.default_rng(policy_seed).random(
        (episodes, env.horizon)
    )
    states = env._draw(
        np.broadcast_to(env.initial_state_probability, (episodes, env.states)),
        streams["state_u"],
    )
    subtypes = env._draw(
        np.broadcast_to(env.subtype_prevalence, (episodes, env.subtypes)),
        streams["subtype_u"],
    )
    marginal = np.asarray(env.contract.target_action_frequency)[env.supported]
    marginal /= marginal.sum()
    pending = env.supported[
        env._draw(
            np.broadcast_to(marginal, (episodes, len(env.supported))),
            streams["prior_u"],
        )
    ]
    previous_mask = np.ones((episodes, env.contract.feature_dim), dtype=bool)
    recency = np.zeros((episodes, env.contract.feature_dim), dtype=np.float32)
    alive = np.ones(episodes, dtype=bool)
    returns = np.zeros(episodes)
    terminal_count = np.zeros(episodes, np.int16)
    actions_all: list[np.ndarray] = []
    unsupported_mass = 0.0
    probability_rows = 0
    for step in range(env.horizon):
        observation, mask, recency = env._emit(
            states,
            subtypes,
            previous_mask,
            recency,
            pending,
            streams["noise"][:, step],
            streams["mask_u"][:, step],
        )
        probability = policy(observation, mask, recency, pending, step)
        if probability.shape != (episodes, env.contract.action_count):
            raise RuntimeError("target-policy shape failure")
        if not np.isfinite(probability).all() or np.max(
            np.abs(probability.sum(axis=1) - 1.0)
        ) > 1e-8:
            raise RuntimeError("target-policy normalization failure")
        unsupported = np.ones(env.contract.action_count, dtype=bool)
        unsupported[env.supported] = False
        unsupported_mass += float(probability[:, unsupported].sum())
        probability_rows += episodes
        if unsupported_mass > 1e-10:
            raise RuntimeError("target-policy support failure")
        action = env._draw(probability, streams["policy_u"][:, step])
        action[~alive] = env.supported[0]
        transition = env.transition[states, subtypes, pending, action]
        next_state = env._draw(transition, streams["transition_u"][:, step])
        reward = np.zeros(episodes)
        for table in env.reward_components.values():
            reward += table[states, subtypes, pending, action]
        reward += _dense_realization(
            env, streams, states, subtypes, pending, action, alive, step
        )
        hazard = 1.0 if step == env.horizon - 1 else env.contract.termination_hazards[step]
        terminate = alive & (streams["termination_u"][:, step] < hazard)
        if env.contract.primary_reward_type == "terminal":
            death = streams["outcome_u"][:, step] < env.death_probability(next_state)
            terminal = np.zeros(episodes)
            terminal[terminate] = np.where(
                death[terminate],
                env.contract.terminal_reward_minimum,
                env.contract.terminal_reward_maximum,
            )
            reward += terminal
            terminal_count[terminate] += 1
        returns += (env.discount**step) * np.where(alive, reward, 0.0)
        actions_all.append(action[alive].copy())
        alive &= ~terminate
        states, pending, previous_mask = next_state, action, mask
    return {
        "returns": returns,
        "mean_return": float(returns.mean()),
        "return_se": float(returns.std(ddof=1) / math.sqrt(episodes)),
        "terminal_emission_max": int(terminal_count.max()),
        "distinct_actions": int(np.unique(np.concatenate(actions_all)).size),
        "unsupported_mass": unsupported_mass / max(probability_rows, 1),
    }


def generator_return_range(env: KDD198EnvironmentV2) -> float:
    """Frozen theoretical range used only for reporting and precision."""
    discounted_steps = sum(env.discount**step for step in range(env.horizon))
    maximum_cost = sum(
        float(env.generator[name])
        for name in ("treatment_cost_scale", "toxicity_scale", "switching_cost_scale")
    )
    if env.contract.primary_reward_type == "terminal":
        return 2.0 + discounted_steps * maximum_cost
    primary = next(target for target in env.contract.dense_targets if target.primary)
    dense_bound = math.sqrt(primary.variance / 1.01) * (
        1.0 + float(env.generator["dense_response_fraction"])
    )
    return discounted_steps * (2.0 * dense_bound + maximum_cost)


def return_summary(values: np.ndarray, normalization_range: float, confidence_z: float) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    se = float(values.std(ddof=1) / math.sqrt(len(values)))
    mean = float(values.mean())
    return {
        "mean_return": mean,
        "standard_error": se,
        "ci_lower": mean - confidence_z * se,
        "ci_upper": mean + confidence_z * se,
        "normalized_mean_return": mean / normalization_range,
        "normalized_standard_error": se / normalization_range,
        "effective_episode_count": int(len(values)),
    }


def paired_summary(
    left: np.ndarray,
    right: np.ndarray,
    normalization_range: float,
    confidence_z: float,
) -> dict[str, float]:
    difference = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    summary = return_summary(difference, normalization_range, confidence_z)
    return {
        "paired_mean_difference": summary["mean_return"],
        "paired_standard_error": summary["standard_error"],
        "paired_ci_lower": summary["ci_lower"],
        "paired_ci_upper": summary["ci_upper"],
        "normalized_paired_mean_difference": summary["normalized_mean_return"],
        "normalized_paired_standard_error": summary["normalized_standard_error"],
        "effective_episode_count": summary["effective_episode_count"],
    }


def exact_agreement_summary(
    values: np.ndarray,
    exact_value: float,
    normalization_range: float,
    confidence_z: float,
    normalized_absolute_floor: float,
    standard_error_multiplier: float,
) -> dict[str, float | bool]:
    summary = return_summary(values, normalization_range, confidence_z)
    normalized_error = abs(summary["mean_return"] - exact_value) / normalization_range
    tolerance = max(
        normalized_absolute_floor,
        standard_error_multiplier * summary["normalized_standard_error"],
    )
    return {
        "exact_dp_value": float(exact_value),
        "simulator_monte_carlo_value": summary["mean_return"],
        "absolute_error": abs(summary["mean_return"] - exact_value),
        "normalized_absolute_error": normalized_error,
        "normalized_agreement_tolerance": tolerance,
        "monte_carlo_standard_error": summary["standard_error"],
        "episode_count": summary["effective_episode_count"],
        "agreement_pass": normalized_error <= tolerance,
    }
