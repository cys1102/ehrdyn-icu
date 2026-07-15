from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ope import per_decision_is, trajectory_is, weighted_is, weighted_per_decision_is


@dataclass(frozen=True)
class FiniteMDP:
    transition: np.ndarray
    reward: np.ndarray
    initial: np.ndarray
    support: np.ndarray
    behavior: np.ndarray
    discount: float = 0.99

    @property
    def horizon(self) -> int:
        return int(self.transition.shape[0])

    @property
    def states(self) -> int:
        return int(self.transition.shape[1])

    @property
    def actions(self) -> int:
        return int(self.transition.shape[2])


def make_smoke_mdp(regime: str, *, horizon: int = 4) -> FiniteMDP:
    if regime not in {"null_response", "moderate"}:
        raise ValueError(f"Unknown smoke regime: {regime}")
    states, actions = 5, 4
    absorbing = states - 1
    transition = np.zeros((horizon, states, actions, states), dtype=np.float64)
    reward = np.zeros_like(transition)
    support = np.ones((states, actions), dtype=bool)
    support[:-1, -1] = False
    support[absorbing, -1] = False
    strength = 0.0 if regime == "null_response" else 0.08
    for step in range(horizon):
        for state in range(states):
            for action in range(actions):
                if state == absorbing:
                    transition[step, state, action, absorbing] = 1.0
                    continue
                effect = strength * (action - 1.0)
                improve = float(np.clip(0.20 + effect, 0.02, 0.55))
                worsen = float(np.clip(0.16 - effect, 0.02, 0.55))
                terminate = 0.03
                stay = 1.0 - improve - worsen - terminate
                transition[step, state, action, max(0, state - 1)] += improve
                transition[step, state, action, min(absorbing - 1, state + 1)] += worsen
                transition[step, state, action, state] += stay
                transition[step, state, action, absorbing] += terminate
                for next_state in range(states):
                    dense = 0.05 * (state - next_state)
                    terminal = 1.0 - 0.25 * next_state if step == horizon - 1 and next_state != absorbing else 0.0
                    reward[step, state, action, next_state] = dense + terminal
    if regime == "null_response":
        transition[:] = transition[:, :, :1, :]
        reward[:] = reward[:, :, :1, :]
    initial = np.array([0.1, 0.25, 0.4, 0.25, 0.0], dtype=np.float64)
    behavior = np.zeros((horizon, states, actions), dtype=np.float64)
    base = np.array([0.5, 0.3, 0.2, 0.0], dtype=np.float64)
    for step in range(horizon):
        for state in range(states):
            probabilities = base.copy()
            probabilities[~support[state]] = 0.0
            behavior[step, state] = probabilities / probabilities.sum()
    return FiniteMDP(transition, reward, initial, support, behavior)


def backward_induction(env: FiniteMDP) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((env.horizon + 1, env.states), dtype=np.float64)
    policy = np.zeros((env.horizon, env.states, env.actions), dtype=np.float64)
    for step in range(env.horizon - 1, -1, -1):
        q_values = np.sum(
            env.transition[step]
            * (env.reward[step] + env.discount * values[step + 1][None, None, :]),
            axis=-1,
        )
        q_values = np.where(env.support, q_values, -np.inf)
        choices = np.argmax(q_values, axis=1)
        policy[step, np.arange(env.states), choices] = 1.0
        values[step] = q_values[np.arange(env.states), choices]
    return policy, values


def evaluate_policy_exact(env: FiniteMDP, policy: np.ndarray) -> float:
    values = np.zeros((env.horizon + 1, env.states), dtype=np.float64)
    for step in range(env.horizon - 1, -1, -1):
        q_values = np.sum(
            env.transition[step]
            * (env.reward[step] + env.discount * values[step + 1][None, None, :]),
            axis=-1,
        )
        values[step] = np.sum(policy[step] * q_values, axis=1)
    return float(np.dot(env.initial, values[0]))


def categorical_cem_first_action(
    env: FiniteMDP,
    *,
    state: int,
    planning_horizon: int,
    seed: int,
    candidates: int = 64,
    elites: int = 8,
    iterations: int = 3,
    smoothing: float = 0.2,
) -> tuple[int, dict[str, int | bool]]:
    horizon = min(planning_horizon, env.horizon)
    supported = np.flatnonzero(env.support[state])
    probabilities = np.zeros((horizon, env.actions), dtype=np.float64)
    probabilities[:, supported] = 1.0 / len(supported)
    generator = np.random.default_rng(seed)
    minimum_unique = candidates
    for _ in range(iterations):
        sequences = np.column_stack(
            [generator.choice(env.actions, candidates, p=probabilities[position]) for position in range(horizon)]
        )
        minimum_unique = min(minimum_unique, len(np.unique(sequences, axis=0)))
        scores = np.array([_open_loop_value(env, state, sequence) for sequence in sequences])
        elite_sequences = sequences[np.argsort(scores)[-elites:]]
        update = np.zeros_like(probabilities)
        for position in range(horizon):
            update[position] = np.bincount(elite_sequences[:, position], minlength=env.actions) / elites
        probabilities = smoothing * probabilities + (1.0 - smoothing) * update
        probabilities[:, ~env.support[state]] = 0.0
        probabilities /= probabilities.sum(axis=1, keepdims=True)
    action = int(np.argmax(probabilities[0]))
    return action, {
        "iterations": iterations,
        "candidate_sequences": candidates,
        "elite_sequences": elites,
        "minimum_unique_sequences": minimum_unique,
        "support_mask_bypass": bool(not env.support[state, action]),
        "receding_horizon_first_action_only": True,
    }


def paired_null_returns(env: FiniteMDP, policies: list[np.ndarray], *, episodes: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    initial_uniform = generator.random(episodes)
    transition_uniform = generator.random((episodes, env.horizon))
    returns = []
    for policy_index, policy in enumerate(policies):
        action_uniform = np.random.default_rng(seed + 100 + policy_index).random((episodes, env.horizon))
        returns.append(_simulate(env, policy, initial_uniform, transition_uniform, action_uniform))
    return np.stack(returns)


def run_smoke(seed: int = 3408) -> dict[str, object]:
    null_env = make_smoke_mdp("null_response")
    moderate_env = make_smoke_mdp("moderate")
    null_optimal, _ = backward_induction(null_env)
    moderate_optimal, moderate_values = backward_induction(moderate_env)
    maximum = _fixed_policy(moderate_env, 2)
    paired = paired_null_returns(null_env, [null_env.behavior, null_optimal, _fixed_policy(null_env, 2)], episodes=256, seed=seed)
    maximum_pairwise_gap = float(np.max(np.abs(paired - paired[:1])))
    action, planner = categorical_cem_first_action(
        moderate_env,
        state=2,
        planning_horizon=4,
        seed=seed,
    )
    toy_rewards = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.25], [0.5, 0.5, 0.5]])
    toy_behavior = np.full_like(toy_rewards, 0.5)
    toy_target = np.array([[0.6, 0.5, 0.7], [0.4, 0.6, 0.5], [0.5, 0.4, 0.6]])
    ope = {
        "IS": trajectory_is(toy_rewards, toy_target, toy_behavior),
        "WIS": weighted_is(toy_rewards, toy_target, toy_behavior),
        "PDIS": per_decision_is(toy_rewards, toy_target, toy_behavior),
        "WPDIS": weighted_per_decision_is(toy_rewards, toy_target, toy_behavior),
    }
    finite_ope = all(np.isfinite(item["estimate"]) and item["ess"] > 0 for item in ope.values())
    return {
        "synthetic_only": True,
        "numeric_paper_reproduction": False,
        "null_response_maximum_paired_return_gap": maximum_pairwise_gap,
        "null_response_gate_pass": maximum_pairwise_gap <= 1e-12,
        "exact_optimal_value": float(np.dot(moderate_env.initial, moderate_values[0])),
        "maximum_supported_value": evaluate_policy_exact(moderate_env, maximum),
        "cem_first_action": action,
        "planner_audit": planner,
        "ope_formula_receipts": ope,
        "ope_formula_gate_pass": bool(finite_ope),
        "pass": bool(maximum_pairwise_gap <= 1e-12 and not planner["support_mask_bypass"] and finite_ope),
    }


def _fixed_policy(env: FiniteMDP, action: int) -> np.ndarray:
    policy = np.zeros((env.horizon, env.states, env.actions), dtype=np.float64)
    for state in range(env.states):
        choice = action if env.support[state, action] else int(np.flatnonzero(env.support[state])[0])
        policy[:, state, choice] = 1.0
    return policy


def _open_loop_value(env: FiniteMDP, state: int, sequence: np.ndarray) -> float:
    distribution = np.zeros(env.states, dtype=np.float64)
    distribution[state] = 1.0
    value = 0.0
    discount = 1.0
    for step, action in enumerate(sequence):
        expected_reward = np.sum(
            distribution[:, None]
            * env.transition[step, :, action, :]
            * env.reward[step, :, action, :]
        )
        value += discount * float(expected_reward)
        distribution = distribution @ env.transition[step, :, action, :]
        discount *= env.discount
    return value


def _simulate(
    env: FiniteMDP,
    policy: np.ndarray,
    initial_uniform: np.ndarray,
    transition_uniform: np.ndarray,
    action_uniform: np.ndarray,
) -> np.ndarray:
    states = _sample_rows(np.broadcast_to(env.initial, (len(initial_uniform), env.states)), initial_uniform)
    returns = np.zeros(len(states), dtype=np.float64)
    discount = 1.0
    for step in range(env.horizon):
        action_probabilities = policy[step, states]
        actions = _sample_rows(action_probabilities, action_uniform[:, step])
        next_probabilities = env.transition[step, states, actions]
        next_states = _sample_rows(next_probabilities, transition_uniform[:, step])
        returns += discount * env.reward[step, states, actions, next_states]
        states = next_states
        discount *= env.discount
    return returns


def _sample_rows(probabilities: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    cumulative = np.cumsum(probabilities, axis=1)
    return np.minimum((uniforms[:, None] > cumulative).sum(axis=1), probabilities.shape[1] - 1)
