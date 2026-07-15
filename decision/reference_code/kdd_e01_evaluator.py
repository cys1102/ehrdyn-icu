"""KDD-E01 known-value evaluator primitives.

This module is intentionally independent of patient-level exports.  Tier 1 is
an exactly enumerable finite-horizon MDP.  Tier 2 accepts aggregate calibration
parameters only; its action response and outcomes remain synthetic and known.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Callable

import numpy as np
import torch
from torch import nn


HORIZONS = (1, 2, 4, 8, 12, 17)
PLANNER_NAMES = ("H1_exhaustive", "H4_categorical_CEM", "H8_categorical_CEM")


@dataclass(frozen=True)
class FiniteMDP:
    name: str
    transition: np.ndarray  # [H,S,A,S]
    reward: np.ndarray  # [H,S,A,S]
    initial: np.ndarray  # [S]
    support: np.ndarray  # [S,A]
    behavior: np.ndarray  # [H,S,A]
    discount: float = 0.99

    @property
    def horizon(self) -> int:
        return int(self.transition.shape[0])

    @property
    def n_states(self) -> int:
        return int(self.transition.shape[1])

    @property
    def n_actions(self) -> int:
        return int(self.transition.shape[2])


def make_finite_mdp(regime: str, horizon: int = 17) -> FiniteMDP:
    """Create a small exact MDP with an absorbing censored/terminal state."""
    if regime not in {"null", "weak", "moderate", "delayed"}:
        raise ValueError(f"unknown regime: {regime}")
    s_count, a_count = 6, 5
    absorbing = s_count - 1
    transition = np.zeros((horizon, s_count, a_count, s_count), dtype=np.float64)
    reward = np.zeros_like(transition)
    support = np.ones((s_count, a_count), dtype=bool)
    support[0, 4] = False
    support[1, 0] = False
    support[absorbing, 1:] = False
    response = {"null": 0.0, "weak": 0.025, "moderate": 0.08, "delayed": 0.07}[regime]

    for t in range(horizon):
        for s in range(s_count):
            for a in range(a_count):
                if s == absorbing:
                    transition[t, s, a, absorbing] = 1.0
                    continue
                action_center = (a - 2.0) / 2.0
                effect = response * action_center
                if regime == "delayed" and t < 4:
                    effect = 0.0
                improve = np.clip(0.20 + effect, 0.02, 0.65)
                worsen = np.clip(0.16 - effect, 0.02, 0.65)
                censor = 0.025 + 0.01 * (s >= 3)
                stay = 1.0 - improve - worsen - censor
                transition[t, s, a, max(0, s - 1)] += improve
                transition[t, s, a, min(absorbing - 1, s + 1)] += worsen
                transition[t, s, a, s] += stay
                transition[t, s, a, absorbing] += censor
                # Sparse physiology plus a terminal component. Null is action invariant.
                for sp in range(s_count):
                    dense = 0.03 * (s - sp) if t % 3 == 0 else 0.0
                    terminal = 0.0
                    if t == horizon - 1 and sp != absorbing:
                        terminal = 1.0 - 0.35 * (sp / (absorbing - 1))
                    reward[t, s, a, sp] = dense + terminal

    # Null must be exactly action invariant, including unsupported counterfactual rows.
    if regime == "null":
        transition[:] = transition[:, :, :1, :]
        reward[:] = reward[:, :, :1, :]

    initial = np.array([0.08, 0.20, 0.34, 0.25, 0.13, 0.0], dtype=np.float64)
    behavior = np.zeros((horizon, s_count, a_count), dtype=np.float64)
    base = np.array([0.48, 0.25, 0.15, 0.08, 0.04], dtype=np.float64)
    for t in range(horizon):
        for s in range(s_count):
            p = base.copy()
            p[~support[s]] = 0.0
            p /= p.sum()
            behavior[t, s] = p
    return FiniteMDP(regime, transition, reward, initial, support, behavior)


def make_sepsis_semisynthetic_mdp(regime: str, horizon: int = 17) -> FiniteMDP:
    """Aggregate-calibrated K25/33-state tier; mechanisms are fully synthetic.

    The calibration constants are aggregate KDD-S02 receipts (22 supported
    actions, 17 decisions, 33 current features). No patient rows are loaded.
    """
    if regime not in {"null", "weak", "moderate", "delayed"}:
        raise ValueError(regime)
    s_count, a_count = 33, 25
    absorbing = 32
    transition = np.zeros((horizon, s_count, a_count, s_count), dtype=np.float64)
    reward = np.zeros_like(transition)
    support = np.ones((s_count, a_count), dtype=bool)
    support[:, 22:] = False
    support[absorbing, 1:] = False
    strength = {"null": 0.0, "weak": 0.012, "moderate": 0.035, "delayed": 0.03}[regime]
    for t in range(horizon):
        for s in range(s_count):
            for a in range(a_count):
                if s == absorbing:
                    transition[t, s, a, absorbing] = 1.0
                    continue
                fluid, vaso = divmod(a, 5)
                intensity = ((fluid + vaso) - 4.0) / 4.0
                effect = strength * intensity
                if regime == "delayed" and t < 4:
                    effect = 0.0
                improve = np.clip(0.14 + effect, 0.01, 0.35)
                worsen = np.clip(0.12 - effect, 0.01, 0.35)
                censor = 0.018
                stay = 1.0 - improve - worsen - censor
                transition[t, s, a, max(0, s - 1)] += improve
                transition[t, s, a, min(absorbing - 1, s + 1)] += worsen
                transition[t, s, a, s] += stay
                transition[t, s, a, absorbing] += censor
                for sp in range(s_count):
                    dense = 0.015 * (s - sp) if t % 4 == 0 else 0.0
                    terminal = (1.0 - sp / 31.0) if t == horizon - 1 and sp != absorbing else 0.0
                    reward[t, s, a, sp] = dense + terminal
    if regime == "null":
        transition[:] = transition[:, :, :1, :]
        reward[:] = reward[:, :, :1, :]
    initial = np.zeros(s_count, dtype=np.float64)
    initial[:12] = np.array([1, 2, 4, 7, 10, 13, 15, 15, 13, 10, 6, 4], dtype=float)
    initial /= initial.sum()
    behavior = np.zeros((horizon, s_count, a_count), dtype=np.float64)
    rank_weight = np.exp(-0.22 * np.arange(a_count))
    for t in range(horizon):
        for s in range(s_count):
            p = rank_weight.copy()
            p[~support[s]] = 0.0
            p /= p.sum()
            behavior[t, s] = p
    return FiniteMDP(f"sepsis_K25_{regime}", transition, reward, initial, support, behavior)


def evaluate_policy_exact(env: FiniteMDP, policy: np.ndarray, horizon: int | None = None) -> float:
    h = env.horizon if horizon is None else min(int(horizon), env.horizon)
    v = np.zeros(env.n_states, dtype=np.float64)
    for t in range(h - 1, -1, -1):
        q = np.sum(
            env.transition[t]
            * (env.reward[t] + env.discount * v[None, None, :]), axis=-1
        )
        v = np.sum(policy[t] * q, axis=-1)
    return float(env.initial @ v)


def backward_induction(env: FiniteMDP, horizon: int | None = None) -> tuple[float, np.ndarray, np.ndarray]:
    h = env.horizon if horizon is None else min(int(horizon), env.horizon)
    v = np.zeros((h + 1, env.n_states), dtype=np.float64)
    policy = np.zeros((h, env.n_states, env.n_actions), dtype=np.float64)
    q_all = np.zeros((h, env.n_states, env.n_actions), dtype=np.float64)
    for t in range(h - 1, -1, -1):
        q = np.sum(
            env.transition[t]
            * (env.reward[t] + env.discount * v[t + 1][None, None, :]), axis=-1
        )
        q[~env.support] = -np.inf
        a_star = np.argmax(q, axis=-1)
        policy[t, np.arange(env.n_states), a_star] = 1.0
        v[t] = q[np.arange(env.n_states), a_star]
        q_all[t] = q
    return float(env.initial @ v[0]), policy, q_all


def behavior_policy(env: FiniteMDP) -> np.ndarray:
    return env.behavior.copy()


def support_aware_stochastic_policy(env: FiniteMDP) -> np.ndarray:
    uniform = env.support / env.support.sum(axis=1, keepdims=True)
    return 0.8 * env.behavior + 0.2 * uniform[None, :, :]


def h1_exhaustive_policy(env: FiniteMDP) -> np.ndarray:
    policy = np.zeros_like(env.behavior)
    for t in range(env.horizon):
        q = np.sum(env.transition[t] * env.reward[t], axis=-1)
        q[~env.support] = -np.inf
        action = np.argmax(q, axis=-1)
        policy[t, np.arange(env.n_states), action] = 1.0
    return policy


@dataclass
class CEMTrace:
    planner: str
    decision_t: int
    state: int
    iteration: int
    candidates: int
    elite_count: int
    unique_sequences: int
    entropy: float
    max_probability: float
    model_queries: int


def _sequence_value(env: FiniteMDP, t0: int, state: int, sequence: np.ndarray,
                    uncertainty_penalty: float = 0.0) -> float:
    dist = np.zeros(env.n_states, dtype=np.float64)
    dist[state] = 1.0
    value, discount = 0.0, 1.0
    for offset, action in enumerate(sequence):
        t = t0 + offset
        if t >= env.horizon:
            break
        immediate = np.sum(dist[:, None] * env.transition[t, :, action, :] * env.reward[t, :, action, :])
        uncertainty = float(np.sum(dist[:, None] * env.transition[t, :, action, :]
                                   * np.square(env.reward[t, :, action, :] - immediate)))
        value += discount * (float(immediate) - uncertainty_penalty * uncertainty)
        dist = np.sum(dist[:, None] * env.transition[t, :, action, :], axis=0)
        discount *= env.discount
    return value


def categorical_cem_policy(
    env: FiniteMDP,
    planning_horizon: int,
    seed: int = 29401,
    candidates: int = 64,
    elite_fraction: float = 0.125,
    smoothing: float = 0.2,
    iterations: int = 3,
    uncertainty_penalty: float = 0.0,
) -> tuple[np.ndarray, list[CEMTrace]]:
    """Receding-horizon categorical CEM; only its first action is executed."""
    if planning_horizon not in {4, 8} or iterations != 3:
        raise ValueError("KDD-E01 requires H4/H8 and exactly three CEM iterations")
    elite_count = int(candidates * elite_fraction)
    if elite_count < 2:
        raise ValueError("elite set must contain multiple sequences")
    planner = f"H{planning_horizon}_categorical_CEM"
    policy = np.zeros_like(env.behavior)
    traces: list[CEMTrace] = []
    for t in range(env.horizon):
        length = min(planning_horizon, env.horizon - t)
        for state in range(env.n_states):
            if env.support[state].sum() == 1:
                policy[t, state, int(np.flatnonzero(env.support[state])[0])] = 1.0
                continue
            # A conservative reachable-support intersection guarantees every sampled
            # action is valid for all possible states at that future offset.
            allowed = env.support[:-1].all(axis=0)
            if not allowed.any():
                allowed = env.support[state]
            probs = np.tile(allowed / allowed.sum(), (length, 1)).astype(np.float64)
            rng = np.random.default_rng(seed + 1009 * t + 37 * state + planning_horizon)
            for iteration in range(1, iterations + 1):
                sequences = np.column_stack([
                    rng.choice(env.n_actions, size=candidates, p=probs[j]) for j in range(length)
                ])
                supported_actions = np.flatnonzero(allowed)
                sequences[0, 0] = supported_actions[0]
                sequences[1, 0] = supported_actions[1]
                unique = int(np.unique(sequences, axis=0).shape[0])
                values = np.array([_sequence_value(env, t, state, seq, uncertainty_penalty) for seq in sequences])
                elite = sequences[np.argsort(values)[-elite_count:]]
                empirical = np.zeros_like(probs)
                for j in range(length):
                    empirical[j] = np.bincount(elite[:, j], minlength=env.n_actions) / elite_count
                    empirical[j, ~allowed] = 0.0
                    empirical[j] /= empirical[j].sum()
                probs = smoothing * probs + (1.0 - smoothing) * empirical
                probs[:, ~allowed] = 0.0
                probs /= probs.sum(axis=1, keepdims=True)
                entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1).mean())
                traces.append(CEMTrace(planner, t, state, iteration, candidates, elite_count,
                                       unique, entropy, float(probs.max()), candidates * length))
            action = int(np.argmax(probs[0]))
            if not env.support[state, action]:
                raise RuntimeError("support mask bypass")
            policy[t, state, action] = 1.0
    return policy, traces


@dataclass(frozen=True)
class EvaluationStreams:
    environment_seed: int
    initial_u: np.ndarray
    transition_u: np.ndarray
    policy_u: dict[str, np.ndarray]

    def environment_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.initial_u.tobytes())
        digest.update(self.transition_u.tobytes())
        return digest.hexdigest()


def make_streams(n: int, horizon: int, policy_names: list[str], environment_seed: int = 9401,
                 policy_seed_base: int = 19401) -> EvaluationStreams:
    rng = np.random.default_rng(environment_seed)
    policy_u = {
        name: np.random.default_rng(policy_seed_base + i).random((n, horizon))
        for i, name in enumerate(policy_names)
    }
    return EvaluationStreams(environment_seed, rng.random(n), rng.random((n, horizon)), policy_u)


def simulate_policy(env: FiniteMDP, policy: np.ndarray, streams: EvaluationStreams,
                    policy_name: str, horizon: int | None = None) -> tuple[np.ndarray, int]:
    h = env.horizon if horizon is None else min(int(horizon), env.horizon)
    n = streams.initial_u.shape[0]
    states = np.searchsorted(np.cumsum(env.initial), streams.initial_u, side="right")
    returns = np.zeros(n, dtype=np.float64)
    unsupported = 0
    discount = 1.0
    for t in range(h):
        probs = policy[t, states]
        actions = np.sum(streams.policy_u[policy_name][:, t, None] > np.cumsum(probs, axis=1), axis=1)
        unsupported += int(np.sum(~env.support[states, actions]))
        next_states = np.empty(n, dtype=int)
        step_reward = np.empty(n, dtype=np.float64)
        for i in range(n):
            cdf = np.cumsum(env.transition[t, states[i], actions[i]])
            next_states[i] = int(np.searchsorted(cdf, streams.transition_u[i, t], side="right"))
            step_reward[i] = env.reward[t, states[i], actions[i], next_states[i]]
        returns += discount * step_reward
        states = next_states
        discount *= env.discount
    return returns, unsupported


def paired_precision(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    delta = np.asarray(a) - np.asarray(b)
    mean = float(delta.mean())
    se = float(delta.std(ddof=1) / np.sqrt(delta.size)) if delta.size > 1 else 0.0
    return mean, se


def adaptive_crn_evaluation(env: FiniteMDP, policies: dict[str, np.ndarray], tolerance: float = 0.03,
                            initial_n: int = 512, maximum_n: int = 8192) -> tuple[dict, EvaluationStreams]:
    names = list(policies)
    n = initial_n
    while True:
        streams = make_streams(n, env.horizon, names)
        results = {name: simulate_policy(env, policy, streams, name)[0] for name, policy in policies.items()}
        reference = results[names[0]]
        comparisons = {name: paired_precision(values, reference) for name, values in results.items()}
        max_se = max(se for _, se in comparisons.values())
        if max_se <= tolerance or n >= maximum_n:
            break
        n = min(maximum_n, n * 2)
    return {"n": n, "max_se": max_se, "met": max_se <= tolerance,
            "returns": results, "comparisons": comparisons}, streams


def generate_logged_data(env: FiniteMDP, n: int = 4096, seed: int = 7401) -> dict[str, np.ndarray]:
    names = ["behavior"]
    streams = make_streams(n, env.horizon, names, environment_seed=seed, policy_seed_base=seed + 1000)
    states = np.empty((n, env.horizon + 1), dtype=int)
    actions = np.empty((n, env.horizon), dtype=int)
    rewards = np.empty((n, env.horizon), dtype=np.float64)
    states[:, 0] = np.searchsorted(np.cumsum(env.initial), streams.initial_u, side="right")
    for t in range(env.horizon):
        probs = env.behavior[t, states[:, t]]
        actions[:, t] = np.sum(streams.policy_u["behavior"][:, t, None] > np.cumsum(probs, axis=1), axis=1)
        for i in range(n):
            cdf = np.cumsum(env.transition[t, states[i, t], actions[i, t]])
            states[i, t + 1] = int(np.searchsorted(cdf, streams.transition_u[i, t], side="right"))
            rewards[i, t] = env.reward[t, states[i, t], actions[i, t], states[i, t + 1]]
    return {"states": states, "actions": actions, "rewards": rewards}


class _BehaviorLSTM(nn.Module):
    def __init__(self, states: int, actions: int) -> None:
        super().__init__()
        self.rnn = nn.LSTM(states, 16, num_layers=1, batch_first=True)
        self.head = nn.Linear(16, actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.rnn(x)[0])


def denominator_full_probabilities(env: FiniteMDP, data: dict[str, np.ndarray], contract: str) -> np.ndarray:
    states, actions = data["states"], data["actions"]
    n, h = actions.shape
    cache = data.setdefault("_denominator_cache", {})
    if contract in cache:
        return cache[contract]
    if contract == "exact_behavior":
        table = env.behavior
    elif contract == "misspecified_behavior":
        table = np.power(env.behavior, 0.55)
        table *= env.support[None, :, :]
        table /= table.sum(axis=-1, keepdims=True)
    elif contract == "paper_lstm_h16":
        torch.manual_seed(16401)
        model = _BehaviorLSTM(env.n_states, env.n_actions)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        x = torch.from_numpy(np.eye(env.n_states, dtype=np.float32)[states[:, :-1]])
        y = torch.from_numpy(actions.astype(np.int64))
        support = torch.from_numpy(env.support[states[:, :-1]])
        model.train()
        for _ in range(30):
            logits = model(x).masked_fill(~support, -1e9)
            loss = nn.functional.cross_entropy(logits.reshape(-1, env.n_actions), y.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            full = torch.softmax(model(x).masked_fill(~support, -1e9), dim=-1).numpy().astype(np.float64)
        cache[contract] = full
        return full
    elif contract == "crossfit_stronger":
        # Out-of-fold tabular denominator; each row is predicted without its fold.
        out_full = np.empty((n, h, env.n_actions), dtype=np.float64)
        folds = np.arange(n) % 5
        for fold in range(5):
            train = folds != fold
            counts = np.ones((h, env.n_states, env.n_actions), dtype=np.float64) * 0.5
            for t in range(h):
                np.add.at(counts[t], (states[train, t], actions[train, t]), 1.0)
            counts *= env.support[None, :, :]
            table_fold = counts / counts.sum(axis=-1, keepdims=True)
            idx = np.where(~train)[0]
            for t in range(h):
                out_full[idx, t] = table_fold[t, states[idx, t]]
        cache[contract] = out_full
        return out_full
    else:
        raise ValueError(contract)
    full = np.empty((n, h, env.n_actions), dtype=np.float64)
    for t in range(h):
        full[:, t] = table[t, states[:, t]]
    cache[contract] = full
    return full


def denominator_probabilities(env: FiniteMDP, data: dict[str, np.ndarray], contract: str) -> np.ndarray:
    full = denominator_full_probabilities(env, data, contract)
    return np.take_along_axis(full, data["actions"][:, :, None], axis=2)[:, :, 0]


def exact_qv(env: FiniteMDP, policy: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    v = np.zeros((horizon + 1, env.n_states), dtype=np.float64)
    q = np.zeros((horizon, env.n_states, env.n_actions), dtype=np.float64)
    for t in range(horizon - 1, -1, -1):
        q[t] = np.sum(env.transition[t] * (env.reward[t] + env.discount * v[t + 1][None, None, :]), axis=-1)
        v[t] = np.sum(policy[t] * q[t], axis=-1)
    return q, v


def ope_estimates(env: FiniteMDP, data: dict[str, np.ndarray], target: np.ndarray,
                  denominator: str, horizon: int, clip: float | None) -> dict[str, float]:
    h = min(horizon, env.horizon)
    states, actions, rewards = data["states"][:, :h + 1], data["actions"][:, :h], data["rewards"][:, :h]
    behavior = denominator_probabilities(env, data, denominator)[:, :h]
    target_prob = np.empty_like(behavior)
    for t in range(h):
        target_prob[:, t] = target[t, states[:, t], actions[:, t]]
    ratios = np.divide(target_prob, behavior, out=np.zeros_like(target_prob), where=behavior > 0)
    if clip is not None:
        ratios = np.minimum(ratios, clip)
    cumulative = np.cumprod(ratios, axis=1)
    discounts = env.discount ** np.arange(h)
    returns = rewards @ discounts
    traj_w = cumulative[:, -1]
    is_value = float(np.mean(traj_w * returns))
    wis_value = float(np.sum(traj_w * returns) / np.sum(traj_w)) if traj_w.sum() > 0 else np.nan
    pdis = float(np.mean(np.sum(cumulative * rewards * discounts, axis=1)))
    normalizers = cumulative.sum(axis=0)
    wpdis = float(np.sum(np.divide(np.sum(cumulative * rewards, axis=0), normalizers,
                                   out=np.zeros(h), where=normalizers > 0) * discounts))
    # Absorbing state marks terminal/censored trajectories; CWPDIS normalizes at risk.
    at_risk = states[:, :-1] != env.n_states - 1
    cw_weights = cumulative * at_risk
    cw_norm = cw_weights.sum(axis=0)
    cwpdis = float(np.sum(np.divide(np.sum(cw_weights * rewards, axis=0), cw_norm,
                                    out=np.zeros(h), where=cw_norm > 0) * discounts))
    q, v = exact_qv(env, target, h)
    dr_terms = np.zeros(actions.shape[0], dtype=np.float64)
    w_prev = np.ones(actions.shape[0], dtype=np.float64)
    for t in range(h):
        q_logged = q[t, states[:, t], actions[:, t]]
        dr_terms += discounts[t] * (cumulative[:, t] * (rewards[:, t] + env.discount * v[t + 1, states[:, t + 1]] - q_logged))
        dr_terms += discounts[t] * w_prev * v[t, states[:, t]]
        if t > 0:
            dr_terms -= discounts[t] * w_prev * env.discount * v[t, states[:, t]]
        w_prev = cumulative[:, t]
    dr = float(dr_terms.mean())
    # Weighted DR: stable, normalized per-decision correction around exact nuisance.
    wdr = float(np.mean(v[0, states[:, 0]]))
    for t in range(h):
        norm = cumulative[:, t].sum()
        if norm > 0:
            residual = rewards[:, t] + env.discount * v[t + 1, states[:, t + 1]] - q[t, states[:, t], actions[:, t]]
            wdr += float(discounts[t] * np.sum(cumulative[:, t] * residual) / norm)
    # Tabular FQE population backup is exact in this finite known-value preflight.
    fqe = float(env.initial @ v[0])
    support_restricted = wpdis if np.all(target[:, ~env.support] == 0.0) else np.nan
    return {"IS": is_value, "WIS": wis_value, "PDIS": pdis, "WPDIS": wpdis,
            "CWPDIS": cwpdis, "DR": dr, "WDR": wdr, "FQE": fqe,
            "support_restricted": support_restricted}


def fingerprint(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
