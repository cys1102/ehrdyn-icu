from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, stdev
from typing import Callable

from .errors import ReleaseContractError


History = tuple[tuple[float, ...], tuple[int, ...], tuple[int, ...], int, int]


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _softmax(values: list[float]) -> list[float]:
    top = max(values)
    weights = [math.exp(value - top) for value in values]
    total = sum(weights)
    return [value / total for value in weights]


def _sample(probabilities: list[float], uniform: float) -> int:
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if uniform <= cumulative:
            return index
    return len(probabilities) - 1


def _coordinates(action_count: int) -> list[tuple[int, ...]]:
    if action_count == 25:
        return [(first, second) for first in range(5) for second in range(5)]
    if action_count == 4:
        return [(first, second) for first in range(2) for second in range(2)]
    if action_count == 2:
        return [(value,) for value in range(2)]
    raise ReleaseContractError(f"Unsupported public POMDP action count: {action_count}")


@dataclass(frozen=True)
class PublicPOMDPConfig:
    mechanism_version: str
    profile: str
    action_count: int
    supported_actions: tuple[int, ...]
    feature_count: int
    horizon: int
    discount: float
    latent_states: int
    latent_subtypes: int
    observation_loading: float
    observation_noise: float
    missingness_base: float
    response_strength: float
    delayed_fraction: float
    action_cost: float
    toxicity: float
    switching_cost: float
    terminal_hazards: tuple[float, ...]

    @classmethod
    def load(cls, path: Path, profile: str) -> "PublicPOMDPConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        profiles = raw.get("profiles", {})
        if profile not in profiles:
            raise ReleaseContractError(f"Unknown public POMDP profile: {profile}")
        merged = dict(raw["generator"])
        merged.update(profiles[profile])
        return cls(
            mechanism_version=str(raw["mechanism_version"]),
            profile=profile,
            action_count=int(merged["action_count"]),
            supported_actions=tuple(int(value) for value in merged["supported_actions"]),
            feature_count=int(merged["feature_count"]),
            horizon=int(merged["horizon"]),
            discount=float(merged["discount"]),
            latent_states=int(merged["latent_states"]),
            latent_subtypes=int(merged["latent_subtypes"]),
            observation_loading=float(merged["observation_loading"]),
            observation_noise=float(merged["observation_noise"]),
            missingness_base=float(merged["missingness_base"]),
            response_strength=float(merged["response_strength"]),
            delayed_fraction=float(merged["delayed_fraction"]),
            action_cost=float(merged["action_cost"]),
            toxicity=float(merged["toxicity"]),
            switching_cost=float(merged["switching_cost"]),
            terminal_hazards=tuple(float(value) for value in merged["terminal_hazards"]),
        )

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), default=list)
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class StepNoise:
    observation: tuple[float, ...]
    mask: tuple[float, ...]
    transition: float
    termination: float
    outcome: float
    dense_sign: float


@dataclass
class LoggedStep:
    context: int
    action: int
    reward: float
    behavior_probability: float
    next_context: int
    terminal: bool


class PublicPOMDP:
    """Aggregate-safe KDD198-v2-style finite POMDP with episode-local noise."""

    def __init__(self, config: PublicPOMDPConfig, environment_seed: int):
        self.config = config
        self.environment_seed = int(environment_seed)
        self.coordinates = _coordinates(config.action_count)
        rng = random.Random(_stable_seed("environment", environment_seed, config.profile))
        raw = [rng.gammavariate(value, 1.0) for value in (5.0, 7.0, 5.0)]
        self.subtype_probability = [value / sum(raw) for value in raw]
        self.feature_offsets = [rng.uniform(-0.025, 0.025) for _ in range(config.feature_count)]
        self.response_multiplier = rng.uniform(0.9, 1.1)
        self.cost_multiplier = rng.uniform(0.9, 1.1)
        payload = {
            "config": config.fingerprint,
            "environment_seed": environment_seed,
            "subtype_probability": self.subtype_probability,
            "feature_offsets": self.feature_offsets,
            "response_multiplier": self.response_multiplier,
            "cost_multiplier": self.cost_multiplier,
        }
        self.mechanism_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def ideal_action(self, state: int, subtype: int) -> int:
        if self.config.action_count == 25:
            proposed = max(0, min(4, state + subtype - 1)) * 5 + max(0, min(4, state - subtype + 1))
        elif self.config.action_count == 4:
            proposed = int(state >= (3 if subtype == 0 else 2)) * 2 + int(state >= (3 if subtype == 2 else 2))
        else:
            proposed = int(state >= (3 if subtype == 0 else 2 if subtype == 1 else 1))
        return min(self.config.supported_actions, key=lambda action: abs(action - proposed))

    def _mismatch(self, state: int, subtype: int, action: int) -> float:
        ideal = self.coordinates[self.ideal_action(state, subtype)]
        actual = self.coordinates[action]
        scales = [max(max(values[index] for values in self.coordinates), 1) for index in range(len(actual))]
        return fmean(((a - b) / scale) ** 2 for a, b, scale in zip(actual, ideal, scales))

    def _severity_context(self, observation: tuple[float, ...], mask: tuple[int, ...], previous: int) -> int:
        observed = [value * (1 if index % 2 == 0 else -1) for index, value in enumerate(observation) if mask[index]]
        score = fmean(observed) if observed else 0.0
        severity = max(0, min(4, round((score / max(self.config.observation_loading, 1e-6) + 1.0) * 2.0)))
        return int(severity) * self.config.action_count + int(previous)

    def _noise(self, episode_seed: int) -> tuple[int, int, list[StepNoise]]:
        rng = random.Random(_stable_seed("episode", self.environment_seed, episode_seed))
        state = _sample([0.12, 0.20, 0.36, 0.22, 0.10], rng.random())
        subtype = _sample(self.subtype_probability, rng.random())
        values = []
        for _ in range(self.config.horizon):
            values.append(StepNoise(
                tuple(rng.gauss(0.0, 1.0) for _ in range(self.config.feature_count)),
                tuple(rng.random() for _ in range(self.config.feature_count)),
                rng.random(), rng.random(), rng.random(), rng.random(),
            ))
        return state, subtype, values

    def _emit(self, state: int, subtype: int, previous: int, previous_mask: list[int], recency: list[int], noise: StepNoise) -> tuple[tuple[float, ...], tuple[int, ...], tuple[int, ...]]:
        observation, mask, updated = [], [], []
        for index in range(self.config.feature_count):
            sign = 1.0 if index % 2 == 0 else -1.0
            mean = sign * self.config.observation_loading * (2.0 * state / 4.0 - 1.0) + 0.30 * (subtype - 1) + self.feature_offsets[index]
            missing = min(0.98, max(0.02, self.config.missingness_base + 0.04 * (state / 4.0 - 0.5) + 0.04 * (0.5 - previous_mask[index]) + 0.01 * min(recency[index], 8)))
            present = int(noise.mask[index] >= missing)
            mask.append(present)
            observation.append(mean + self.config.observation_noise * noise.observation[index] if present else 0.0)
            updated.append(0 if present else recency[index] + 1)
        return tuple(observation), tuple(mask), tuple(updated)

    def _transition(self, state: int, subtype: int, previous: int, action: int, uniform: float) -> int:
        delayed = 0.5 - self._mismatch(state, subtype, previous)
        current = 0.5 - self._mismatch(state, subtype, action)
        score = self.config.response_strength * self.response_multiplier * (
            self.config.delayed_fraction * delayed + (1.0 - self.config.delayed_fraction) * current
        )
        move = _sample(_softmax([-0.45 + score, 0.65, -0.45 - score]), uniform) - 1
        return max(0, min(self.config.latent_states - 1, state + move))

    def _reward(self, state: int, subtype: int, previous: int, action: int, dense_sign: float) -> float:
        intensity = fmean(coordinate / max(max(values[index] for values in self.coordinates), 1) for index, coordinate in enumerate(self.coordinates[action]))
        response = 0.5 - self._mismatch(state, subtype, action)
        dense = 0.05 * ((1.0 if dense_sign < 0.5 else -1.0) + 0.10 * math.tanh(response))
        return dense - self.cost_multiplier * (
            self.config.action_cost * intensity
            + self.config.toxicity * intensity * max(0.0, 1.0 - state / 4.0)
            + self.config.switching_cost * float(action != previous)
        )

    def run_episode(self, episode_seed: int, policy: Callable[[History, random.Random], tuple[int, list[float]]]) -> tuple[float, list[LoggedStep]]:
        state, subtype, noises = self._noise(episode_seed)
        policy_rng = random.Random(_stable_seed("policy", episode_seed))
        previous = self.config.supported_actions[0]
        previous_mask = [1] * self.config.feature_count
        recency = [0] * self.config.feature_count
        total, records = 0.0, []
        for time, noise in enumerate(noises):
            observation, mask, recency_tuple = self._emit(state, subtype, previous, previous_mask, recency, noise)
            context = self._severity_context(observation, mask, previous)
            action, probabilities = policy((observation, mask, recency_tuple, previous, time), policy_rng)
            if action not in self.config.supported_actions or len(probabilities) != self.config.action_count:
                raise ReleaseContractError("Policy returned an invalid public POMDP action distribution")
            next_state = self._transition(state, subtype, previous, action, noise.transition)
            reward = self._reward(state, subtype, previous, action, noise.dense_sign)
            hazard = 1.0 if time == self.config.horizon - 1 else self.config.terminal_hazards[time]
            terminal = noise.termination < hazard
            if terminal:
                death_probability = 1.0 / (1.0 + math.exp(-(next_state - 2.0)))
                reward += -1.0 if noise.outcome < death_probability else 1.0
            next_context = next_state * self.config.action_count + action
            records.append(LoggedStep(context, action, reward, probabilities[action], next_context, terminal))
            total += self.config.discount**time * reward
            previous_mask, recency = list(mask), list(recency_tuple)
            state, previous = next_state, action
            if terminal:
                break
        return total, records


def behavior_policy(environment: PublicPOMDP) -> Callable[[History, random.Random], tuple[int, list[float]]]:
    supported = environment.config.supported_actions
    def policy(history: History, rng: random.Random) -> tuple[int, list[float]]:
        previous = history[3]
        probabilities = [0.0] * environment.config.action_count
        for action in supported:
            probabilities[action] = 0.45 / len(supported)
        probabilities[previous] += 0.55
        action = _sample(probabilities, rng.random())
        return action, probabilities
    return policy


def fit_behavior_cloning(environment: PublicPOMDP, trajectories: list[list[LoggedStep]]) -> Callable[[History, random.Random], tuple[int, list[float]]]:
    counts: dict[int, list[float]] = {}
    for trajectory in trajectories:
        for row in trajectory:
            counts.setdefault(row.context, [1.0 if action in environment.config.supported_actions else 0.0 for action in range(environment.config.action_count)])[row.action] += 1.0
    def policy(history: History, rng: random.Random) -> tuple[int, list[float]]:
        context = environment._severity_context(history[0], history[1], history[3])
        local = counts.get(context, [1.0 if action in environment.config.supported_actions else 0.0 for action in range(environment.config.action_count)])
        total = sum(local)
        probabilities = [value / total for value in local]
        return _sample(probabilities, rng.random()), probabilities
    return policy


def fit_h4_component_planner(environment: PublicPOMDP, trajectories: list[list[LoggedStep]], seed: int) -> tuple[Callable[[History, random.Random], tuple[int, list[float]]], dict[str, int | float]]:
    rewards: dict[tuple[int, int], list[float]] = {}
    next_contexts: dict[tuple[int, int], list[int]] = {}
    for trajectory in trajectories:
        for row in trajectory:
            rewards.setdefault((row.context, row.action), []).append(row.reward)
            next_contexts.setdefault((row.context, row.action), []).append(row.next_context)
    trace = {"horizon": 4, "cem_iterations": 3, "candidates_per_iteration": 64, "elites": 8, "smoothing": 0.2, "sequences_evaluated": 0, "executed_actions": 0}
    supported = list(environment.config.supported_actions)
    def score(context: int, sequence: list[int]) -> float:
        value = 0.0
        for depth, action in enumerate(sequence):
            key = (context, action)
            value += environment.config.discount**depth * (fmean(rewards[key]) if key in rewards else -0.1)
            if key in next_contexts:
                context = round(fmean(next_contexts[key]))
        return value
    def policy(history: History, rng: random.Random) -> tuple[int, list[float]]:
        del rng
        context = environment._severity_context(history[0], history[1], history[3])
        local_rng = random.Random(_stable_seed("h4", seed, context, history[4], history[3]))
        categorical = [[1.0 / len(supported)] * len(supported) for _ in range(4)]
        for _ in range(3):
            candidates = []
            for _candidate in range(64):
                sequence = [supported[_sample(categorical[depth], local_rng.random())] for depth in range(4)]
                candidates.append((score(context, sequence), sequence))
            trace["sequences_evaluated"] = int(trace["sequences_evaluated"]) + len(candidates)
            elites = sorted(candidates, reverse=True)[:8]
            for depth in range(4):
                frequency = [sum(item[1][depth] == action for item in elites) / 8.0 for action in supported]
                categorical[depth] = [0.2 * old + 0.8 * new for old, new in zip(categorical[depth], frequency)]
        probabilities = [0.0] * environment.config.action_count
        for local, action in enumerate(supported):
            probabilities[action] = categorical[0][local]
        chosen = max(supported, key=lambda action: probabilities[action])
        trace["executed_actions"] = int(trace["executed_actions"]) + 1
        return chosen, probabilities
    return policy, trace


def run_public_pomdp_smoke(config_path: Path, profile: str, environment_seed: int, episodes: int, seed: int) -> dict[str, object]:
    if episodes < 8:
        raise ReleaseContractError("POMDP smoke requires at least eight episodes")
    config = PublicPOMDPConfig.load(config_path, profile)
    environment = PublicPOMDP(config, environment_seed)
    behavior = behavior_policy(environment)
    training = [environment.run_episode(seed + index, behavior)[1] for index in range(128)]
    bc = fit_behavior_cloning(environment, training)
    planner, trace = fit_h4_component_planner(environment, training, seed)
    policies = {"behavior": behavior, "behavior_cloning": bc, "tabular_component_plus_h4": planner}
    rows = []
    for name, policy in policies.items():
        returns = [environment.run_episode(seed + 10_000 + index, policy)[0] for index in range(episodes)]
        rows.append({"method": name, "mean_return": fmean(returns), "return_se": stdev(returns) / math.sqrt(len(returns))})
    reconstruction = PublicPOMDP(config, environment_seed)
    return {
        "mechanism_version": config.mechanism_version,
        "profile": profile,
        "environment_seed": environment_seed,
        "episode_seed_namespace": seed + 10_000,
        "training_seed_namespace": seed,
        "environment_hash": environment.mechanism_hash,
        "reconstruction_hash_equal": reconstruction.mechanism_hash == environment.mechanism_hash,
        "policies": rows,
        "planner_trace": trace,
        "claim_boundary": "constructed POMDP truth only; no EHR treatment or clinical claim",
    }
