from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


def _softmax(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    value = value - np.max(value)
    output = np.exp(value)
    return output / output.sum()


def action_coordinates(action_count: int) -> np.ndarray:
    if action_count == 25:
        return np.asarray([(i / 4, j / 4) for i in range(5) for j in range(5)], dtype=np.float64)
    if action_count == 4:
        return np.asarray([(i, j) for i in range(2) for j in range(2)], dtype=np.float64)
    if action_count == 2:
        return np.asarray([[0.0], [1.0]], dtype=np.float64)
    raise ValueError(action_count)


@dataclass(frozen=True, slots=True)
class DenseRewardTarget:
    name: str
    variance: float
    availability_or_nonzero_fraction: float
    fraction_semantics: str
    primary: bool


@dataclass(frozen=True, slots=True)
class ProfileContract:
    profile: str
    action_count: int
    supported_actions: tuple[int, ...]
    feature_dim: int
    target_missingness: float
    target_mean_horizon: float
    target_early_termination: float
    target_action_frequency: tuple[float, ...]
    target_conditional_max: float
    target_conditional_entropy: float
    target_switch_rate: float
    termination_hazards: tuple[float, ...]
    primary_reward: str
    primary_reward_type: str
    terminal_event_prevalence: float | None
    terminal_reward_minimum: float | None
    terminal_reward_maximum: float | None
    dense_targets: tuple[DenseRewardTarget, ...]


@dataclass(frozen=True, slots=True)
class BehaviorCalibration:
    profile: str
    supported_actions: tuple[int, ...]
    transition_matrix: tuple[tuple[float, ...], ...]
    context_sharpening: float
    conditional_blend_used: float
    context_bins: int
    optimizer_steps: int
    fingerprint: str

    @property
    def matrix(self) -> np.ndarray:
        return np.asarray(self.transition_matrix, dtype=np.float64)

    def distribution(self, previous_action: int, context_bin: int, action_count: int) -> np.ndarray:
        supported = np.asarray(self.supported_actions, dtype=int)
        if previous_action in self.supported_actions:
            row_index = self.supported_actions.index(previous_action)
        else:
            row_index = 0
        base = self.matrix[row_index]
        quantile = (int(context_bin) + 0.5) / self.context_bins
        context_index = int(np.searchsorted(np.cumsum(base), quantile, side="right"))
        context_index = min(context_index, len(supported) - 1)
        local = (1.0 - self.context_sharpening) * base
        local[context_index] += self.context_sharpening
        output = np.zeros(action_count, dtype=np.float64)
        output[supported] = local
        output[supported[-1]] = 1.0 - output[supported[:-1]].sum()
        return output

    def context_bin(self, observation: np.ndarray, mask: np.ndarray, recency: np.ndarray,
                    prior_action: np.ndarray, time: int) -> np.ndarray:
        """Map observable history channels to a context; no simulator object is available."""
        signs = np.where(np.arange(observation.shape[1]) % 2 == 0, 1.0, -1.0)
        signed = np.sum(observation * signs[None, :], axis=1)
        score = signed + 0.17 * mask.sum(axis=1) + 0.031 * recency.sum(axis=1) + 0.113 * prior_action + 0.071 * time
        unit = np.mod(np.sin(score * 12.9898 + 78.233) * 43758.5453, 1.0)
        return np.minimum((unit * self.context_bins).astype(int), self.context_bins - 1)


def behavior_summary(contract: ProfileContract, calibration: BehaviorCalibration) -> dict[str, float]:
    supported = np.asarray(calibration.supported_actions, dtype=int)
    marginal_full = np.asarray(contract.target_action_frequency, dtype=np.float64)
    marginal = marginal_full[supported]
    marginal = marginal / marginal.sum()
    base = calibration.matrix
    q = calibration.context_sharpening
    eye = np.eye(len(supported), dtype=np.float64)
    distributions = (1.0 - q) * base[:, None, :] + q * eye[None, :, :]
    weights = marginal[:, None] * base
    observed_marginal = marginal @ base
    max_probability = float(np.sum(weights * distributions.max(axis=-1)))
    entropy = -np.sum(distributions * np.log(np.maximum(distributions, 1e-15)), axis=-1) / math.log(contract.action_count)
    conditional_entropy = float(np.sum(weights * entropy))
    switch_rate = float(1.0 - np.sum(marginal * np.diag(base)))
    return {
        "action_frequency_tv": float(0.5 * np.abs(observed_marginal - marginal).sum()),
        "conditional_max_probability": max_probability,
        "conditional_normalized_entropy": conditional_entropy,
        "action_switch_rate": switch_rate,
        "supported_action_count": int(len(supported)),
        "probability_sum_error": float(np.max(np.abs(distributions.sum(axis=-1) - 1.0))),
    }


class ObservationHistoryPolicy:
    """Frozen observation-only reference; no latent state/subtype/environment object."""

    latent_state_access = False
    latent_subtype_access = False
    transition_truth_access = False

    def __init__(self, feature_dim: int, action_count: int, supported_actions: tuple[int, ...]):
        self.feature_dim = int(feature_dim)
        self.action_count = int(action_count)
        self.supported_actions = tuple(int(value) for value in supported_actions)
        self.weights = np.where(np.arange(feature_dim) % 2 == 0, 1.0, -1.0).astype(np.float64)

    def update_belief(self, prior_belief: np.ndarray, observation: np.ndarray,
                      mask: np.ndarray, recency: np.ndarray, prior_action: np.ndarray) -> np.ndarray:
        signed = observation * self.weights[None, :]
        count = np.maximum(mask.sum(axis=1), 1)
        current = np.sum(signed * mask, axis=1) / count
        recency_penalty = np.clip(np.mean(recency, axis=1) / 20.0, 0.0, 0.25)
        action_memory = prior_action / max(self.action_count - 1, 1)
        estimate = np.clip((current + 1.0) * 2.0 - recency_penalty + 0.05 * action_memory, 0.0, 4.0)
        return 0.65 * prior_belief + 0.35 * estimate

    def actions(self, belief: np.ndarray) -> np.ndarray:
        severity = np.clip(np.rint(belief), 0, 4).astype(int)
        if self.action_count == 25:
            proposed = severity * 5 + severity
        elif self.action_count == 4:
            proposed = np.where(severity >= 2, 3, 0)
        else:
            proposed = (severity >= 2).astype(int)
        supported = np.asarray(self.supported_actions, dtype=int)
        return np.asarray([supported[np.argmin(np.abs(supported - value))] for value in proposed], dtype=np.int16)


@dataclass(frozen=True, slots=True)
class FrozenProfileParameters:
    terminal_logit_intercept: float | None
    missingness_offset: float
    dense_scales: tuple[tuple[str, float], ...]
    dense_centers: tuple[tuple[str, float], ...]
    fingerprint: str

    def scale(self, name: str) -> float:
        return dict(self.dense_scales)[name]

    def center(self, name: str) -> float:
        return dict(self.dense_centers)[name]


class RepairedPOMDP:
    def __init__(self, contract: ProfileContract, behavior: BehaviorCalibration, seed: int,
                 generator: dict[str, Any], frozen: FrozenProfileParameters | None = None):
        self.contract = contract; self.behavior = behavior; self.seed = int(seed); self.generator = generator
        rng = np.random.default_rng(seed)
        self.coordinates = action_coordinates(contract.action_count)
        self.latent_states = int(generator["latent_states"]); self.latent_subtypes = int(generator["latent_subtypes"])
        self.horizon = int(generator["horizon"]); self.discount = float(generator["discount"])
        self.subtype_prevalence = rng.dirichlet(np.full(self.latent_subtypes, float(generator["subtype_prevalence_concentration"]) / self.latent_subtypes))
        self.observation_noise = float(rng.uniform(*generator["observation_noise_range"]))
        self.efficacy = float(rng.uniform(*generator["efficacy_range"])); self.toxicity = float(rng.uniform(*generator["toxicity_range"]))
        self.action_cost = float(rng.uniform(*generator["action_cost_range"])); self.delayed_effect = float(rng.uniform(*generator["delayed_effect_range"]))
        self.switch_cost = float(rng.uniform(*generator["switch_cost_range"])); self.baseline_drift = float(rng.uniform(*generator["baseline_drift_range"]))
        self.initial_state_probability = np.asarray([0.12, 0.20, 0.36, 0.22, 0.10], dtype=np.float64)
        feature_rng = np.random.default_rng(seed + 700_000)
        signs = np.where(np.arange(contract.feature_dim) % 2 == 0, 1.0, -1.0)
        self.loadings = signs * feature_rng.uniform(0.65, 1.0, size=contract.feature_dim)
        self.feature_missing_offset = feature_rng.uniform(-0.04, 0.04, size=contract.feature_dim)
        self.frozen = frozen
        self.mechanism_hash = self._hash()

    def _hash(self) -> str:
        payload = {
            "contract": asdict(self.contract), "behavior": asdict(self.behavior), "seed": self.seed,
            "generator": self.generator, "subtype_prevalence": self.subtype_prevalence.tolist(),
            "observation_noise": self.observation_noise, "efficacy": self.efficacy,
            "toxicity": self.toxicity, "action_cost": self.action_cost,
            "delayed_effect": self.delayed_effect, "switch_cost": self.switch_cost,
            "baseline_drift": self.baseline_drift, "loadings": self.loadings.tolist(),
            "missing_offset": self.feature_missing_offset.tolist(),
            "frozen": asdict(self.frozen) if self.frozen else None,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def ideal_action(self, state: int, subtype: int) -> int:
        if self.contract.action_count == 25:
            first = int(np.clip(state + subtype - 1, 0, 4)); second = int(np.clip(state - subtype + 1, 0, 4)); proposed = first * 5 + second
        elif self.contract.action_count == 4:
            first = int(state >= (3 if subtype == 0 else 2)); second = int(state >= (3 if subtype == 2 else 2)); proposed = first * 2 + second
        else:
            proposed = int(state >= (3, 2, 2)[subtype])
        supported = np.asarray(self.contract.supported_actions, dtype=int)
        return int(supported[np.argmin(np.abs(supported - proposed))])

    def mismatch(self, state: int, subtype: int, action: int) -> float:
        target = self.coordinates[self.ideal_action(state, subtype)]
        return float(np.mean(np.square(self.coordinates[action] - target)))

    def transition_probability(self, state: int, subtype: int, pending_action: int, null_response: bool = False) -> np.ndarray:
        if null_response:
            response = 0.0
        else:
            mismatch = self.mismatch(state, subtype, pending_action)
            intensity = float(np.mean(self.coordinates[pending_action]))
            response = self.efficacy * (0.55 - mismatch) - self.toxicity * intensity * intensity + self.baseline_drift
        local = _softmax(np.asarray([-0.25 + response, 0.55, -0.25 - response]))
        output = np.zeros(self.latent_states, dtype=np.float64)
        for delta, probability in zip((-1, 0, 1), local, strict=True):
            output[int(np.clip(state + delta, 0, self.latent_states - 1))] += probability
        return output

    def base_dense_signal(self, state: int, subtype: int, pending_action: int, action: int, null_response: bool = False) -> float:
        burden = -float(self.generator["dense_burden_scale"]) * state / (self.latent_states - 1)
        if null_response: return burden
        delayed = self.delayed_effect * (0.5 - self.mismatch(state, subtype, pending_action))
        intensity = float(np.mean(self.coordinates[action])); cost = self.action_cost * intensity * intensity
        switch = self.switch_cost * float(np.mean(np.abs(self.coordinates[action] - self.coordinates[pending_action])))
        return burden + delayed - cost - switch

    def death_probability(self, state: int) -> float:
        if self.contract.terminal_event_prevalence is None or self.frozen is None or self.frozen.terminal_logit_intercept is None:
            return 0.0
        logit = self.frozen.terminal_logit_intercept + float(self.generator["terminal_state_logit_slope"]) * (state - 2.0)
        return float(1.0 / (1.0 + math.exp(-logit)))

    def expected_primary_dense(self, state: int, subtype: int, pending: int, action: int, null_response: bool = False) -> float:
        if self.contract.primary_reward_type != "dense" or self.frozen is None: return 0.0
        target = next(value for value in self.contract.dense_targets if value.primary)
        value = 0.1 * self.frozen.scale(target.name) * math.tanh(
            self.base_dense_signal(state, subtype, pending, action, null_response) - self.frozen.center(target.name)
        )
        return target.availability_or_nonzero_fraction * value

    def exact_values(self, policy: str = "oracle", null_response: bool = False) -> tuple[float, np.ndarray]:
        actions = self.contract.action_count; supported = np.asarray(self.contract.supported_actions, dtype=int)
        value = np.zeros((self.horizon + 1, self.latent_states, self.latent_subtypes, actions), dtype=np.float64)
        selected = np.zeros((self.horizon, self.latent_states, self.latent_subtypes, actions), dtype=np.int16)
        for time in range(self.horizon - 1, -1, -1):
            hazard = 1.0 if time == self.horizon - 1 else self.contract.termination_hazards[time]
            for state in range(self.latent_states):
                for subtype in range(self.latent_subtypes):
                    for pending in supported:
                        transition = self.transition_probability(state, subtype, int(pending), null_response)
                        q = np.full(actions, -np.inf, dtype=np.float64)
                        for action in supported:
                            dense = self.expected_primary_dense(state, subtype, int(pending), int(action), null_response)
                            synthetic_cost = 0.05 * self.base_dense_signal(state, subtype, int(pending), int(action), null_response)
                            future = 0.0
                            for next_state, probability in enumerate(transition):
                                terminal = 0.0
                                if self.contract.primary_reward_type == "terminal":
                                    terminal = (1.0 - 2.0 * self.death_probability(next_state))
                                continuation = hazard * terminal + (1.0 - hazard) * value[time + 1, next_state, subtype, action]
                                future += probability * continuation
                            q[action] = dense + synthetic_cost + self.discount * future
                        if policy == "oracle": action = int(np.argmax(q))
                        elif policy == "minimum": action = int(supported[0])
                        elif policy == "maximum": action = int(supported[-1])
                        elif policy == "severity": action = self.ideal_action(state, 1)
                        elif policy == "random":
                            value[time, state, subtype, pending] = float(np.mean(q[supported])); selected[time, state, subtype, pending] = -1; continue
                        else: raise ValueError(policy)
                        selected[time, state, subtype, pending] = action; value[time, state, subtype, pending] = q[action]
        initial_pending = int(supported[len(supported) // 2])
        initial = float(np.sum(value[0, :, :, initial_pending] * self.initial_state_probability[:, None] * self.subtype_prevalence[None, :]))
        return initial, selected

    def emission(self, states: np.ndarray, subtypes: np.ndarray, previous_mask: np.ndarray,
                 recency: np.ndarray, prior_action: np.ndarray, noise: np.ndarray, mask_u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        normalized = states[:, None] / (self.latent_states - 1)
        mean = self.loadings[None, :] * (2 * normalized - 1) + (subtypes[:, None] - 1) * 0.12
        observation = mean + self.observation_noise * noise
        action_intensity = prior_action[:, None] / max(self.contract.action_count - 1, 1)
        observe_probability = (
            1.0 - self.contract.target_missingness + self.feature_missing_offset[None, :]
            + (self.frozen.missingness_offset if self.frozen is not None else 0.0)
            + 0.08 * (normalized - 0.5) * np.sign(self.loadings)[None, :]
            + float(self.generator["mask_prior_observed_effect"]) * (previous_mask.astype(float) - 0.5)
            - float(self.generator["mask_recency_effect"]) * np.minimum(recency, 8.0)
            + float(self.generator["mask_past_action_effect"]) * (action_intensity - 0.5)
        )
        mask = mask_u < np.clip(observe_probability, 0.02, 0.99)
        observation = np.where(mask, observation, 0.0)
        next_recency = np.where(mask, 0.0, recency + 1.0)
        return observation.astype(np.float32), mask, next_recency.astype(np.float32)

    def _streams(self, episodes: int, seed: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        return {
            "subtype_u": rng.random(episodes), "state_u": rng.random(episodes),
            "prior_action_u": rng.random(episodes), "transition_u": rng.random((episodes, self.horizon)),
            "termination_u": rng.random((episodes, self.horizon)), "outcome_u": rng.random((episodes, self.horizon)),
            "policy_u": rng.random((episodes, self.horizon)),
            "noise": rng.normal(size=(episodes, self.horizon, self.contract.feature_dim)),
            "mask_u": rng.random((episodes, self.horizon, self.contract.feature_dim)),
            "dense_u": rng.random((episodes, self.horizon, max(1, len(self.contract.dense_targets)))),
        }

    @staticmethod
    def _draw_categorical(probability: np.ndarray, uniform: np.ndarray) -> np.ndarray:
        return np.sum(uniform[:, None] > np.cumsum(probability, axis=1), axis=1).astype(int)

    def simulate(self, episodes: int, stream_seed: int, policy: str) -> dict[str, Any]:
        streams = self._streams(episodes, stream_seed)
        states = self._draw_categorical(np.broadcast_to(self.initial_state_probability, (episodes, self.latent_states)), streams["state_u"])
        subtypes = self._draw_categorical(np.broadcast_to(self.subtype_prevalence, (episodes, self.latent_subtypes)), streams["subtype_u"])
        supported = np.asarray(self.contract.supported_actions, dtype=int)
        marginal = np.asarray(self.contract.target_action_frequency)[supported]; marginal = marginal / marginal.sum()
        local_prior = self._draw_categorical(np.broadcast_to(marginal, (episodes, len(supported))), streams["prior_action_u"])
        pending = supported[local_prior].astype(np.int16)
        previous_mask = np.ones((episodes, self.contract.feature_dim), dtype=bool)
        recency = np.zeros((episodes, self.contract.feature_dim), dtype=np.float32)
        belief = np.full(episodes, 2.0); alive = np.ones(episodes, dtype=bool); returns = np.zeros(episodes)
        lengths = np.zeros(episodes, dtype=np.int16); actions_log: list[np.ndarray] = []; previous_log: list[np.ndarray] = []
        probability_log: list[np.ndarray] = []; mask_log: list[np.ndarray] = []; terminal_events = np.zeros(episodes, dtype=bool)
        dense_values: dict[str, list[np.ndarray]] = {target.name: [] for target in self.contract.dense_targets}
        dense_available: dict[str, int] = {target.name: 0 for target in self.contract.dense_targets}
        dense_sign_offset: dict[str, int] = {target.name: 0 for target in self.contract.dense_targets}
        reference = ObservationHistoryPolicy(self.contract.feature_dim, self.contract.action_count, self.contract.supported_actions)
        for time in range(self.horizon):
            observation, mask, recency = self.emission(states, subtypes, previous_mask, recency, pending,
                                                       streams["noise"][:, time], streams["mask_u"][:, time])
            previous_mask = mask; mask_log.append(mask[alive])
            probability = np.zeros((episodes, self.contract.action_count), dtype=np.float64)
            if policy == "ehr_matched":
                bins = self.behavior.context_bin(observation, mask, recency, pending, time)
                for index in np.flatnonzero(alive): probability[index] = self.behavior.distribution(int(pending[index]), int(bins[index]), self.contract.action_count)
                action = self._draw_categorical(probability, streams["policy_u"][:, time]).astype(np.int16)
            elif policy in {"history", "observation_history_severity"}:
                belief = reference.update_belief(belief, observation, mask, recency, pending)
                action = reference.actions(belief)
                probability[np.arange(episodes), action] = 1.0
            elif policy == "minimum":
                action = np.full(episodes, supported[0], dtype=np.int16); probability[:, supported[0]] = 1.0
            elif policy == "maximum":
                action = np.full(episodes, supported[-1], dtype=np.int16); probability[:, supported[-1]] = 1.0
            elif policy == "random":
                local = np.minimum((streams["policy_u"][:, time] * len(supported)).astype(int), len(supported) - 1)
                action = supported[local].astype(np.int16); probability[:, supported] = 1.0 / len(supported)
            else: raise ValueError(policy)
            action[~alive] = supported[0]
            previous_log.append(pending[alive].copy()); actions_log.append(action[alive].copy()); probability_log.append(probability[alive].copy())
            base = np.asarray([self.base_dense_signal(int(s), int(z), int(p), int(a)) for s, z, p, a in zip(states, subtypes, pending, action, strict=True)])
            primary_dense = np.zeros(episodes)
            if self.frozen is not None:
                for component_index, target in enumerate(self.contract.dense_targets):
                    available = alive & (streams["dense_u"][:, time, component_index] < target.availability_or_nonzero_fraction)
                    available_index = np.flatnonzero(available)
                    signs = np.where(
                        (np.arange(len(available_index)) + dense_sign_offset[target.name]) % 2 == 0,
                        1.0, -1.0,
                    )
                    dense_sign_offset[target.name] += len(available_index)
                    values = self.frozen.scale(target.name) * (
                        signs + 0.1 * np.tanh(base[available_index] - self.frozen.center(target.name))
                    )
                    dense_values[target.name].append(values); dense_available[target.name] += int(available.sum())
                    if target.primary: primary_dense[available_index] = values
            next_states = np.empty_like(states)
            for index in range(episodes):
                transition = self.transition_probability(int(states[index]), int(subtypes[index]), int(pending[index]))
                next_states[index] = int(np.sum(streams["transition_u"][index, time] > np.cumsum(transition)))
            hazard = 1.0 if time == self.horizon - 1 else self.contract.termination_hazards[time]
            terminate = alive & (streams["termination_u"][:, time] < hazard)
            terminal_reward = np.zeros(episodes)
            if self.contract.primary_reward_type == "terminal":
                death_probability = np.asarray([self.death_probability(int(state)) for state in next_states])
                death = streams["outcome_u"][:, time] < death_probability
                terminal_events |= terminate & death
                terminal_reward[terminate] = np.where(death[terminate], self.contract.terminal_reward_minimum, self.contract.terminal_reward_maximum)
            synthetic_cost = 0.05 * base
            returns += (self.discount ** time) * np.where(alive, primary_dense + terminal_reward + synthetic_cost, 0.0)
            lengths[alive] = time + 1; alive &= ~terminate; states = next_states; pending = action
        actions_all = np.concatenate(actions_log); previous_all = np.concatenate(previous_log); probabilities = np.concatenate(probability_log)
        masks = np.concatenate(mask_log)
        dense_summary = {}
        total_decisions = len(actions_all)
        for target in self.contract.dense_targets:
            values = np.concatenate(dense_values[target.name]) if dense_values[target.name] else np.asarray([], dtype=float)
            dense_summary[target.name] = {
                "variance": float(np.var(values)) if len(values) else math.nan,
                "available_or_nonzero_fraction": dense_available[target.name] / max(total_decisions, 1),
                "available_count": dense_available[target.name], "decision_denominator": total_decisions,
            }
        return {
            "returns": returns, "mean_return": float(np.mean(returns)), "return_se": float(np.std(returns, ddof=1) / math.sqrt(episodes)),
            "mean_horizon": float(np.mean(lengths)), "early_termination": float(np.mean(lengths < self.horizon)),
            "missingness": float(np.mean(~masks)), "actions": actions_all, "previous_actions": previous_all,
            "probabilities": probabilities, "terminal_event_prevalence": float(np.mean(terminal_events)) if self.contract.terminal_event_prevalence is not None else math.nan,
            "dense": dense_summary, "distinct_actions": int(np.unique(actions_all).size),
        }


def fit_profile_parameters(contract: ProfileContract, environments: list[RepairedPOMDP], episodes: int, seed_base: int) -> FrozenProfileParameters:
    terminal_intercept: float | None = None
    if contract.terminal_event_prevalence is not None:
        # Development-only state occupancy approximation. The fixed intercept is
        # fitted before fresh final seeds are instantiated.
        state_weight = np.asarray([0.12, 0.20, 0.36, 0.22, 0.10])
        slope = float(environments[0].generator["terminal_state_logit_slope"])
        low, high = -12.0, 12.0
        for _ in range(100):
            mid = (low + high) / 2
            probability = 1.0 / (1.0 + np.exp(-(mid + slope * (np.arange(5) - 2.0))))
            if float(state_weight @ probability) < contract.terminal_event_prevalence: low = mid
            else: high = mid
        terminal_intercept = (low + high) / 2
    centers: list[tuple[str, float]] = []; scales: list[tuple[str, float]] = []
    if contract.dense_targets:
        raw: list[float] = []
        for environment_index, environment in enumerate(environments):
            rng = np.random.default_rng(seed_base + environment_index)
            for _ in range(max(1024, episodes // len(environments))):
                state = int(rng.choice(5, p=environment.initial_state_probability)); subtype = int(rng.choice(3, p=environment.subtype_prevalence))
                pending = int(rng.choice(contract.supported_actions)); action = int(rng.choice(contract.supported_actions))
                raw.append(environment.base_dense_signal(state, subtype, pending, action))
        raw_array = np.asarray(raw); center = float(np.median(raw_array))
        for target in contract.dense_targets:
            # The stochastic component is antithetic and unit variance; the
            # bounded 0.1 action-response term preserves a known conditional
            # response without making rare-component variance unstable.
            centers.append((target.name, center)); scales.append((target.name, math.sqrt(target.variance / 1.01)))
    def candidate(intercept: float | None, missingness_offset: float) -> FrozenProfileParameters:
        payload = {"terminal_logit_intercept": intercept, "missingness_offset": missingness_offset,
                   "dense_scales": scales, "dense_centers": centers}
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return FrozenProfileParameters(intercept, missingness_offset, tuple(scales), tuple(centers), fingerprint)

    if terminal_intercept is not None:
        low, high = terminal_intercept - 3.0, terminal_intercept + 3.0
        for iteration in range(18):
            mid = (low + high) / 2
            current = candidate(mid, 0.0)
            observed = []
            for environment_index, environment in enumerate(environments):
                environment.frozen = current
                observed.append(environment.simulate(min(256, episodes), seed_base + 20_000 + environment_index, "ehr_matched")["terminal_event_prevalence"])
            if float(np.mean(observed)) < contract.terminal_event_prevalence: low = mid
            else: high = mid
        terminal_intercept = (low + high) / 2
    low, high = -0.5, 0.5
    for iteration in range(18):
        mid = (low + high) / 2
        current = candidate(terminal_intercept, mid)
        observed = []
        for environment_index, environment in enumerate(environments):
            environment.frozen = current
            observed.append(environment.simulate(min(256, episodes), seed_base + 30_000 + environment_index, "ehr_matched")["missingness"])
        if float(np.mean(observed)) > contract.target_missingness: low = mid
        else: high = mid
    missingness_offset = (low + high) / 2
    payload = {"terminal_logit_intercept": terminal_intercept, "missingness_offset": missingness_offset,
               "dense_scales": scales, "dense_centers": centers}
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    frozen = FrozenProfileParameters(terminal_intercept, missingness_offset, tuple(scales), tuple(centers), fingerprint)
    for environment in environments: environment.frozen = frozen
    return frozen
