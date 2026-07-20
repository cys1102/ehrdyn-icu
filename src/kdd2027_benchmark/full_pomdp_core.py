from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .full_pomdp_types import BehaviorCalibration, ProfileContract, action_coordinates


def _softmax_rows(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    output = np.exp(shifted)
    return output / output.sum(axis=-1, keepdims=True)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


@dataclass(frozen=True, slots=True)
class HistoryComparator:
    profile: str
    action_count: int
    supported_actions: tuple[int, ...]
    belief_smoothing: float
    state_offset: float
    subtype_assumption: int
    observation_loading: float
    subtype_observation_shift: float
    fingerprint: str

    latent_state_access = False
    latent_subtype_access = False
    future_observation_access = False
    future_mask_access = False
    simulator_parameter_access = False

    @classmethod
    def create(cls, contract: ProfileContract, smoothing: float, offset: float,
               subtype: int, observation_loading: float, subtype_observation_shift: float) -> "HistoryComparator":
        payload = {
            "profile": contract.profile,
            "action_count": contract.action_count,
            "supported_actions": contract.supported_actions,
            "belief_smoothing": float(smoothing),
            "state_offset": float(offset),
            "subtype_assumption": int(subtype),
            "observation_loading": float(observation_loading),
            "subtype_observation_shift": float(subtype_observation_shift),
            "inference_inputs": ["observations", "masks", "recency", "prior_actions"],
        }
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return cls(contract.profile, contract.action_count, contract.supported_actions, float(smoothing),
                   float(offset), int(subtype), float(observation_loading), float(subtype_observation_shift), fingerprint)

    def update(self, belief: np.ndarray, subtype_belief: np.ndarray, observation: np.ndarray,
               mask: np.ndarray, recency: np.ndarray, prior_action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        signs = np.where(np.arange(observation.shape[1]) % 2 == 0, 1.0, -1.0)
        count = np.maximum(mask.sum(axis=1), 1)
        signed = np.sum(observation * signs[None, :] * mask, axis=1) / count
        current = np.clip((signed / max(self.observation_loading, 1e-8) + 1.0) * 2.0 + self.state_offset, 0.0, 4.0)
        current -= np.clip(np.mean(recency, axis=1), 0.0, 10.0) * 0.01
        current += (prior_action / max(self.action_count - 1, 1) - 0.5) * 0.03
        raw_mean = np.sum(observation * mask, axis=1) / count
        current_subtype = np.clip(raw_mean / max(self.subtype_observation_shift, 1e-8) + 1.0, 0.0, 2.0)
        return (
            self.belief_smoothing * belief + (1.0 - self.belief_smoothing) * current,
            self.belief_smoothing * subtype_belief + (1.0 - self.belief_smoothing) * current_subtype,
        )

    def actions(self, belief: np.ndarray, subtype_belief: np.ndarray) -> np.ndarray:
        severity = np.clip(np.rint(belief), 0, 4).astype(int)
        inferred_subtype = np.clip(np.rint(subtype_belief), 0, 2).astype(int)
        if self.action_count == 25:
            first = np.clip(severity + inferred_subtype - 1, 0, 4)
            second = np.clip(severity - inferred_subtype + 1, 0, 4)
            proposed = first * 5 + second
        elif self.action_count == 4:
            first = severity >= np.where(inferred_subtype == 0, 3, 2)
            second = severity >= np.where(inferred_subtype == 2, 3, 2)
            proposed = first.astype(int) * 2 + second.astype(int)
        else:
            # Binary HF actions use a development-selected conservative
            # threshold; the noisy subtype belief is not allowed to trigger
            # low-severity treatment.
            threshold = (3, 2, 1)[self.subtype_assumption]
            proposed = (severity >= threshold).astype(int)
        supported = np.asarray(self.supported_actions, dtype=int)
        return np.asarray([supported[np.argmin(np.abs(supported - value))] for value in proposed], dtype=np.int16)


@dataclass(frozen=True, slots=True)
class EnvironmentConstruction:
    response_strength: float
    terminal_logit_intercept: float | None
    mask_offset: float
    construction_iterations: int
    exact_adaptivity_gap: float
    fingerprint: str


class R2Environment:
    """Known-construction finite POMDP with explicit reward accounting."""

    def __init__(self, contract: ProfileContract, behavior: BehaviorCalibration, seed: int,
                 generator: dict[str, Any], construction: EnvironmentConstruction | None = None):
        self.contract = contract
        self.behavior = behavior
        self.seed = int(seed)
        self.generator = dict(generator)
        self.states = int(generator["latent_states"])
        self.subtypes = int(generator["latent_subtypes"])
        self.horizon = int(generator["horizon"])
        self.discount = float(generator["discount"])
        self.coordinates = action_coordinates(contract.action_count)
        self.supported = np.asarray(contract.supported_actions, dtype=int)
        rng = np.random.default_rng(seed)
        self.subtype_prevalence = rng.dirichlet(np.asarray([5.0, 7.0, 5.0]))
        self.initial_state_probability = np.asarray([0.12, 0.20, 0.36, 0.22, 0.10], dtype=float)
        self.observation_noise = float(generator["observation_noise"])
        self.observation_loading = float(generator["observation_loading"])
        self.feature_offsets = np.random.default_rng(seed + 700_000).uniform(-0.025, 0.025, contract.feature_dim)
        self.construction = construction or EnvironmentConstruction(
            float(generator["response_strength_initial"]),
            math.log(contract.terminal_event_prevalence / (1.0 - contract.terminal_event_prevalence))
            if contract.terminal_event_prevalence is not None else None,
            float(generator["mask_calibration_offset"]), 0, math.nan, "unfrozen",
        )
        self._precompute()
        self.mechanism_hash = self._hash()

    def _hash(self) -> str:
        payload = {
            "contract": asdict(self.contract), "behavior": asdict(self.behavior), "seed": self.seed,
            "generator": self.generator, "construction": asdict(self.construction),
            "subtype_prevalence": self.subtype_prevalence.tolist(), "feature_offsets": self.feature_offsets.tolist(),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def ideal_action(self, state: int, subtype: int) -> int:
        if self.contract.action_count == 25:
            proposed = int(np.clip(state + subtype - 1, 0, 4)) * 5 + int(np.clip(state - subtype + 1, 0, 4))
        elif self.contract.action_count == 4:
            proposed = int(state >= (3 if subtype == 0 else 2)) * 2 + int(state >= (3 if subtype == 2 else 2))
        else:
            proposed = int(state >= (3, 2, 1)[subtype])
        return int(self.supported[np.argmin(np.abs(self.supported - proposed))])

    def _mismatch(self, state: int, subtype: int, action: int) -> float:
        target = self.coordinates[self.ideal_action(state, subtype)]
        return float(np.mean(np.square(self.coordinates[action] - target)))

    def _precompute(self) -> None:
        shape = (self.states, self.subtypes, self.contract.action_count, self.contract.action_count, self.states)
        self.transition = np.zeros(shape, dtype=np.float64)
        self.null_transition = np.zeros(shape, dtype=np.float64)
        self.reward_components: dict[str, np.ndarray] = {
            name: np.zeros(shape[:-1], dtype=np.float64)
            for name in ("dense_physiology", "treatment_cost", "toxicity", "switching_cost")
        }
        g = self.generator
        for state in range(self.states):
            for subtype in range(self.subtypes):
                for pending in self.supported:
                    for action in self.supported:
                        delayed = 0.5 - self._mismatch(state, subtype, int(pending))
                        current = 0.5 - self._mismatch(state, subtype, int(action))
                        score = self.construction.response_strength * (
                            float(g["delayed_response_weight"]) * delayed + float(g["current_response_weight"]) * current
                        )
                        intensity = float(np.mean(self.coordinates[action]))
                        logits = np.asarray([-0.45 + score, 0.65, -0.45 - score])
                        local = _softmax_rows(logits[None, :])[0]
                        null_local = _softmax_rows(np.asarray([[-0.45, 0.65, -0.45]]))[0]
                        for delta, probability, null_probability in zip((-1, 0, 1), local, null_local, strict=True):
                            destination = int(np.clip(state + delta, 0, self.states - 1))
                            self.transition[state, subtype, pending, action, destination] += probability
                            self.null_transition[state, subtype, pending, action, destination] += null_probability
                        dense = 0.0
                        if self.contract.primary_reward_type == "dense":
                            target = next(item for item in self.contract.dense_targets if item.primary)
                            scale = math.sqrt(target.variance / 1.01)
                            dense = target.availability_or_nonzero_fraction * float(g["dense_response_fraction"]) * scale * math.tanh(score)
                        distance = float(np.mean(np.abs(self.coordinates[action] - self.coordinates[pending])))
                        self.reward_components["dense_physiology"][state, subtype, pending, action] = dense
                        self.reward_components["treatment_cost"][state, subtype, pending, action] = -float(g["treatment_cost_scale"]) * intensity * intensity
                        self.reward_components["toxicity"][state, subtype, pending, action] = -float(g["toxicity_scale"]) * (1.0 - state / 4.0) * intensity * intensity
                        self.reward_components["switching_cost"][state, subtype, pending, action] = -float(g["switching_cost_scale"]) * distance

    def death_probability(self, state: np.ndarray | int) -> np.ndarray:
        if self.contract.terminal_event_prevalence is None or self.construction.terminal_logit_intercept is None:
            return np.zeros_like(np.asarray(state), dtype=float)
        logit = self.construction.terminal_logit_intercept + 0.7 * (np.asarray(state, dtype=float) - 2.0)
        return 1.0 / (1.0 + np.exp(-logit))

    def immediate_reward(self, state: int, subtype: int, pending: int, action: int,
                         null_response: bool = False) -> float:
        if null_response:
            return 0.0
        return float(sum(component[state, subtype, pending, action] for component in self.reward_components.values()))

    def exact_values(self, policy: str, null_response: bool = False) -> tuple[float, np.ndarray]:
        actions = self.contract.action_count
        transition_table = self.null_transition if null_response else self.transition
        value = np.zeros((self.horizon + 1, self.states, self.subtypes, actions), dtype=float)
        selected = np.full((self.horizon, self.states, self.subtypes, actions), -1, dtype=np.int16)
        for time in range(self.horizon - 1, -1, -1):
            hazard = 1.0 if time == self.horizon - 1 else self.contract.termination_hazards[time]
            for state in range(self.states):
                for subtype in range(self.subtypes):
                    for pending in self.supported:
                        q = np.full(actions, -np.inf)
                        for action in self.supported:
                            transition = transition_table[state, subtype, pending, action]
                            terminal = 0.0
                            if self.contract.primary_reward_type == "terminal" and not null_response:
                                terminal = float(np.sum(transition * (1.0 - 2.0 * self.death_probability(np.arange(self.states)))))
                            elif self.contract.primary_reward_type == "terminal" and null_response:
                                terminal = float(np.sum(transition * (1.0 - 2.0 * self.death_probability(np.arange(self.states)))))
                            future = float(np.sum(transition * value[time + 1, :, subtype, action]))
                            q[action] = self.immediate_reward(state, subtype, int(pending), int(action), null_response) + hazard * terminal + self.discount * (1.0 - hazard) * future
                        if policy == "oracle": chosen = int(np.argmax(q))
                        elif policy == "minimum": chosen = int(self.supported[0])
                        elif policy == "maximum": chosen = int(self.supported[-1])
                        elif policy == "severity": chosen = self.ideal_action(state, 1)
                        elif policy == "random":
                            value[time, state, subtype, pending] = float(np.mean(q[self.supported])); continue
                        else: raise ValueError(policy)
                        selected[time, state, subtype, pending] = chosen
                        value[time, state, subtype, pending] = q[chosen]
        marginal = np.asarray(self.contract.target_action_frequency)[self.supported]
        marginal = marginal / marginal.sum()
        initial = 0.0
        for local, pending in enumerate(self.supported):
            initial += marginal[local] * float(np.sum(value[0, :, :, pending] * self.initial_state_probability[:, None] * self.subtype_prevalence[None, :]))
        return initial, selected

    def observation_overlap(self) -> float:
        adjacent_difference = 2.0 * self.observation_loading / (self.states - 1)
        return float(2.0 * _normal_cdf(-adjacent_difference / (2.0 * self.observation_noise)))

    def _streams(self, episodes: int, stream_seed: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(stream_seed)
        return {
            "state_u": rng.random(episodes), "subtype_u": rng.random(episodes), "prior_u": rng.random(episodes),
            "transition_u": rng.random((episodes, self.horizon)), "termination_u": rng.random((episodes, self.horizon)),
            "outcome_u": rng.random((episodes, self.horizon)), "policy_u": rng.random((episodes, self.horizon)),
            "noise": rng.normal(size=(episodes, self.horizon, self.contract.feature_dim)),
            "mask_u": rng.random((episodes, self.horizon, self.contract.feature_dim)),
            "dense_u": rng.random((episodes, self.horizon, max(1, len(self.contract.dense_targets)))),
        }

    @staticmethod
    def _draw(probabilities: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
        return np.sum(uniforms[:, None] > np.cumsum(probabilities, axis=1), axis=1).astype(np.int16)

    def _emit(self, states: np.ndarray, subtypes: np.ndarray, previous_mask: np.ndarray,
              recency: np.ndarray, prior_action: np.ndarray, noise: np.ndarray,
              mask_u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        normalized = states[:, None] / 4.0
        signs = np.where(np.arange(self.contract.feature_dim) % 2 == 0, 1.0, -1.0)
        mean = self.observation_loading * signs[None, :] * (2.0 * normalized - 1.0)
        mean += float(self.generator["subtype_observation_shift"]) * (subtypes[:, None] - 1.0)
        observation = mean + self.observation_noise * noise
        intensity = prior_action[:, None] / max(self.contract.action_count - 1, 1)
        probability = (
            1.0 - self.contract.target_missingness + self.construction.mask_offset + self.feature_offsets[None, :]
            + float(self.generator["mask_severity_effect"]) * (normalized - 0.5)
            + float(self.generator["mask_prior_observed_effect"]) * (previous_mask.astype(float) - 0.5)
            - float(self.generator["mask_recency_effect"]) * np.minimum(recency, 8.0)
            + float(self.generator["mask_past_action_effect"]) * (intensity - 0.5)
        )
        mask = mask_u < np.clip(probability, 0.02, 0.99)
        observation = np.where(mask, observation, 0.0)
        return observation.astype(np.float32), mask, np.where(mask, 0.0, recency + 1.0).astype(np.float32)

    def simulate(self, episodes: int, stream_seed: int, policy: str,
                 comparator: HistoryComparator | None = None, null_response: bool = False,
                 exact_actions: np.ndarray | None = None, collect_probabilities: bool = False) -> dict[str, Any]:
        streams = self._streams(episodes, stream_seed)
        states = self._draw(np.broadcast_to(self.initial_state_probability, (episodes, self.states)), streams["state_u"])
        subtypes = self._draw(np.broadcast_to(self.subtype_prevalence, (episodes, self.subtypes)), streams["subtype_u"])
        marginal = np.asarray(self.contract.target_action_frequency)[self.supported]; marginal /= marginal.sum()
        pending = self.supported[self._draw(np.broadcast_to(marginal, (episodes, len(self.supported))), streams["prior_u"])]
        previous_mask = np.ones((episodes, self.contract.feature_dim), dtype=bool)
        recency = np.zeros((episodes, self.contract.feature_dim), dtype=np.float32)
        belief = np.full(episodes, 2.0); subtype_belief = np.full(episodes, comparator.subtype_assumption if comparator is not None else 1.0); alive = np.ones(episodes, dtype=bool)
        returns = np.zeros(episodes); lengths = np.zeros(episodes, dtype=np.int16)
        actions_log: list[np.ndarray] = []; previous_log: list[np.ndarray] = []; probability_log: list[np.ndarray] = []
        mask_log: list[np.ndarray] = []; state_log: list[np.ndarray] = []; terminal_events = np.zeros(episodes, dtype=bool)
        component_sums = {name: 0.0 for name in (*self.reward_components.keys(), "terminal")}
        terminal_emissions = np.zeros(episodes, dtype=np.int16)
        dense_values = {target.name: [] for target in self.contract.dense_targets}; dense_count = {target.name: 0 for target in self.contract.dense_targets}; dense_offset = {target.name: 0 for target in self.contract.dense_targets}
        for time in range(self.horizon):
            observation, mask, recency = self._emit(states, subtypes, previous_mask, recency, pending, streams["noise"][:, time], streams["mask_u"][:, time])
            previous_mask = mask
            probability = np.zeros((episodes, self.contract.action_count), dtype=float)
            if policy == "ehr_matched":
                bins = self.behavior.context_bin(observation, mask, recency, pending, time)
                for index in np.flatnonzero(alive): probability[index] = self.behavior.distribution(int(pending[index]), int(bins[index]), self.contract.action_count)
                action = self._draw(probability, streams["policy_u"][:, time])
            elif policy == "smart_like_exploratory":
                probability[:, self.supported] = 1.0 / len(self.supported); action = self._draw(probability, streams["policy_u"][:, time])
            elif policy == "concentrated_behavior":
                probability[:, self.supported] = (1.0 - float(self.generator["concentrated_previous_action_mass"])) * marginal
                probability[np.arange(episodes), pending] += float(self.generator["concentrated_previous_action_mass"])
                action = self._draw(probability, streams["policy_u"][:, time])
            elif policy == "history":
                if comparator is None: raise ValueError("history comparator required")
                belief, subtype_belief = comparator.update(belief, subtype_belief, observation, mask, recency, pending)
                action = comparator.actions(belief, subtype_belief)
                probability[np.arange(episodes), action] = 1.0
            elif policy == "minimum":
                action = np.full(episodes, self.supported[0], dtype=np.int16); probability[:, self.supported[0]] = 1.0
            elif policy == "maximum":
                action = np.full(episodes, self.supported[-1], dtype=np.int16); probability[:, self.supported[-1]] = 1.0
            elif policy == "random":
                probability[:, self.supported] = 1.0 / len(self.supported); action = self._draw(probability, streams["policy_u"][:, time])
            elif policy == "severity":
                action = np.asarray([self.ideal_action(int(state), 1) for state in states], dtype=np.int16); probability[np.arange(episodes), action] = 1.0
            elif policy == "oracle":
                if exact_actions is None: raise ValueError("oracle action table required")
                action = exact_actions[time, states, subtypes, pending]; probability[np.arange(episodes), action] = 1.0
            else: raise ValueError(policy)
            action[~alive] = self.supported[0]
            active = np.flatnonzero(alive)
            actions_log.append(action[active].copy()); previous_log.append(pending[active].copy()); mask_log.append(mask[active].copy()); state_log.append(states[active].copy())
            if collect_probabilities: probability_log.append(probability[active].copy())
            transition_table = self.null_transition if null_response else self.transition
            transition = transition_table[states, subtypes, pending, action]
            next_states = self._draw(transition, streams["transition_u"][:, time])
            immediate = np.zeros(episodes)
            if not null_response:
                for name, table in self.reward_components.items():
                    values = table[states, subtypes, pending, action]
                    immediate += values; component_sums[name] += float(values[alive].sum())
            if not null_response:
                for component_index, target in enumerate(self.contract.dense_targets):
                    if target.primary:
                        expected_all = self.reward_components["dense_physiology"][states, subtypes, pending, action]
                        immediate[alive] -= expected_all[alive]
                        component_sums["dense_physiology"] -= float(expected_all[alive].sum())
                    available = alive & (streams["dense_u"][:, time, component_index] < target.availability_or_nonzero_fraction)
                    indices = np.flatnonzero(available)
                    signs = np.where((np.arange(len(indices)) + dense_offset[target.name]) % 2 == 0, 1.0, -1.0)
                    dense_offset[target.name] += len(indices)
                    score = self.construction.response_strength * np.asarray([
                        0.5 - self._mismatch(int(states[i]), int(subtypes[i]), int(action[i])) for i in indices
                    ])
                    scale = math.sqrt(target.variance / 1.01)
                    values = scale * (signs + float(self.generator["dense_response_fraction"]) * np.tanh(score))
                    dense_values[target.name].append(values); dense_count[target.name] += len(indices)
                    if target.primary:
                        immediate[indices] += values
                        component_sums["dense_physiology"] += float(values.sum())
            hazard = 1.0 if time == self.horizon - 1 else self.contract.termination_hazards[time]
            terminate = alive & (streams["termination_u"][:, time] < hazard)
            terminal = np.zeros(episodes)
            if self.contract.primary_reward_type == "terminal" and not null_response:
                death = streams["outcome_u"][:, time] < self.death_probability(next_states)
                terminal_events |= terminate & death
                terminal[terminate] = np.where(death[terminate], self.contract.terminal_reward_minimum, self.contract.terminal_reward_maximum)
                terminal_emissions[terminate] += 1; component_sums["terminal"] += float(terminal.sum())
            returns += (self.discount ** time) * np.where(alive, immediate + terminal, 0.0)
            lengths[alive] = time + 1; alive &= ~terminate; states = next_states; pending = action
        actions = np.concatenate(actions_log); previous = np.concatenate(previous_log); masks = np.concatenate(mask_log); latent = np.concatenate(state_log)
        dense_summary = {}
        for target in self.contract.dense_targets:
            values = np.concatenate(dense_values[target.name]) if dense_values[target.name] else np.asarray([])
            dense_summary[target.name] = {
                "variance": float(np.var(values)) if len(values) > 1 else math.nan,
                "available_or_nonzero_fraction": dense_count[target.name] / max(len(actions), 1),
                "available_count": dense_count[target.name], "decision_denominator": len(actions),
            }
        result = {
            "returns": returns, "mean_return": float(np.mean(returns)), "return_se": float(np.std(returns, ddof=1) / math.sqrt(episodes)),
            "actions": actions, "previous_actions": previous, "masks": masks, "latent_states": latent,
            "mean_horizon": float(np.mean(lengths)), "early_termination": float(np.mean(lengths < self.horizon)),
            "missingness": float(np.mean(~masks)), "terminal_event_prevalence": float(np.mean(terminal_events)) if self.contract.terminal_event_prevalence is not None else math.nan,
            "terminal_emission_max": int(terminal_emissions.max()), "terminal_emission_total": int(terminal_emissions.sum()),
            "dense": dense_summary, "distinct_actions": int(np.unique(actions).size), "component_sums": component_sums,
        }
        if collect_probabilities: result["probabilities"] = np.concatenate(probability_log)
        return result

    def partial_observation_audit(self, episodes: int, seed: int) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        states = rng.integers(0, self.states, size=episodes); subtypes = rng.integers(0, self.subtypes, size=episodes)
        mask = np.ones((episodes, self.contract.feature_dim), dtype=bool); recency = np.zeros_like(mask, dtype=float); action = np.zeros(episodes, dtype=int)
        observations = []
        for time in range(3):
            obs, local_mask, recency = self._emit(states, subtypes, mask, recency, action, rng.normal(size=(episodes, self.contract.feature_dim)), rng.random((episodes, self.contract.feature_dim)))
            mask = local_mask; observations.append((obs, local_mask))
        signs = np.where(np.arange(self.contract.feature_dim) % 2 == 0, 1.0, -1.0)
        def decode(items: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
            numerator = sum(np.sum(obs * signs[None, :] * local_mask, axis=1) for obs, local_mask in items)
            denominator = sum(np.maximum(local_mask.sum(axis=1), 1) for _, local_mask in items)
            score = numerator / np.maximum(denominator, 1)
            return np.clip(np.rint((score / self.observation_loading + 1.0) * 2.0), 0, 4).astype(int)
        single = float(np.mean(decode(observations[:1]) == states)); history = float(np.mean(decode(observations) == states))
        first_mask = observations[0][1]
        mild = float(np.mean(first_mask[states <= 1])); severe = float(np.mean(first_mask[states >= 3]))
        return {
            "observation_overlap": self.observation_overlap(), "single_observation_decode_accuracy": single,
            "three_observation_decode_accuracy": history, "history_decode_improvement": history - single,
            "mask_informativeness": abs(severe - mild), "mild_observation_rate": mild, "severe_observation_rate": severe,
        }


def adaptivity_gap(environment: R2Environment) -> tuple[float, dict[str, float], np.ndarray]:
    values: dict[str, float] = {}
    oracle, selected = environment.exact_values("oracle")
    for name in ("minimum", "maximum", "random", "severity"):
        values[name] = environment.exact_values(name)[0]
    values["oracle"] = oracle
    return oracle - max(values[name] for name in ("minimum", "maximum", "random", "severity")), values, selected


def _terminal_root(contract: ProfileContract, terminal_states: np.ndarray, iterations: int) -> float | None:
    if contract.terminal_event_prevalence is None: return None
    low, high = -12.0, 12.0
    target = float(contract.terminal_event_prevalence)
    for _ in range(iterations):
        middle = (low + high) / 2.0
        observed = float(np.mean(1.0 / (1.0 + np.exp(-(middle + 0.7 * (terminal_states - 2.0))))))
        if observed < target: low = middle
        else: high = middle
    return (low + high) / 2.0


def construction_terminal_states(environment: R2Environment, episodes: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    states = environment._draw(np.broadcast_to(environment.initial_state_probability, (episodes, environment.states)), rng.random(episodes))
    subtypes = environment._draw(np.broadcast_to(environment.subtype_prevalence, (episodes, environment.subtypes)), rng.random(episodes))
    marginal = np.asarray(environment.contract.target_action_frequency)[environment.supported]; marginal /= marginal.sum()
    pending = environment.supported[environment._draw(np.broadcast_to(marginal, (episodes, len(environment.supported))), rng.random(episodes))]
    alive = np.ones(episodes, dtype=bool); terminal_states = np.zeros(episodes, dtype=np.int16)
    for time in range(environment.horizon):
        action = environment.supported[environment._draw(np.broadcast_to(marginal, (episodes, len(environment.supported))), rng.random(episodes))]
        transition = environment.transition[states, subtypes, pending, action]
        next_states = environment._draw(transition, rng.random(episodes))
        hazard = 1.0 if time == environment.horizon - 1 else environment.contract.termination_hazards[time]
        terminate = alive & (rng.random(episodes) < hazard)
        terminal_states[terminate] = next_states[terminate]
        alive &= ~terminate; states = next_states; pending = action
    return terminal_states


def construct_environment(contract: ProfileContract, behavior: BehaviorCalibration, seed: int,
                          generator: dict[str, Any], tolerances: dict[str, float],
                          quadrature_iterations: int, construction_episodes: int = 65536) -> R2Environment:
    strength = float(generator["response_strength_initial"]); iteration = 0
    selected_environment: R2Environment | None = None; selected_gap = -math.inf
    while strength <= float(generator["response_strength_maximum"]) + 1e-12:
        mask_offset = float(generator["mask_calibration_offset"])
        provisional = EnvironmentConstruction(strength, None, mask_offset, iteration, math.nan, "provisional")
        environment = R2Environment(contract, behavior, seed, generator, provisional)
        terminal_states = construction_terminal_states(environment, construction_episodes, seed + 71_000_000)
        intercept = _terminal_root(contract, terminal_states, quadrature_iterations)
        payload = {"seed": seed, "response_strength": strength, "terminal_intercept": intercept, "mask_offset": mask_offset, "iteration": iteration}
        frozen = EnvironmentConstruction(strength, intercept, mask_offset, iteration, math.nan, hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest())
        environment = R2Environment(contract, behavior, seed, generator, frozen)
        gap, _, _ = adaptivity_gap(environment)
        selected_environment = environment; selected_gap = gap
        if gap > float(tolerances["oracle_adaptivity_margin"]): break
        strength += float(generator["response_strength_step"]); iteration += 1
    assert selected_environment is not None
    final_payload = {"seed": seed, "response_strength": selected_environment.construction.response_strength,
                     "terminal_intercept": selected_environment.construction.terminal_logit_intercept,
                     "mask_offset": selected_environment.construction.mask_offset, "iteration": iteration,
                     "exact_adaptivity_gap": selected_gap}
    construction = EnvironmentConstruction(
        selected_environment.construction.response_strength, selected_environment.construction.terminal_logit_intercept,
        selected_environment.construction.mask_offset, iteration, selected_gap,
        hashlib.sha256(json.dumps(final_payload, sort_keys=True).encode()).hexdigest(),
    )
    return R2Environment(contract, behavior, seed, generator, construction)


def seed_namespaces_valid(config: dict[str, Any]) -> bool:
    groups = [set(config[name]) for name in (
        "smoke_seeds", "development_seeds", "retired_kdd165_seeds",
        "retired_kdd165r_development_seeds", "retired_kdd165r_final_seeds", "fresh_final_seeds",
    )]
    return len(config["fresh_final_seeds"]) == 8 and all(not (left & right) for index, left in enumerate(groups) for right in groups[index + 1:])


def overlap_gate(environment: R2Environment, tolerances: dict[str, float], audit: dict[str, float]) -> bool:
    return (
        audit["observation_overlap"] >= float(tolerances["observation_overlap_minimum"])
        and audit["single_observation_decode_accuracy"] <= float(tolerances["single_observation_decode_maximum"])
        and audit["single_observation_decode_accuracy"] >= float(tolerances["single_observation_decode_minimum"])
        and audit["history_decode_improvement"] >= float(tolerances["history_decode_improvement_minimum"])
        and audit["mask_informativeness"] >= float(tolerances["mask_informativeness_minimum"])
    )


def fit_history_comparator(contract: ProfileContract, environments: list[R2Environment],
                           grid: dict[str, list[float | int]], episodes: int, seed: int,
                           confidence_z: float) -> tuple[HistoryComparator, list[dict[str, Any]]]:
    controls: dict[tuple[int, str], np.ndarray] = {}
    for environment in environments:
        for name in ("minimum", "maximum", "random"):
            controls[(environment.seed, name)] = environment.simulate(episodes, seed + environment.seed, name)["returns"]
    rows: list[dict[str, Any]] = []; best: HistoryComparator | None = None; best_score = -math.inf
    for smoothing in grid["belief_smoothing"]:
        for offset in grid["state_offset"]:
            for subtype in grid["subtype_assumption"]:
                comparator = HistoryComparator.create(contract, float(smoothing), float(offset), int(subtype), environments[0].observation_loading, float(environments[0].generator["subtype_observation_shift"]))
                lower_bounds = []; distinct = []
                for environment in environments:
                    history = environment.simulate(episodes, seed + environment.seed, "history", comparator=comparator)
                    distinct.append(history["distinct_actions"])
                    for name in ("minimum", "maximum", "random"):
                        difference = history["returns"] - controls[(environment.seed, name)]
                        mean = float(np.mean(difference)); se = float(np.std(difference, ddof=1) / math.sqrt(len(difference)))
                        lower_bounds.append(mean - confidence_z * se)
                score = min(lower_bounds)
                rows.append({
                    "profile": contract.profile, "comparator_sha256": comparator.fingerprint,
                    "belief_smoothing": smoothing, "state_offset": offset, "subtype_assumption": subtype,
                    "minimum_paired_lower_bound": score, "minimum_distinct_actions": min(distinct),
                    "selection_role": "development_synthetic_only", "latent_access_at_inference": False,
                })
                if score > best_score:
                    best_score = score; best = comparator
    assert best is not None
    for row in rows: row["selected"] = row["comparator_sha256"] == best.fingerprint
    return best, rows
