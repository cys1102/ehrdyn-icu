"""Versioned KDD198 repair for an audited dense-reward noise defect.

The frozen :class:`R2Environment` is never mutated.  This successor changes
only the dense reward sign from a batch-global alternating counter to an
episode-local exogenous draw.  It is an unauthorised candidate until KDD199
reconstructs and re-accepts every environment.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np

from .full_pomdp_core import HistoryComparator, R2Environment


MECHANISM_VERSION = "kdd198_dense_reward_exogenous_sign_v2"


def independent_dense_values(
    sign_uniforms: np.ndarray,
    score: np.ndarray,
    variance: float,
    response_fraction: float,
) -> np.ndarray:
    """Draw dense values from episode-local exogenous sign uniforms."""
    signs = np.where(np.asarray(sign_uniforms) < 0.5, 1.0, -1.0)
    return math.sqrt(float(variance) / 1.01) * (
        signs + float(response_fraction) * np.tanh(np.asarray(score))
    )


class KDD198EnvironmentV2(R2Environment):
    """Non-promoted mechanism repair; all prior environment hashes stay valid."""

    def _hash(self) -> str:
        payload = {
            "parent_parameter_hash": super()._hash(),
            "mechanism_version": MECHANISM_VERSION,
            "dense_reward_sign": "independent_episode_time_component_uniform",
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def _streams(self, episodes: int, stream_seed: int) -> dict[str, np.ndarray]:
        streams = super()._streams(episodes, stream_seed)
        rng = np.random.default_rng(stream_seed + 198_000_001)
        streams["dense_sign_u"] = rng.random(
            (episodes, self.horizon, max(1, len(self.contract.dense_targets)))
        )
        return streams

    def simulate(
        self,
        episodes: int,
        stream_seed: int,
        policy: str,
        comparator: HistoryComparator | None = None,
        null_response: bool = False,
        exact_actions: np.ndarray | None = None,
        collect_probabilities: bool = False,
    ) -> dict[str, Any]:
        streams = self._streams(episodes, stream_seed)
        states = self._draw(
            np.broadcast_to(self.initial_state_probability, (episodes, self.states)),
            streams["state_u"],
        )
        subtypes = self._draw(
            np.broadcast_to(self.subtype_prevalence, (episodes, self.subtypes)),
            streams["subtype_u"],
        )
        marginal = np.asarray(self.contract.target_action_frequency)[self.supported]
        marginal /= marginal.sum()
        pending = self.supported[
            self._draw(
                np.broadcast_to(marginal, (episodes, len(self.supported))),
                streams["prior_u"],
            )
        ]
        previous_mask = np.ones((episodes, self.contract.feature_dim), dtype=bool)
        recency = np.zeros((episodes, self.contract.feature_dim), dtype=np.float32)
        belief = np.full(episodes, 2.0)
        subtype_belief = np.full(
            episodes, comparator.subtype_assumption if comparator is not None else 1.0
        )
        alive = np.ones(episodes, dtype=bool)
        returns = np.zeros(episodes)
        lengths = np.zeros(episodes, dtype=np.int16)
        actions_log: list[np.ndarray] = []
        previous_log: list[np.ndarray] = []
        probability_log: list[np.ndarray] = []
        mask_log: list[np.ndarray] = []
        state_log: list[np.ndarray] = []
        terminal_events = np.zeros(episodes, dtype=bool)
        component_sums = {
            name: 0.0 for name in (*self.reward_components.keys(), "terminal")
        }
        terminal_emissions = np.zeros(episodes, dtype=np.int16)
        dense_values = {target.name: [] for target in self.contract.dense_targets}
        dense_count = {target.name: 0 for target in self.contract.dense_targets}
        for time in range(self.horizon):
            observation, mask, recency = self._emit(
                states,
                subtypes,
                previous_mask,
                recency,
                pending,
                streams["noise"][:, time],
                streams["mask_u"][:, time],
            )
            previous_mask = mask
            probability = np.zeros((episodes, self.contract.action_count), dtype=float)
            if policy == "ehr_matched":
                bins = self.behavior.context_bin(
                    observation, mask, recency, pending, time
                )
                for index in np.flatnonzero(alive):
                    probability[index] = self.behavior.distribution(
                        int(pending[index]), int(bins[index]), self.contract.action_count
                    )
                action = self._draw(probability, streams["policy_u"][:, time])
            elif policy == "smart_like_exploratory":
                probability[:, self.supported] = 1.0 / len(self.supported)
                action = self._draw(probability, streams["policy_u"][:, time])
            elif policy == "concentrated_behavior":
                probability[:, self.supported] = (
                    1.0 - float(self.generator["concentrated_previous_action_mass"])
                ) * marginal
                probability[np.arange(episodes), pending] += float(
                    self.generator["concentrated_previous_action_mass"]
                )
                action = self._draw(probability, streams["policy_u"][:, time])
            elif policy == "history":
                if comparator is None:
                    raise ValueError("history comparator required")
                belief, subtype_belief = comparator.update(
                    belief, subtype_belief, observation, mask, recency, pending
                )
                action = comparator.actions(belief, subtype_belief)
                probability[np.arange(episodes), action] = 1.0
            elif policy == "minimum":
                action = np.full(episodes, self.supported[0], dtype=np.int16)
                probability[:, self.supported[0]] = 1.0
            elif policy == "maximum":
                action = np.full(episodes, self.supported[-1], dtype=np.int16)
                probability[:, self.supported[-1]] = 1.0
            elif policy == "random":
                probability[:, self.supported] = 1.0 / len(self.supported)
                action = self._draw(probability, streams["policy_u"][:, time])
            elif policy == "severity":
                action = np.asarray(
                    [self.ideal_action(int(state), 1) for state in states],
                    dtype=np.int16,
                )
                probability[np.arange(episodes), action] = 1.0
            elif policy == "oracle":
                if exact_actions is None:
                    raise ValueError("oracle action table required")
                action = exact_actions[time, states, subtypes, pending]
                probability[np.arange(episodes), action] = 1.0
            else:
                raise ValueError(policy)
            action[~alive] = self.supported[0]
            active = np.flatnonzero(alive)
            actions_log.append(action[active].copy())
            previous_log.append(pending[active].copy())
            mask_log.append(mask[active].copy())
            state_log.append(states[active].copy())
            if collect_probabilities:
                probability_log.append(probability[active].copy())
            transition_table = self.null_transition if null_response else self.transition
            transition = transition_table[states, subtypes, pending, action]
            next_states = self._draw(transition, streams["transition_u"][:, time])
            immediate = np.zeros(episodes)
            if not null_response:
                for name, table in self.reward_components.items():
                    values = table[states, subtypes, pending, action]
                    immediate += values
                    component_sums[name] += float(values[alive].sum())
                for component_index, target in enumerate(self.contract.dense_targets):
                    if target.primary:
                        expected_all = self.reward_components["dense_physiology"][
                            states, subtypes, pending, action
                        ]
                        immediate[alive] -= expected_all[alive]
                        component_sums["dense_physiology"] -= float(
                            expected_all[alive].sum()
                        )
                    available = alive & (
                        streams["dense_u"][:, time, component_index]
                        < target.availability_or_nonzero_fraction
                    )
                    indices = np.flatnonzero(available)
                    score = self.construction.response_strength * np.asarray(
                        [
                            0.5
                            - self._mismatch(
                                int(states[i]), int(subtypes[i]), int(action[i])
                            )
                            for i in indices
                        ]
                    )
                    values = independent_dense_values(
                        streams["dense_sign_u"][indices, time, component_index],
                        score,
                        target.variance,
                        float(self.generator["dense_response_fraction"]),
                    )
                    dense_values[target.name].append(values)
                    dense_count[target.name] += len(indices)
                    if target.primary:
                        immediate[indices] += values
                        component_sums["dense_physiology"] += float(values.sum())
            hazard = (
                1.0
                if time == self.horizon - 1
                else self.contract.termination_hazards[time]
            )
            terminate = alive & (streams["termination_u"][:, time] < hazard)
            terminal = np.zeros(episodes)
            if self.contract.primary_reward_type == "terminal" and not null_response:
                death = streams["outcome_u"][:, time] < self.death_probability(next_states)
                terminal_events |= terminate & death
                terminal[terminate] = np.where(
                    death[terminate],
                    self.contract.terminal_reward_minimum,
                    self.contract.terminal_reward_maximum,
                )
                terminal_emissions[terminate] += 1
                component_sums["terminal"] += float(terminal.sum())
            returns += (self.discount**time) * np.where(
                alive, immediate + terminal, 0.0
            )
            lengths[alive] = time + 1
            alive &= ~terminate
            states = next_states
            pending = action
        actions = np.concatenate(actions_log)
        previous = np.concatenate(previous_log)
        masks = np.concatenate(mask_log)
        latent = np.concatenate(state_log)
        dense_summary = {}
        for target in self.contract.dense_targets:
            values = (
                np.concatenate(dense_values[target.name])
                if dense_values[target.name]
                else np.asarray([])
            )
            dense_summary[target.name] = {
                "variance": float(np.var(values)) if len(values) > 1 else math.nan,
                "available_or_nonzero_fraction": dense_count[target.name]
                / max(len(actions), 1),
                "available_count": dense_count[target.name],
                "decision_denominator": len(actions),
            }
        result = {
            "returns": returns,
            "mean_return": float(np.mean(returns)),
            "return_se": float(np.std(returns, ddof=1) / math.sqrt(episodes)),
            "actions": actions,
            "previous_actions": previous,
            "masks": masks,
            "latent_states": latent,
            "mean_horizon": float(np.mean(lengths)),
            "early_termination": float(np.mean(lengths < self.horizon)),
            "missingness": float(np.mean(~masks)),
            "terminal_event_prevalence": float(np.mean(terminal_events))
            if self.contract.terminal_event_prevalence is not None
            else math.nan,
            "terminal_emission_max": int(terminal_emissions.max()),
            "terminal_emission_total": int(terminal_emissions.sum()),
            "dense": dense_summary,
            "distinct_actions": int(np.unique(actions).size),
            "component_sums": component_sums,
        }
        if collect_probabilities:
            result["probabilities"] = np.concatenate(probability_log)
        return result
