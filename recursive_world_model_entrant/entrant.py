#!/usr/bin/env python3
"""Entrant-owned recurrent Gaussian model using only the public JSONL protocol."""
from __future__ import annotations

import json
import math
import random
import sys


PROTOCOL = "kdd235a.runtime.v1"
PLANNER_SEED = 1903408
METADATA: dict = {}
PLANNED_VECTOR: list[float] | None = None


def rollout(payload: dict, one_step: bool = False) -> dict:
    horizon = 1 if one_step else int(payload["horizon"])
    means: list[list[float]] = []
    scales: list[list[float]] = []
    for observation, mask, recency, actions in zip(
        payload["observations"], payload["masks"], payload["recency"], payload["action_sequences"]
    ):
        state = [float(value) if observed else 0.0 for value, observed in zip(observation, mask)]
        memory = sum(float(value) for value in recency) / max(len(recency), 1)
        for step in range(horizon):
            action = int(actions[step])
            centered_action = action / max(int(METADATA.get("action_count", 2)) - 1, 1) - 0.5
            state = [
                0.94 * value + 0.015 * centered_action * (1.0 if index % 2 == 0 else -1.0)
                + 0.002 * math.tanh(memory)
                for index, value in enumerate(state)
            ]
            memory = 0.85 * memory + 0.15
            means.append(list(state))
            scales.append([0.55 + 0.03 * min(memory, 5.0) for _ in state])
    return {
        "schema_version": "kdd235a.prediction.v1",
        "prediction_type": "independent_gaussian",
        "horizon": horizon,
        "mean": means,
        "scale": scales,
    }


def _sequence_score(sequence: list[int], supported: list[int]) -> float:
    denominator = max(max(supported) - min(supported), 1)
    normalized = [(action - min(supported)) / denominator for action in sequence]
    switching = sum(abs(normalized[index] - normalized[index - 1]) for index in range(1, 4))
    physiology = sum((value - 0.42) ** 2 for value in normalized)
    return -(physiology + 0.08 * switching)


def _frozen_h4_vector() -> list[float]:
    global PLANNED_VECTOR
    if PLANNED_VECTOR is not None:
        return PLANNED_VECTOR
    action_count = int(METADATA["action_count"])
    supported = [int(value) for value in METADATA["supported_actions"]]
    probabilities = [[1.0 / len(supported) for _ in supported] for _ in range(4)]
    rng = random.Random(PLANNER_SEED + 1009 * int(METADATA["environment_seed"]))
    best_sequence: list[int] | None = None
    best_score = -math.inf
    for _ in range(3):
        candidates: list[list[int]] = []
        for _candidate in range(64):
            sequence = [rng.choices(supported, weights=probabilities[step], k=1)[0] for step in range(4)]
            candidates.append(sequence)
        ranked = sorted(enumerate(candidates), key=lambda item: (-_sequence_score(item[1], supported), item[0]))
        local_score = _sequence_score(ranked[0][1], supported)
        if local_score > best_score + 1e-12:
            best_score = local_score
            best_sequence = list(ranked[0][1])
        elites = [sequence for _, sequence in ranked[:8]]
        for step in range(4):
            empirical = [sum(sequence[step] == action for sequence in elites) / 8.0 for action in supported]
            probabilities[step] = [0.2 * old + 0.8 * new for old, new in zip(probabilities[step], empirical)]
            total = sum(probabilities[step])
            probabilities[step] = [value / total for value in probabilities[step]]
    if best_sequence is None:
        raise RuntimeError("planner_no_candidate")
    PLANNED_VECTOR = [0.0] * action_count
    PLANNED_VECTOR[best_sequence[0]] = 1.0
    return PLANNED_VECTOR


def policy(payload: dict) -> dict:
    vector = _frozen_h4_vector()
    return {
        "schema_version": "kdd235a.policy.v1",
        "planner": {
            "name": "H4_support_only_sequence_categorical_CEM",
            "horizon": 4,
            "candidates": 64,
            "iterations": 3,
            "elites": 8,
            "smoothing": 0.2,
            "support_only": True,
            "execute_first_action": True,
            "planner_seed": PLANNER_SEED,
        },
        "probabilities": [list(vector) for _ in payload["observations"]],
    }


for line in sys.stdin:
    try:
        request = json.loads(line)
        operation = request["operation"]
        payload = request["payload"]
        if operation == "initialize":
            METADATA.clear()
            METADATA.update(payload)
            PLANNED_VECTOR = None
            result = {"accepted": True, "checkpoint_id": "kdd235b-recurrent-gaussian-v1"}
        elif operation == "fit_or_load":
            if payload.get("sealed_final_opened") is not False:
                raise ValueError("final_role_must_remain_closed_during_fit")
            result = {"accepted": True, "checkpoint_id": "kdd235b-recurrent-gaussian-v1"}
        elif operation == "predict_one_step":
            result = rollout(payload, True)
        elif operation == "predict_rollout":
            result = rollout(payload, False)
        elif operation == "predict_policy":
            result = policy(payload)
        else:
            raise ValueError("unsupported_operation")
        response = {"protocol_version": PROTOCOL, "status": "ok", "result": result}
    except Exception as error:
        response = {"protocol_version": PROTOCOL, "status": "error", "error": str(error)}
    print(json.dumps(response, sort_keys=True, separators=(",", ":"), allow_nan=False), flush=True)
