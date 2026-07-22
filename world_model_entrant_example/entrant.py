#!/usr/bin/env python3
"""Minimal entrant-owned recursive model; depends only on the public JSONL contract."""
from __future__ import annotations

import json
import math
import sys


MODE = sys.argv[1] if len(sys.argv) > 1 else "point"
PROTOCOL = "kdd235a.runtime.v1"


def predict(payload: dict, recursive: bool) -> dict:
    observations = payload["observations"]
    masks = payload["masks"]
    sequences = payload["action_sequences"]
    horizon = int(payload["horizon"] if recursive else 1)
    means = []
    member_rows = [[], [], []]
    for row, mask, actions in zip(observations, masks, sequences):
        state = [float(value) if observed else 0.0 for value, observed in zip(row, mask)]
        for step in range(horizon):
            action = int(actions[step])
            state = [0.94 * value + 0.01 * (action + 1) * (1 if index % 2 == 0 else -1)
                     for index, value in enumerate(state)]
            means.append(list(state))
            for member_index, offset in enumerate((-0.05, 0.0, 0.05)):
                member_rows[member_index].append([value + offset for value in state])
    result = {"schema_version": "kdd235a.prediction.v1", "prediction_type": MODE,
              "horizon": horizon, "mean": means}
    if MODE == "independent_gaussian":
        result["scale"] = [[0.75 for _ in row] for row in means]
    elif MODE == "gaussian_ensemble":
        scales = [[[0.5 + 0.1 * member for _ in row] for row in means] for member in range(3)]
        result["members"] = [{"mean": member_rows[index], "scale": scales[index]} for index in range(3)]
        within, between, total = [], [], []
        for row_index, row in enumerate(means):
            within_row, between_row, total_row = [], [], []
            for feature_index, mean in enumerate(row):
                within_value = sum(scales[m][row_index][feature_index] ** 2 for m in range(3)) / 3
                between_value = sum((member_rows[m][row_index][feature_index] - mean) ** 2 for m in range(3)) / 3
                within_row.append(within_value); between_row.append(between_value); total_row.append(within_value + between_value)
            within.append(within_row); between.append(between_row); total.append(total_row)
        result.update({"within_variance": within, "between_variance": between, "total_variance": total})
    return result


def policy(payload: dict) -> dict:
    action_count = int(payload["action_count"])
    supported = [int(value) for value in payload["supported_actions"]]
    probabilities = []
    for row in payload["observations"]:
        chosen = supported[int(abs(sum(row))) % len(supported)]
        vector = [0.0] * action_count
        vector[chosen] = 1.0
        probabilities.append(vector)
    return {"schema_version": "kdd235a.policy.v1", "planner": {
        "name": "H4_support_only_sequence_categorical_CEM", "horizon": 4, "candidates": 64,
        "iterations": 3, "elites": 8, "smoothing": 0.2, "support_only": True,
        "execute_first_action": True, "planner_seed": 1903408}, "probabilities": probabilities}


for line in sys.stdin:
    try:
        request = json.loads(line)
        operation, payload = request["operation"], request["payload"]
        if operation in ("initialize", "fit_or_load"):
            result = {"accepted": True, "checkpoint_id": "minimal-public-example-v1"}
        elif operation == "predict_one_step":
            result = predict(payload, False)
        elif operation == "predict_rollout":
            result = predict(payload, True)
        elif operation == "predict_policy":
            result = policy(payload)
        else:
            raise ValueError("unsupported_operation")
        response = {"protocol_version": PROTOCOL, "status": "ok", "result": result}
    except Exception as error:
        response = {"protocol_version": PROTOCOL, "status": "error", "error": str(error)}
    print(json.dumps(response, sort_keys=True, separators=(",", ":"), allow_nan=False), flush=True)
