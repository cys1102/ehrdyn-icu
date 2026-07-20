#!/usr/bin/env python3
"""Deterministic observable-history policy example for the KDD215 JSONL API."""
from __future__ import annotations

import json
import math
import sys


def policy(payload: dict) -> list[list[float]]:
    supported = payload["supported_actions"]
    action_count = payload["action_count"]
    output = []
    for observation, mask, recency, previous in zip(
        payload["observations"], payload["masks"], payload["recency"], payload["previous_actions"]
    ):
        observed = max(sum(mask), 1)
        signed = sum(value * (1 if index % 2 == 0 else -1) * mask[index]
                     for index, value in enumerate(observation)) / observed
        severity = min(4, max(0, round((signed / 0.65 + 1.0) * 2.0 - sum(recency) / max(len(recency), 1) * 0.01)))
        if action_count == 25:
            target = severity * 5 + severity
        elif action_count == 4:
            target = 3 if severity >= 2 else 0
        else:
            target = int(severity >= 2)
        logits = [-abs(action - target) / max(action_count - 1, 1) + 0.08 * (action == previous) for action in supported]
        top = max(logits)
        weights = [math.exp(value - top) for value in logits]
        total = sum(weights)
        row = [0.0] * action_count
        for action, weight in zip(supported, weights):
            row[action] = weight / total
        output.append(row)
    return output


for line in sys.stdin:
    try:
        request = json.loads(line)
        if request.get("operation") != "predict_policy":
            raise ValueError("unsupported_operation")
        result = {"probabilities": policy(request["payload"])}
        response = {"protocol_version": "kdd215.runtime.v1", "status": "ok", "result": result}
    except Exception as error:
        response = {"protocol_version": "kdd215.runtime.v1", "status": "error", "error": str(error)}
    print(json.dumps(response, sort_keys=True, separators=(",", ":"), allow_nan=False), flush=True)
