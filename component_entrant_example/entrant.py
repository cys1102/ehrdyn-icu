#!/usr/bin/env python3
"""Observable-history Gaussian component entrant; transition-only by design."""
from __future__ import annotations

import json
import sys


for line in sys.stdin:
    try:
        request = json.loads(line)
        if request.get("operation") != "predict_component":
            raise ValueError("unsupported_operation")
        payload = request["payload"]
        means = [list(row) for row in payload["observations"]]
        scales = [[0.75 + min(value, 8) * 0.02 for value in row] for row in payload["recency"]]
        response = {
            "protocol_version": "kdd215.runtime.v1", "status": "ok",
            "result": {"mean": means, "scale": scales, "distribution": "independent_gaussian"},
        }
    except Exception as error:
        response = {"protocol_version": "kdd215.runtime.v1", "status": "error", "error": str(error)}
    print(json.dumps(response, sort_keys=True, separators=(",", ":"), allow_nan=False), flush=True)
