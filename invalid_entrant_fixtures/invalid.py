#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import time


mode = os.environ.get("INVALID_MODE", "malformed")
counter = 0
for line in sys.stdin:
    counter += 1
    request = json.loads(line)
    rows = len(request.get("payload", {}).get("observations", [0]))
    actions = int(request.get("payload", {}).get("action_count", 2))
    if mode == "slow":
        time.sleep(60)
    if mode == "crash":
        raise SystemExit(7)
    if mode == "malformed":
        print("not-json", flush=True)
        continue
    row = [1.0 / actions] * actions
    if mode == "nonfinite": row[0] = float("nan")
    if mode == "unsupported":
        supported = set(request.get("payload", {}).get("supported_actions", []))
        invalid = next((index for index in range(actions) if index not in supported), actions - 1)
        row = [0.0] * actions; row[invalid] = 1.0
    if mode == "negative": row[0] = -0.1; row[1] += 0.1
    if mode == "nonnormalized": row[0] += 0.2
    if mode == "wrong_dimension": row = row[:-1]
    if mode == "nondeterministic": row[0] += counter * 0.01; row[-1] -= counter * 0.01
    print(json.dumps({"protocol_version":"kdd215.runtime.v1","status":"ok","result":{"probabilities":[row] * rows}}, allow_nan=True), flush=True)
