from __future__ import annotations

import csv
import math
import random
from pathlib import Path

FIELDS = (
    "synthetic_episode_key",
    "step_index",
    "feature_name",
    "feature_group",
    "current_value",
    "previous_value",
    "target_value",
    "prediction_mean",
    "prediction_std",
    "action_index",
    "reward_component",
)
FEATURES = (
    ("mbp", "vital"),
    ("heart_rate", "vital"),
    ("creatinine", "sparse_lab"),
    ("lactate", "sparse_lab"),
)


def generate_fixture(path: Path, episodes: int = 8, seed: int = 3408) -> int:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for episode in range(episodes):
            previous = {name: rng.uniform(-0.5, 0.5) for name, _group in FEATURES}
            current = dict(previous)
            for step in range(17):
                action = (episode + step) % 3
                for feature_index, (name, group) in enumerate(FEATURES):
                    drift = 0.04 * (action - 1) * (1 if feature_index < 2 else -1)
                    target = current[name] + drift + rng.gauss(0.0, 0.08)
                    prediction = current[name] + 0.5 * drift
                    reward = max(-1.0, min(1.0, current[name] - target))
                    writer.writerow(
                        {
                            "synthetic_episode_key": f"syn-{episode:04d}",
                            "step_index": step,
                            "feature_name": name,
                            "feature_group": group,
                            "current_value": f"{current[name]:.8f}",
                            "previous_value": f"{previous[name]:.8f}",
                            "target_value": f"{target:.8f}",
                            "prediction_mean": f"{prediction:.8f}",
                            "prediction_std": f"{0.15 + 0.01 * math.fabs(current[name]):.8f}",
                            "action_index": action,
                            "reward_component": f"{reward:.8f}",
                        }
                    )
                    previous[name], current[name] = current[name], target
                    row_count += 1
    return row_count
