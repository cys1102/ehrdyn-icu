"""Deterministic synthetic rows for the successor evaluator contract."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from pathlib import Path

from ..errors import ReleaseContractError
from . import FEATURE_NAMES, MODES, NORMALIZATION_RULE, SUCCESSOR_BENCHMARK_VERSION, TASKS
from .evaluator import FIXED_COLUMNS, create_evaluation_contract


def generate_fixture(
    predictions: Path,
    normalization: Path,
    evaluation_contract: Path | None = None,
    *,
    subjects: int = 4,
    transitions: int = 3,
    seed: int = 3408,
) -> dict[str, object]:
    if subjects < 2 or transitions < 2:
        raise ReleaseContractError("Successor fixture requires at least two subjects and two transitions")
    rng = random.Random(seed)
    cluster_column = "synthetic_subject_key"
    sequence_column = "synthetic_sequence_key"
    fields = [cluster_column, sequence_column, *sorted(FIXED_COLUMNS)]
    rows: list[dict[str, object]] = []
    methods = ("persistence_locf", "clean_gaussian_transition")
    for task_index, task in enumerate(TASKS):
        action_count = {"sepsis": 3, "respiratory": 3, "aki": 4, "af_flutter": 2, "heart_failure": 2}[task]
        for subject_index in range(subjects):
            cluster = f"syn-subject-{subject_index:03d}"
            sequence = f"syn-sequence-{task_index:02d}-{subject_index:03d}"
            for mode in MODES:
                for transition in range(transitions):
                    action = str((task_index + subject_index + transition) % action_count)
                    for feature_index, feature in enumerate(FEATURE_NAMES):
                        base = 0.1 * task_index + 0.03 * subject_index + 0.01 * feature_index
                        current = base + 0.04 * transition
                        target = current + 0.08 * math.sin((feature_index + 1) * (transition + 1))
                        observed = int((feature_index + transition + subject_index) % 5 != 0)
                        for method in methods:
                            if method == "persistence_locf":
                                mean = current
                                standard_deviation = ""
                            else:
                                recursive_bias = 0.015 * transition if mode == "conditional_recursive" else 0.0
                                mean = target + recursive_bias + rng.uniform(-0.025, 0.025)
                                standard_deviation = 0.12 + 0.01 * ((feature_index + transition) % 4)
                            rows.append(
                                {
                                    cluster_column: cluster,
                                    sequence_column: sequence,
                                    "role": "synthetic",
                                    "task_id": task,
                                    "mode": mode,
                                    "method_id": method,
                                    "transition_index": transition,
                                    "horizon": 1 if mode == "one_step" else transition + 1,
                                    "feature_name": feature,
                                    "target_value": f"{target:.10f}",
                                    "target_observed": observed,
                                    "prediction_mean": f"{mean:.10f}",
                                    "prediction_std": standard_deviation if standard_deviation == "" else f"{standard_deviation:.10f}",
                                    "action_class": action,
                                }
                            )
    predictions.parent.mkdir(parents=True, exist_ok=True)
    with predictions.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    receipt = {
        "benchmark_version": SUCCESSOR_BENCHMARK_VERSION,
        "fit_role": "train",
        "rule": NORMALIZATION_RULE,
        "synthetic": True,
        "features": {name: {"mean": 0.0, "population_std": 1.0} for name in FEATURE_NAMES},
    }
    normalization.parent.mkdir(parents=True, exist_ok=True)
    normalization.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    contract_path = evaluation_contract or predictions.with_suffix(".evaluation_contract.json")
    _ = create_evaluation_contract(
        predictions,
        normalization,
        contract_path,
        cluster_key_column=cluster_column,
        sequence_key_column=sequence_column,
        synthetic=True,
    )
    return {
        "prediction_rows": len(rows),
        "synthetic_subjects": subjects,
        "tasks": len(TASKS),
        "modes": len(MODES),
        "methods": len(methods),
        "seed": seed,
        "evaluation_contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
    }
