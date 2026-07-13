#!/usr/bin/env python3
"""Build private EHRDyn-ICU arrays from credentialed SQL exports.

All outputs contain restricted row-level data and must remain inside the user's
credentialed environment. Only the aggregate receipt is suitable for review
before a separate privacy check.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from kdd2027_benchmark.config import validate_config_directory  # noqa: E402
from kdd2027_benchmark.split import deterministic_split  # noqa: E402


FEATURES = (
    "age", "gender_male", "weight", "readmission", "elixhauser_score_proxy",
    "heart_rate", "sbp", "mbp", "dbp", "respiratory_rate", "temperature_c", "spo2",
    "shock_index", "sofa_proxy", "gcs_proxy", "fio2", "sirs_proxy", "lactate",
    "pao2", "paco2", "ph", "base_excess", "co2_bicarbonate", "pao2_fio2", "wbc",
    "platelet", "bun", "creatinine", "ptt", "pt", "inr", "ast", "alt",
    "total_bilirubin", "magnesium", "ionized_calcium", "calcium", "urine_output",
    "mechanical_ventilation", "step_id",
)
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURES)}
STATIC = ("age", "gender_male", "readmission", "elixhauser_score_proxy")
ACTION_COLUMNS = (
    "fluid_bolus", "vasopressor", "diuretic", "inotrope", "rate_rhythm_control",
    "anticoagulation", "rrt_crrt", "peep",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--static-context", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs" / "tasks")
    parser.add_argument("--expected", type=Path, default=ROOT / "credentialed" / "expected" / "frozen_task_aggregate_checks.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configs = {str(row["task_id"]): row for row in validate_config_directory(args.config_dir)}
    observations = pd.read_csv(args.observations)
    actions = pd.read_csv(args.actions)
    static = pd.read_csv(args.static_context)
    _require(observations, {"subject_id", "stay_id", "task_id", "step_index", "feature_name", "feature_value"}, "observations")
    _require(actions, {"subject_id", "stay_id", "task_id", "step_index", "mortality_90d", *ACTION_COLUMNS}, "actions")
    _require(static, {"subject_id", "stay_id", "task_id", *STATIC}, "static context")
    unknown = set(observations["feature_name"].dropna().astype(str)) - set(FEATURES)
    if unknown:
        raise ValueError("Unknown observation features: " + ",".join(sorted(unknown)))

    args.output_dir.mkdir(parents=True, exist_ok=False)
    receipts = []
    for task_id, config in configs.items():
        action_rows = actions.loc[actions["task_id"].eq(task_id)].copy()
        if action_rows.empty:
            raise ValueError(f"No action rows for {task_id}")
        receipt = _build_task(task_id, config, observations, action_rows, static, args.output_dir)
        receipts.append(receipt)

    expected = pd.read_csv(args.expected)
    parity = _parity_rows(receipts, expected)
    output = {
        "benchmark_version": next(iter(configs.values()))["benchmark_version"],
        "restricted_outputs": True,
        "task_receipts": receipts,
        "parity": parity,
        "parity_pass": all(row["pass"] for row in parity),
    }
    (args.output_dir / "aggregate_receipt.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"tasks": len(receipts), "parity_pass": output["parity_pass"]}, sort_keys=True))
    return 0 if output["parity_pass"] else 2


def _build_task(task_id, config, all_observations, actions, static, output_dir):
    actions = actions.sort_values(["stay_id", "step_index"], kind="stable").reset_index(drop=True)
    duplicate = actions.duplicated(["stay_id", "step_index"])
    if duplicate.any():
        raise ValueError(f"Duplicate action windows for {task_id}")
    if not actions.groupby("stay_id")["step_index"].agg(list).map(lambda values: values == list(range(18))).all():
        raise ValueError(f"Every {task_id} episode must contain steps 0..17")

    episodes = actions[["subject_id", "stay_id", "mortality_90d"]].drop_duplicates("stay_id").reset_index(drop=True)
    episode_index = {int(stay): index for index, stay in enumerate(episodes["stay_id"])}
    values = np.full((len(episodes), 18, len(FEATURES)), np.nan, dtype=np.float64)
    mask = np.zeros_like(values, dtype=bool)

    task_observations = all_observations.loc[all_observations["task_id"].eq(task_id)]
    grouped = task_observations.groupby(["stay_id", "step_index", "feature_name"], observed=True)["feature_value"].median()
    for (stay_id, step, feature), value in grouped.items():
        if int(stay_id) not in episode_index or not np.isfinite(value):
            continue
        e, t, f = episode_index[int(stay_id)], int(step), FEATURE_INDEX[str(feature)]
        values[e, t, f] = float(value)
        mask[e, t, f] = True

    task_static = static.loc[static["task_id"].eq(task_id)].drop_duplicates("stay_id").set_index("stay_id")
    for row in episodes.itertuples(index=False):
        source = task_static.loc[int(row.stay_id)]
        for feature in STATIC:
            value = float(source[feature])
            if np.isfinite(value):
                values[episode_index[int(row.stay_id)], :, FEATURE_INDEX[feature]] = value
                mask[episode_index[int(row.stay_id)], :, FEATURE_INDEX[feature]] = True
    values[:, :, FEATURE_INDEX["step_id"]] = np.arange(18, dtype=np.float64)[None, :]
    mask[:, :, FEATURE_INDEX["step_id"]] = True

    action_values = actions.set_index(["stay_id", "step_index"])
    encoded_actions = np.zeros((len(episodes), 18), dtype=np.int64)
    vasopressor = np.zeros((len(episodes), 18), dtype=np.float64)
    for stay_id, episode in episode_index.items():
        rows = action_values.loc[stay_id].sort_index()
        encoded_actions[episode] = _encode_action(task_id, rows)
        vasopressor[episode] = rows["vasopressor"].fillna(0).to_numpy(dtype=np.float64)

    _derive(values, mask, vasopressor)
    split = np.asarray([deterministic_split(str(value)) for value in episodes["subject_id"]], dtype="U5")
    transformed, imputed, recency, stats = _preprocess(values, mask, split)
    terminal_reward = np.zeros((len(episodes), 18), dtype=np.float32)
    terminal_reward[:, -1] = np.where(episodes["mortality_90d"].to_numpy(dtype=int) == 1, -1.0, 1.0)
    configured_action_count = int(config["action"]["action_count"])
    if encoded_actions.min() < 0 or encoded_actions.max() >= configured_action_count:
        raise ValueError(
            f"{task_id} produced action outside [0, {configured_action_count - 1}]"
        )
    action_ids, action_counts = np.unique(encoded_actions, return_counts=True)
    action_histogram = {
        str(index): int(count) for index, count in zip(action_ids, action_counts, strict=True)
    }
    np.savez_compressed(
        output_dir / f"{task_id}.restricted.npz",
        values=transformed,
        imputed_values=imputed,
        mask=mask.astype(np.float32),
        log_recency=recency,
        action_index=encoded_actions,
        terminal_reward=terminal_reward,
        split=split,
        local_subject_key=episodes["subject_id"].to_numpy(),
        local_stay_key=episodes["stay_id"].to_numpy(),
    )
    (output_dir / f"{task_id}.preprocessing.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "task_id": task_id,
        "episodes": len(episodes),
        "windows": len(episodes) * 18,
        "subjects": int(episodes["subject_id"].nunique()),
        "observation_fraction": float(mask.mean()),
        "mortality_90d": float(episodes["mortality_90d"].mean()),
        "split_counts": {name: int(np.sum(split == name)) for name in ("train", "val", "test")},
        "action_count_observed": int(action_ids.size),
        "configured_action_count": configured_action_count,
        "action_count_matches_config": int(action_ids.size) == configured_action_count,
        "action_class_counts": action_histogram,
    }


def _encode_action(task_id: str, rows: pd.DataFrame) -> np.ndarray:
    def binned(column: str, edges: tuple[float, ...]) -> np.ndarray:
        values = rows[column].fillna(0).to_numpy(dtype=float)
        output = np.zeros(len(values), dtype=np.int64)
        positive = values > 0
        output[positive] = np.searchsorted(np.asarray(edges), values[positive], side="right") + 1
        return output

    if task_id == "kdd2027_sepsis_vasopressor_3bin":
        return binned("vasopressor", (0.495270386338,))
    if task_id == "kdd2027_respiratory_peep_5bin":
        return binned("peep", (5.0, 6.0))
    if task_id == "kdd2027_aki_diuretic_rrt_factorized_3bin":
        return 3 * binned("diuretic", (1.0,)) + binned("rrt_crrt", (1.0,))
    if task_id == "kdd2027_af_rate_anticoag_compact_3bin":
        return 3 * binned("rate_rhythm_control", (1.0,)) + binned("anticoagulation", (1.0,))
    if task_id == "kdd2027_hf_diuretic_binary":
        return (rows["diuretic"].fillna(0).to_numpy(dtype=float) > 0).astype(np.int64)
    if task_id == "kdd2027_ami_hemodynamic_compact_3bin":
        return 3 * binned("vasopressor", (0.699833810329,)) + binned("inotrope", (1.67291665077,))
    if task_id == "kdd2027_shock_fluid_bolus_binary":
        return (rows["fluid_bolus"].fillna(0).to_numpy(dtype=float) > 0).astype(np.int64)
    raise ValueError(f"No frozen action encoder for {task_id}")


def _derive(values: np.ndarray, mask: np.ndarray, vasopressor: np.ndarray) -> None:
    def get(name):
        index = FEATURE_INDEX[name]
        return values[:, :, index], mask[:, :, index]

    def put(name, derived, observed):
        index = FEATURE_INDEX[name]
        values[:, :, index] = np.where(observed, derived, values[:, :, index])
        mask[:, :, index] = observed

    hr, hr_m = get("heart_rate"); sbp, sbp_m = get("sbp")
    put("shock_index", hr / np.maximum(sbp, 1e-6), hr_m & sbp_m & (sbp > 0))
    pao2, pao2_m = get("pao2"); fio2, fio2_m = get("fio2")
    fio2_fraction = np.where(fio2 > 1, fio2 / 100.0, fio2)
    put("pao2_fio2", pao2 / np.maximum(fio2_fraction, 1e-6), pao2_m & fio2_m & (fio2_fraction > 0))
    temp, temp_m = get("temperature_c"); rr, rr_m = get("respiratory_rate")
    paco2, paco2_m = get("paco2"); wbc, wbc_m = get("wbc")
    sirs = np.zeros_like(temp, dtype=float)
    sirs += (((temp >= 38) | (temp <= 36)) & temp_m).astype(float)
    sirs += ((hr > 90) & hr_m).astype(float)
    sirs += (((rr > 20) & rr_m) | ((paco2 < 32) & paco2_m)).astype(float)
    sirs += (((wbc > 12) | (wbc < 4)) & wbc_m).astype(float)
    put("sirs_proxy", sirs.astype(float), temp_m | hr_m | rr_m | paco2_m | wbc_m)
    pf, pf_m = get("pao2_fio2"); platelet, platelet_m = get("platelet")
    bilirubin, bilirubin_m = get("total_bilirubin"); mbp, mbp_m = get("mbp")
    gcs, gcs_m = get("gcs_proxy"); creatinine, creatinine_m = get("creatinine")
    urine, urine_m = get("urine_output")
    sofa = np.select([pf < 100, pf < 200, pf < 300, pf < 400], [4, 3, 2, 1], default=0) * pf_m
    sofa += np.select([platelet < 20, platelet < 50, platelet < 100, platelet < 150], [4, 3, 2, 1], default=0) * platelet_m
    sofa += np.select([bilirubin >= 12, bilirubin >= 6, bilirubin >= 2, bilirubin >= 1.2], [4, 3, 2, 1], default=0) * bilirubin_m
    sofa += np.where(vasopressor > 0, 3, np.where(mbp_m & (mbp < 70), 1, 0))
    sofa += np.select([gcs < 6, gcs < 10, gcs < 13, gcs < 15], [4, 3, 2, 1], default=0) * gcs_m
    renal = np.maximum(
        np.select([creatinine >= 5, creatinine >= 3.5, creatinine >= 2, creatinine >= 1.2], [4, 3, 2, 1], default=0) * creatinine_m,
        np.select([urine < 200, urine < 500], [4, 3], default=0) * urine_m,
    )
    sofa += renal
    put("sofa_proxy", sofa.astype(float), pf_m | platelet_m | bilirubin_m | mbp_m | gcs_m | creatinine_m | urine_m | (vasopressor > 0))


def _preprocess(values, mask, split):
    train = split == "train"
    median = np.zeros(len(FEATURES), dtype=float)
    for feature in range(len(FEATURES)):
        observed = values[train, :, feature][mask[train, :, feature]]
        median[feature] = float(np.median(observed)) if len(observed) else 0.0
    imputed = np.empty_like(values, dtype=float)
    recency = np.zeros_like(values, dtype=np.float32)
    for feature in range(len(FEATURES)):
        current = np.full(len(values), median[feature], dtype=float)
        elapsed = np.zeros(len(values), dtype=float)
        for step in range(18):
            observed = mask[:, step, feature] & np.isfinite(values[:, step, feature])
            current[observed] = values[observed, step, feature]
            elapsed = np.where(observed, 0.0, elapsed + 4.0)
            imputed[:, step, feature] = current
            recency[:, step, feature] = np.log1p(elapsed)
    train_values = imputed[train].reshape(-1, len(FEATURES))
    mean = train_values.mean(axis=0)
    scale = np.maximum(train_values.std(axis=0), 1e-6)
    transformed = ((imputed - mean) / scale).astype(np.float32)
    for name in ("gender_male", "readmission"):
        transformed[:, :, FEATURE_INDEX[name]] = imputed[:, :, FEATURE_INDEX[name]]
    stats = {
        "feature_names": FEATURES,
        "train_median": median.tolist(),
        "train_imputed_mean": mean.tolist(),
        "train_imputed_scale": scale.tolist(),
        "binary_preserved": ["gender_male", "readmission"],
        "imputation": "past-only LOCF initialized by train median",
        "scaling": "train imputed mean/std",
    }
    return transformed, imputed.astype(np.float32), recency, stats


def _parity_rows(receipts, expected):
    by_task = {row["task_id"]: row for row in receipts}
    result = []
    for row in expected.itertuples(index=False):
        metric = str(row.metric)
        observed = by_task[str(row.task_id)][{
            "full_eligible_episodes": "episodes",
            "full_eligible_windows": "windows",
            "full_subjects": "subjects",
            "full_observation_fraction": "observation_fraction",
            "mortality_90d": "mortality_90d",
            "action_count_observed": "action_count_observed",
        }[metric]]
        tolerance = float(row.absolute_tolerance)
        result.append({"task_id": str(row.task_id), "metric": metric, "expected": float(row.expected_value), "observed": float(observed), "absolute_tolerance": tolerance, "pass": abs(float(observed) - float(row.expected_value)) <= tolerance})
    return result


def _require(frame, columns, label):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {','.join(sorted(missing))}")


if __name__ == "__main__":
    raise SystemExit(main())
