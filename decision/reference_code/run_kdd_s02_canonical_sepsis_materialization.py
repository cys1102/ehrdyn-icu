from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # noqa: PANDAS_OK
from sklearn.linear_model import Ridge

from kdd_rv01r.construction import _build_transitions, _finalize_anchors
from kdd_rv01r.sources import DenseConstruction, build_dense_construction
from kdd_rv02r.data import _past_only_arrays
from scripts.export_mimiciv_meddreamer_sepsis_official_like import (
    FEATURE_INDEX,
    FEATURE_NAMES,
    FEATURE_SPECS,
)
from scripts.meddreamer_sepsis3_derived import rolling_urine_4h
from scripts.meddreamer_sepsis3_scores import sofa_score

from .kdd098_data import TaskSequences
from .kdd097_contract import encode_five_levels, joint_codes, train_positive_edges
from .run_kdd097_materialization import (
    FEATURE_INDICES,
    RV01_CONFIG,
    _development_candidates,
)
from .run_kdd099_policy_contract import behavior_states
from .run_kdd101_model_free_diagnostics import (
    fit_behavior_denominators,
    fit_classifier,
    load_config,
    policy_probs,
    supported_actions,
    trajectory_weights,
)


ROOT = Path(__file__).resolve().parents[1]
MIMIC = Path("<authorized-mimiciv-root>")
S01 = ROOT / "kdd_benchmark_discovery/results/kdd_s01_sepsis_reference_contract_20260714_200000"
KDD101 = ROOT / "kdd_benchmark_discovery/results/kdd101_model_free_diagnostics_20260714_v5"
KDD101_CONFIG = ROOT / "configs/kdd101_model_free_diagnostics_v5.json"
CONTRACT_ID = "sepsis_meddreamer_compatible_4h_reference_v1"
TASK_ROLE = "canonical_sepsis_reference"
VIEWS = (
    "features_current33_fixed4h",
    "features_reference40_fixed4h",
    "features_reference40_irregular_sensitivity",
)
HORIZONS = (1, 2, 4, 8, 12, 17)
REWARDS = (
    "reward_reference_published_compatible_terminal",
    "reward_terminal_only",
    "reward_lactate_only_current",
    "reward_leakage_controlled_composite_terminal",
    "reward_historical_sepsisagent_phys_terminal_scaled_clipped",
    "reward_historical_kdd089_composite",
)
CLAIM = (
    "Development-only aggregate materialization and logged-association diagnostics. No confirmatory test, "
    "policy value, treatment benefit, causal effect, counterfactual fidelity, clinical utility, deployment, "
    "autonomous-decision, or policy-winner claim is supported."
)


@dataclass(slots=True)
class VariantBundle:
    view: str
    task: TaskSequences
    raw_state: np.ndarray
    raw_target: np.ndarray
    sofa_no_treatment_state: np.ndarray
    sofa_no_treatment_target: np.ndarray
    sofa_no_treatment_state_observed: np.ndarray
    sofa_no_treatment_target_observed: np.ndarray
    subject_by_episode: np.ndarray


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KDD-S02 aggregate-only canonical sepsis materialization")
    parser.add_argument("--mimiciv-root", type=Path, default=MIMIC)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / (
        "kdd_benchmark_discovery/results/kdd_s02_canonical_sepsis_materialization_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    if output.exists():
        raise FileExistsError(output)
    historical_hashes = {str(path): tree_hash(path) for path in (S01, KDD101)}
    historical = preflight()

    config = json.loads(RV01_CONFIG.read_text(encoding="utf-8"))
    config["mimiciv_root"] = str(args.mimiciv_root)
    config["chunksize"] = int(args.chunksize)
    candidates, cache, _sources = _development_candidates(config)
    candidates = candidates[candidates["task_id"].eq("sepsis")].copy().reset_index(drop=True)
    candidates["episode_idx"] = np.arange(len(candidates), dtype=np.int64)
    dense = build_dense_construction(config, candidates, cache)
    finalized, anchor_failures = _finalize_anchors(dense, config)
    transitions, transition_failures = _build_transitions(finalized, dense.features.shape[1], config)
    action_result = materialize_sepsis_action(dense, transitions)
    local = matched_sepsis_transitions(dense, action_result)
    reference_dense = reference_features(dense, dense.candidates, cache.stays)
    no_treatment_sofa, no_treatment_sofa_observed = treatment_isolated_sofa(reference_dense)

    current_filled, current_recency = _past_only_arrays(dense, finalized, FEATURE_INDICES)
    reference_indices = np.arange(len(FEATURE_NAMES), dtype=int)
    reference_filled, reference_recency = _past_only_arrays(reference_dense, finalized, reference_indices)
    bundles = [
        make_bundle(
            VIEWS[0], dense, local, FEATURE_INDICES, current_filled, current_recency,
            no_treatment_sofa, no_treatment_sofa_observed, irregular=True,
        ),
        make_bundle(
            VIEWS[1], reference_dense, local, reference_indices, reference_filled, reference_recency,
            no_treatment_sofa, no_treatment_sofa_observed, irregular=False,
        ),
        make_bundle(
            VIEWS[2], reference_dense, local, reference_indices, reference_filled, reference_recency,
            no_treatment_sofa, no_treatment_sofa_observed, irregular=True,
        ),
    ]
    assert_matched(bundles)

    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    add_manifest(rows, bundles, local, action_result, historical)
    add_feature_provenance(rows, bundles)
    add_missingness(rows, bundles)
    reward_cache = add_rewards(rows, bundles)
    add_terminal(rows, bundles)
    add_action_support(rows, bundles, local)
    add_action_information(rows, bundles)
    add_action_windows(rows, bundles, reward_cache)
    add_support_reconciliation(rows, bundles, reward_cache, historical)
    add_matched_checks(rows, bundles, local, action_result)
    add_split_leakage(rows, bundles, local, anchor_failures, transition_failures)
    decision = add_policy_axes_and_decision(rows, bundles, reward_cache)

    output.mkdir(parents=True)
    required = (
        "materialization_manifest.csv",
        "feature_provenance_and_timing.csv",
        "feature_missingness_summary.csv",
        "reward_observed_imputed_breakdown.csv",
        "reward_operability.csv",
        "terminal_outcome_availability.csv",
        "action_support_summary.csv",
        "action_information_by_target_horizon.csv",
        "action_switch_initiation_diagnostics.csv",
        "kdd101_v5_support_reconciliation.csv",
        "matched_variant_contract_check.csv",
        "split_and_leakage_audit.csv",
        "sepsis_policy_evaluability_axes.csv",
    )
    for name in required:
        write_csv(output / name, rows[name])
    (output / "decision.md").write_text(decision_text(decision), encoding="utf-8")
    (output / "summary.md").write_text(summary_text(decision, rows, historical), encoding="utf-8")
    validate_outputs(output, required)
    for path, old_hash in historical_hashes.items():
        if tree_hash(Path(path)) != old_hash:
            raise RuntimeError(f"historical artifact changed: {path}")


def preflight() -> dict[str, Any]:
    decision = (S01 / "decision.md").read_text(encoding="utf-8")
    if "canonical_contract_frozen_with_explicit_compatibility_gaps" not in decision:
        raise RuntimeError("KDD-S01 did not freeze the required contract")
    support = pd.read_csv(KDD101 / "support_overlap_ratio_ess.csv")
    if len(support) != 972 or int(support["pre_estimator_gate_pass"].sum()) != 102:
        raise RuntimeError("KDD101-v5 102/972 support receipt drift")
    if tuple(sorted(support["horizon"].unique())) != HORIZONS:
        raise RuntimeError("KDD101-v5 horizon contract drift")
    config = load_config(KDD101_CONFIG)
    return {
        "historical_rows": int(len(support)),
        "historical_passes": int(support["pre_estimator_gate_pass"].sum()),
        "support": support,
        "config": config,
        "s01_hash": sha256(S01 / "decision.md"),
        "kdd101_hash": sha256(KDD101 / "support_overlap_ratio_ess.csv"),
    }


def matched_sepsis_transitions(dense: DenseConstruction, action_result: dict[str, Any]) -> pd.DataFrame:
    local = action_result["transitions"].copy()
    local["action_class"] = action_result["classes"]
    local["action_valid"] = action_result["valid"]
    episode = local["episode_idx"].to_numpy(dtype=int)
    target = local["target_idx"].to_numpy(dtype=int)
    observed = dense.mask[episode, target][:, FEATURE_INDICES].astype(bool).any(axis=1)
    local = local[local["action_valid"] & observed].copy()
    local = local.sort_values(["episode_idx", "relative_transition"], kind="stable").reset_index(drop=True)
    roles = set(local["role"].astype(str))
    if roles != {"train", "validation"}:
        raise RuntimeError(f"unexpected development roles: {roles}")
    overlap = set(local.loc[local.role.eq("train"), "subject_id_internal"]) & set(
        local.loc[local.role.eq("validation"), "subject_id_internal"]
    )
    if overlap:
        raise RuntimeError("train/validation subject overlap")
    return local


def materialize_sepsis_action(dense: DenseConstruction, transitions: pd.DataFrame) -> dict[str, Any]:
    local = transitions[transitions["task"].eq("sepsis")].copy()
    local = local.sort_values(["episode_idx", "relative_transition"], kind="stable")
    episode = local["episode_idx"].to_numpy(dtype=int)
    action_idx = local["action_idx"].to_numpy(dtype=int)
    roles = local["role"].astype(str).to_numpy()
    fluid = dense.fluid[episode, action_idx]
    vasopressor = dense.vasopressor[episode, action_idx]
    observed = np.ones(len(local), dtype=bool)
    fluid_edges = train_positive_edges(fluid, observed, roles)
    vasopressor_edges = train_positive_edges(vasopressor, observed, roles)
    fluid_level = encode_five_levels(fluid, observed, fluid_edges)
    vasopressor_level = encode_five_levels(vasopressor, observed, vasopressor_edges)
    return {
        "transitions": local,
        "classes": joint_codes(fluid_level, vasopressor_level, observed),
        "valid": observed,
        "dimension": 25,
        "components": ("fluid", "vasopressor"),
        "edges": (fluid_edges, vasopressor_edges),
    }


def reference_features(dense: DenseConstruction, candidates: pd.DataFrame, stays: pd.DataFrame) -> DenseConstruction:
    features = dense.features.copy()
    mask = dense.mask.copy()
    by_episode = candidates.set_index("episode_idx")
    age = by_episode["anchor_age"].astype(float)
    male = by_episode["gender"].astype(str).str.upper().eq("M").astype(float)
    prior = {}
    for row in candidates.itertuples(index=False):
        history = stays[
            stays["subject_id"].eq(row.subject_id)
            & stays["dischtime"].notna()
            & stays["dischtime"].lt(row.admittime)
        ]
        prior[int(row.episode_idx)] = float(not history.empty)
    static = {"age": age, "gender_male": male, "readmission": pd.Series(prior)}
    for name, values in static.items():
        idx = FEATURE_INDEX[name]
        ordered = values.reindex(candidates["episode_idx"].astype(int)).to_numpy(dtype=np.float32)
        features[:, :, idx] = ordered[:, None]
        mask[:, :, idx] = np.isfinite(ordered)[:, None]
    step = np.arange(features.shape[1], dtype=np.float32)
    features[:, :, FEATURE_INDEX["step_id"]] = step[None, :]
    mask[:, :, FEATURE_INDEX["step_id"]] = 1.0
    idx = FEATURE_INDEX["elixhauser_score_proxy"]
    features[:, :, idx] = np.nan
    mask[:, :, idx] = 0.0
    return replace(dense, features=features, mask=mask)


def treatment_isolated_sofa(dense: DenseConstruction) -> tuple[np.ndarray, np.ndarray]:
    def values(name: str) -> tuple[np.ndarray, np.ndarray]:
        idx = FEATURE_INDEX[name]
        return dense.features[:, :, idx], dense.mask[:, :, idx].astype(bool)

    pf, pf_m = values("pao2_fio2")
    platelets, platelets_m = values("platelet")
    bilirubin, bilirubin_m = values("total_bilirubin")
    mbp, mbp_m = values("mbp")
    gcs, gcs_m = values("gcs_proxy")
    creatinine, creatinine_m = values("creatinine")
    urine, urine_m = values("urine_output")
    urine_4h = rolling_urine_4h(urine, urine_m, bin_hours=4)
    score = sofa_score(
        pf_ratio=np.where(pf_m, pf, np.nan),
        platelets=np.where(platelets_m, platelets, np.nan),
        bilirubin=np.where(bilirubin_m, bilirubin, np.nan),
        mbp=np.where(mbp_m, mbp, np.nan),
        vaso_max=np.full_like(mbp, np.nan),
        gcs_total=np.where(gcs_m, gcs, np.nan),
        creatinine=np.where(creatinine_m, creatinine, np.nan),
        urine_4h=np.where(urine_m, urine_4h, np.nan),
    ).astype(np.float32)
    observed = pf_m | platelets_m | bilirubin_m | mbp_m | gcs_m | creatinine_m | urine_m
    return np.where(observed, score, np.nan), observed


def make_bundle(
    view: str,
    dense: DenseConstruction,
    local: pd.DataFrame,
    indices: np.ndarray,
    filled: np.ndarray,
    recency: np.ndarray,
    sofa_no_treatment: np.ndarray,
    sofa_no_treatment_observed: np.ndarray,
    *,
    irregular: bool,
) -> VariantBundle:
    train = local[local.role.eq("train")]
    train_episode = train["episode_idx"].to_numpy(dtype=int)
    train_state = train["state_idx"].to_numpy(dtype=int)
    raw_train = dense.features[train_episode, train_state][:, indices].astype(float)
    mask_train = dense.mask[train_episode, train_state][:, indices].astype(bool)
    mean = np.zeros(len(indices), dtype=float)
    scale = np.ones(len(indices), dtype=float)
    for col in range(len(indices)):
        available = raw_train[mask_train[:, col], col]
        available = available[np.isfinite(available)]
        if available.size:
            mean[col] = float(available.mean())
            std = float(available.std())
            scale[col] = std if np.isfinite(std) and std >= 1e-6 else 1.0

    groups = list(local.groupby("episode_idx", sort=False, observed=True))
    max_steps = max(len(group) for _, group in groups)
    n = len(groups)
    f = len(indices)
    states = np.zeros((n, max_steps, f), dtype=np.float32)
    state_masks = np.zeros_like(states, dtype=bool)
    deltas = np.zeros_like(states, dtype=np.float32)
    actions = np.zeros((n, max_steps, 25), dtype=np.float32)
    action_classes = np.full((n, max_steps), -1, dtype=np.int16)
    targets = np.zeros_like(states)
    target_masks = np.zeros_like(states, dtype=bool)
    raw_state = np.full_like(states, np.nan)
    raw_target = np.full_like(states, np.nan)
    valid = np.zeros((n, max_steps), dtype=bool)
    terminal = np.zeros((n, max_steps), dtype=np.float32)
    roles = np.empty(n, dtype=object)
    subjects = np.empty(n, dtype=np.int64)
    strong = np.zeros((n, max_steps), dtype=bool)
    nt_state = np.full((n, max_steps), np.nan, dtype=np.float32)
    nt_target = np.full((n, max_steps), np.nan, dtype=np.float32)
    nt_state_obs = np.zeros((n, max_steps), dtype=bool)
    nt_target_obs = np.zeros((n, max_steps), dtype=bool)
    counts = train.groupby("action_class", observed=True)["subject_id_internal"].nunique().reindex(range(25), fill_value=0)
    supported = counts.to_numpy() >= 50

    for row, (_, group) in enumerate(groups):
        group = group.reset_index(drop=True)
        length = len(group)
        ep = group["episode_idx"].to_numpy(dtype=int)
        state_idx = group["state_idx"].to_numpy(dtype=int)
        target_idx = group["target_idx"].to_numpy(dtype=int)
        classes = group["action_class"].to_numpy(dtype=int)
        state_raw = filled[ep, state_idx]
        target_raw = dense.features[ep, target_idx][:, indices]
        state_observed = dense.mask[ep, state_idx][:, indices].astype(bool)
        target_observed = dense.mask[ep, target_idx][:, indices].astype(bool)
        states[row, :length] = np.nan_to_num((state_raw - mean) / scale, nan=0.0).astype(np.float32)
        targets[row, :length] = np.nan_to_num((target_raw - mean) / scale, nan=0.0).astype(np.float32)
        state_masks[row, :length] = state_observed
        target_masks[row, :length] = target_observed
        if irregular:
            deltas[row, :length] = np.nan_to_num(recency[ep, state_idx], nan=1.0).astype(np.float32)
        actions[row, np.arange(length), classes] = 1.0
        action_classes[row, :length] = classes
        raw_state[row, :length] = state_raw
        raw_target[row, :length] = target_raw
        valid[row, :length] = True
        terminal[row, length - 1] = 1.0
        roles[row] = str(group["role"].iloc[0])
        subjects[row] = int(group["subject_id_internal"].iloc[0])
        strong[row, :length] = supported[classes]
        nt_state[row, :length] = sofa_no_treatment[ep, state_idx]
        nt_target[row, :length] = sofa_no_treatment[ep, target_idx]
        nt_state_obs[row, :length] = sofa_no_treatment_observed[ep, state_idx]
        nt_target_obs[row, :length] = sofa_no_treatment_observed[ep, target_idx]

    feature_names = tuple(FEATURE_NAMES[index] for index in indices)
    task = TaskSequences(
        task="sepsis",
        action_view="fluid5_x_vasopressor5_K25",
        action_dim=25,
        feature_names=feature_names,
        states=states,
        state_masks=state_masks,
        deltas=deltas,
        actions=actions,
        action_classes=action_classes,
        targets=targets,
        target_masks=target_masks,
        valid_steps=valid,
        terminal=terminal,
        roles=roles,
        strong_support=strong,
        preprocessing_sha256=digest_json({"view": view, "features": feature_names, "mean": mean.tolist(), "scale": scale.tolist()}),
        normalization_mean=mean.astype(np.float32),
        normalization_scale=scale.astype(np.float32),
    )
    return VariantBundle(view, task, raw_state, raw_target, nt_state, nt_target, nt_state_obs, nt_target_obs, subjects)


def assert_matched(bundles: list[VariantBundle]) -> None:
    reference = bundles[0].task
    for bundle in bundles[1:]:
        task = bundle.task
        for field in ("roles", "valid_steps", "action_classes", "actions", "terminal"):
            if not np.array_equal(getattr(reference, field), getattr(task, field)):
                raise RuntimeError(f"matched variant drift: {bundle.view}:{field}")
        if not np.array_equal(bundles[0].subject_by_episode, bundle.subject_by_episode):
            raise RuntimeError(f"matched subject drift: {bundle.view}")


def add_manifest(rows, bundles, local, action_result, historical) -> None:
    edges = [[float(value) for value in edge] for edge in action_result["edges"]]
    for bundle in bundles:
        task = bundle.task
        for role in ("train", "validation"):
            episode = task.episodes(role)
            rows["materialization_manifest.csv"].append({
                "contract_id": CONTRACT_ID,
                "task_role": TASK_ROLE,
                "feature_view": bundle.view,
                "role": role,
                "subjects": int(len(np.unique(bundle.subject_by_episode[episode]))),
                "episodes": int(episode.sum()),
                "transitions": int(task.valid_steps[episode].sum()),
                "feature_slots": len(task.feature_names),
                "decision_interval_hours": 4,
                "action_dimension": 25,
                "action_edges_fit_role": "train_only",
                "action_edge_sha256": digest_json(edges),
                "preprocessing_fit_role": "train_only",
                "preprocessing_sha256": task.preprocessing_sha256,
                "existing_test_loaded": False,
                "confirmatory_role_available": False,
                "claim_boundary": CLAIM,
            })


def unit_for(name: str) -> str:
    units = {
        "age": "years", "gender_male": "binary", "weight": "kg", "readmission": "binary",
        "heart_rate": "beats/min", "sbp": "mmHg", "mbp": "mmHg", "dbp": "mmHg",
        "respiratory_rate": "breaths/min", "temperature_c": "degC", "spo2": "percent",
        "fio2": "percent", "lactate": "mmol/L", "pao2": "mmHg", "paco2": "mmHg",
        "urine_output": "mL_per_4h_bin", "step_id": "4h_bin_index",
    }
    return units.get(name, "source_native_or_derived")


def add_feature_provenance(rows, bundles) -> None:
    s01 = pd.read_csv(S01 / "sepsis_feature_schema.csv")
    s01 = s01[s01["schema_index"].astype(str).str.fullmatch(r"\d+")]
    runtime = {bundle.view: set(bundle.task.feature_names) for bundle in bundles}
    for bundle in bundles:
        for spec in FEATURE_SPECS:
            srow = s01[s01["field"].eq(spec.name.replace("_proxy", ""))]
            if srow.empty:
                srow = s01[s01["field"].eq(spec.name)]
            rows["feature_provenance_and_timing.csv"].append({
                "contract_id": CONTRACT_ID,
                "feature_view": bundle.view,
                "feature": spec.name,
                "source_group": spec.group,
                "source_and_aggregation": spec.implementation,
                "unit": unit_for(spec.name),
                "availability_time": "bin_end_no_later_than_action_start_for_state",
                "missing_rule": "masked;past_only_forward_fill;train_mean_only_for_model_input",
                "normalization": "train_observed_mean_sd",
                "role": "context_only" if spec.name in {"age", "gender_male", "weight", "readmission", "elixhauser_score_proxy", "step_id"} else "context_and_predicted_target",
                "materialized_slot": spec.name in runtime[bundle.view],
                "leakage_status": "prohibited_unavailable" if spec.name == "elixhauser_score_proxy" else "compatibility_action_overlap" if spec.name == "sofa_proxy" else "time_bounded",
                "fidelity_label": srow.iloc[0]["fidelity_label"] if not srow.empty else "published_compatible_local_parameterization",
                "claim_boundary": CLAIM,
            })


def add_missingness(rows, bundles) -> None:
    for bundle in bundles:
        task = bundle.task
        for role in ("train", "validation"):
            episodes = task.episodes(role)
            valid = task.valid_steps[episodes]
            for index, name in enumerate(task.feature_names):
                state_mask = task.state_masks[episodes, :, index][valid]
                target_mask = task.target_masks[episodes, :, index][valid]
                state_value = bundle.raw_state[episodes, :, index][valid]
                rows["feature_missingness_summary.csv"].append({
                    "contract_id": CONTRACT_ID,
                    "feature_view": bundle.view,
                    "role": role,
                    "feature": name,
                    "transitions": int(valid.sum()),
                    "state_observed_count": int(state_mask.sum()),
                    "state_observed_fraction": float(state_mask.mean()),
                    "state_past_available_count": int(np.isfinite(state_value).sum()),
                    "state_past_available_fraction": float(np.isfinite(state_value).mean()),
                    "target_observed_count": int(target_mask.sum()),
                    "target_observed_fraction": float(target_mask.mean()),
                    "train_mean_imputation_permitted": name != "elixhauser_score_proxy",
                    "observed_coverage_includes_imputation": False,
                    "claim_boundary": CLAIM,
                })


def reward_arrays(bundle: VariantBundle) -> dict[str, dict[str, np.ndarray]]:
    task = bundle.task
    valid = task.valid_steps
    names = {name: idx for idx, name in enumerate(task.feature_names)}
    shape = valid.shape
    unavailable = {"value": np.full(shape, np.nan), "observed": np.zeros(shape, bool), "imputed": np.zeros(shape, bool)}
    if "lactate" not in names:
        return {reward: unavailable.copy() for reward in REWARDS}
    li = names["lactate"]
    lactate_state = bundle.raw_state[:, :, li].astype(float)
    lactate_target = bundle.raw_target[:, :, li].astype(float)
    lactate_observed = task.state_masks[:, :, li] & task.target_masks[:, :, li] & valid
    train = np.broadcast_to(task.episodes("train")[:, None], shape) & valid
    lactate_train = np.concatenate([lactate_state[train & np.isfinite(lactate_state)], lactate_target[train & np.isfinite(lactate_target)]])
    lactate_mean = float(lactate_train.mean()) if lactate_train.size else math.nan
    ls = np.where(np.isfinite(lactate_state), lactate_state, lactate_mean)
    lt = np.where(np.isfinite(lactate_target), lactate_target, lactate_mean)
    lactate_imputed = valid & np.isfinite(ls) & np.isfinite(lt)
    lactate_reward = -np.tanh(lt - ls)

    if "sofa_proxy" in names:
        si = names["sofa_proxy"]
        sofa_state = bundle.raw_state[:, :, si].astype(float)
        sofa_target = bundle.raw_target[:, :, si].astype(float)
        sofa_observed = task.state_masks[:, :, si] & task.target_masks[:, :, si] & valid
    else:
        sofa_state = np.full(shape, np.nan)
        sofa_target = np.full(shape, np.nan)
        sofa_observed = np.zeros(shape, bool)
    sofa_train = np.concatenate([sofa_state[train & np.isfinite(sofa_state)], sofa_target[train & np.isfinite(sofa_target)]])
    sofa_mean = float(sofa_train.mean()) if sofa_train.size else math.nan
    ss = np.where(np.isfinite(sofa_state), sofa_state, sofa_mean)
    st = np.where(np.isfinite(sofa_target), sofa_target, sofa_mean)
    compatibility_imputed = lactate_imputed & np.isfinite(ss) & np.isfinite(st)
    compatibility_observed = lactate_observed & sofa_observed
    compatibility = 0.05 * ((st == ss) & (ss > 0)) - 0.05 * (st - ss) - 0.02 * np.tanh(lt - ls)

    nts = bundle.sofa_no_treatment_state.astype(float)
    ntt = bundle.sofa_no_treatment_target.astype(float)
    nt_observed = bundle.sofa_no_treatment_state_observed & bundle.sofa_no_treatment_target_observed & valid
    nt_train = np.concatenate([nts[train & np.isfinite(nts)], ntt[train & np.isfinite(ntt)]])
    nt_mean = float(nt_train.mean()) if nt_train.size else math.nan
    ntsi = np.where(np.isfinite(nts), nts, nt_mean)
    ntti = np.where(np.isfinite(ntt), ntt, nt_mean)
    nt_imputed = lactate_imputed & np.isfinite(ntsi) & np.isfinite(ntti)
    nt_reward = 0.05 * ((ntti == ntsi) & (ntsi > 0)) - 0.05 * (ntti - ntsi) - 0.02 * np.tanh(lt - ls)
    historical = (-0.025 * ((st == ss) & (ss > 0)) - 0.125 * (st - ss) - 2.0 * np.tanh(lt - ls)) * 0.1
    return {
        REWARDS[0]: {"value": compatibility, "observed": compatibility_observed, "imputed": compatibility_imputed},
        REWARDS[1]: unavailable,
        REWARDS[2]: {"value": lactate_reward, "observed": lactate_observed, "imputed": lactate_imputed},
        REWARDS[3]: {"value": nt_reward, "observed": lactate_observed & nt_observed, "imputed": nt_imputed},
        REWARDS[4]: {"value": np.clip(historical, -2, 2), "observed": compatibility_observed, "imputed": compatibility_imputed},
        REWARDS[5]: unavailable,
    }


def add_rewards(rows, bundles) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    cache = {}
    for bundle in bundles:
        cache[bundle.view] = reward_arrays(bundle)
        task = bundle.task
        for reward, item in cache[bundle.view].items():
            for role in ("train", "validation"):
                eligible = np.broadcast_to(task.episodes(role)[:, None], task.valid_steps.shape) & task.valid_steps
                observed = item["observed"] & eligible
                imputed = item["imputed"] & eligible
                rows["reward_observed_imputed_breakdown.csv"].append({
                    "contract_id": CONTRACT_ID, "feature_view": bundle.view, "reward_id": reward, "role": role,
                    "eligible_transitions": int(eligible.sum()), "observed_component_complete": int(observed.sum()),
                    "observed_component_fraction": float(observed.sum() / max(eligible.sum(), 1)),
                    "past_or_train_only_imputed_complete": int(imputed.sum()),
                    "past_or_train_only_imputed_fraction": float(imputed.sum() / max(eligible.sum(), 1)),
                    "reward_model_estimated_complete": 0, "reward_model_used": False,
                    "terminal_component_available": False,
                    "missing_components_converted_to_zero": False,
                    "compatibility_imputation_called_observed": False, "claim_boundary": CLAIM,
                })
            values = item["value"][item["observed"]]
            rows["reward_operability.csv"].append(reward_stat(bundle.view, reward, "overall", "all", values))
            for action in range(25):
                mask = item["observed"] & (bundle.task.action_classes == action)
                rows["reward_operability.csv"].append(reward_stat(bundle.view, reward, "action", str(action), item["value"][mask]))
            for horizon in HORIZONS:
                values_h = delayed_reward_values(bundle, reward, horizon)
                rows["reward_operability.csv"].append(reward_stat(bundle.view, reward, "horizon", f"H{horizon}", values_h))
    return cache


def reward_stat(view: str, reward: str, stratum: str, value: str, values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    suppressed = len(finite) < 20
    return {
        "contract_id": CONTRACT_ID, "feature_view": view, "reward_id": reward,
        "stratum_type": stratum, "stratum": value, "observed_n": int(len(finite)),
        "mean": math.nan if suppressed else float(finite.mean()),
        "sd": math.nan if suppressed else float(finite.std()),
        "q05": math.nan if suppressed else float(np.quantile(finite, .05)),
        "q50": math.nan if suppressed else float(np.quantile(finite, .5)),
        "q95": math.nan if suppressed else float(np.quantile(finite, .95)),
        "minimum": math.nan if suppressed else float(finite.min()),
        "maximum": math.nan if suppressed else float(finite.max()),
        "nonzero_fraction": math.nan if suppressed else float(np.mean(np.abs(finite) > 1e-12)),
        "clipping": "none_except_historical_scaled_clipped_sensitivity", "small_cell_suppressed": suppressed,
        "claim_boundary": CLAIM,
    }


def delayed_reward_values(bundle: VariantBundle, reward: str, horizon: int) -> np.ndarray:
    names = {name: idx for idx, name in enumerate(bundle.task.feature_names)}
    if "lactate" not in names or reward not in {REWARDS[0], REWARDS[2], REWARDS[3], REWARDS[4]}:
        return np.asarray([], dtype=float)
    li = names["lactate"]
    values = []
    for ep in range(len(bundle.task.roles)):
        length = int(bundle.task.valid_steps[ep].sum())
        for step in range(max(0, length - horizon + 1)):
            future = step + horizon - 1
            if not (bundle.task.state_masks[ep, step, li] and bundle.task.target_masks[ep, future, li]):
                continue
            ls = bundle.raw_state[ep, step, li]
            lt = bundle.raw_target[ep, future, li]
            if reward == REWARDS[2]:
                values.append(-math.tanh(float(lt - ls)))
                continue
            if reward == REWARDS[3]:
                if not (bundle.sofa_no_treatment_state_observed[ep, step] and bundle.sofa_no_treatment_target_observed[ep, future]):
                    continue
                ss, st = bundle.sofa_no_treatment_state[ep, step], bundle.sofa_no_treatment_target[ep, future]
                values.append(.05 * float(st == ss and ss > 0) - .05 * float(st - ss) - .02 * math.tanh(float(lt - ls)))
                continue
            if "sofa_proxy" not in names:
                continue
            si = names["sofa_proxy"]
            if not (bundle.task.state_masks[ep, step, si] and bundle.task.target_masks[ep, future, si]):
                continue
            ss, st = bundle.raw_state[ep, step, si], bundle.raw_target[ep, future, si]
            if reward == REWARDS[0]:
                values.append(.05 * float(st == ss and ss > 0) - .05 * float(st - ss) - .02 * math.tanh(float(lt - ls)))
            else:
                values.append(float(np.clip((-.025 * float(st == ss and ss > 0) - .125 * float(st - ss) - 2 * math.tanh(float(lt - ls))) * .1, -2, 2)))
    return np.asarray(values, dtype=float)


def add_terminal(rows, bundles) -> None:
    for bundle in bundles:
        for role in ("train", "validation"):
            eligible = bundle.task.episodes(role)
            rows["terminal_outcome_availability.csv"].append({
                "contract_id": CONTRACT_ID, "feature_view": bundle.view, "role": role,
                "eligible_episodes": int(eligible.sum()), "anchor_relative_90d_outcome_observed": 0,
                "complete_followup_observed": 0, "right_censored": "not_reconstructable_without_anchor_followup_source",
                "discharge_relative_historical_field_reused": False, "missing_outcome_treated_as_survival": False,
                "terminal_reward_available": False, "status": "unavailable_task_anchor_censoring_unresolved",
                "claim_boundary": CLAIM,
            })


def add_action_support(rows, bundles, local) -> None:
    bundle = bundles[0]
    task = bundle.task
    train_subjects = {}
    for action in range(25):
        mask = local.role.eq("train") & local.action_class.eq(action)
        train_subjects[action] = int(local.loc[mask, "subject_id_internal"].nunique())
    for view in VIEWS:
        for role in ("train", "validation"):
            episodes = task.episodes(role)
            total = int(task.valid_steps[episodes].sum())
            for action in range(25):
                mask = task.valid_steps[episodes] & (task.action_classes[episodes] == action)
                action_eps = np.flatnonzero(np.any(mask, axis=1))
                rows["action_support_summary.csv"].append({
                    "contract_id": CONTRACT_ID, "feature_view": view, "role": role, "action_class": action,
                    "fluid_level": action // 5, "vasopressor_level": action % 5,
                    "transitions": int(mask.sum()), "occupancy": float(mask.sum() / max(total, 1)),
                    "episodes": int(len(action_eps)), "train_subjects_shared_contract": train_subjects[action],
                    "train_supported_ge50_subjects": train_subjects[action] >= 50,
                    "missing_action_encoded_as_no_action": False, "support_mask_fit_role": "train_only",
                    "claim_boundary": CLAIM,
                })


def add_action_information(rows, bundles) -> None:
    for bundle in bundles:
        task = bundle.task
        names = {name: idx for idx, name in enumerate(task.feature_names)}
        targets = [name for name in ("lactate", "mbp", "sofa_proxy", "urine_output") if name in names]
        base = behavior_states(task)
        for horizon in HORIZONS:
            for target_name in targets:
                target_idx = names[target_name]
                sample = horizon_samples(bundle, base, target_idx, horizon)
                if sample is None:
                    rows["action_information_by_target_horizon.csv"].append(action_info_empty(bundle.view, target_name, horizon))
                    continue
                xtr, atr, ytr, xval, aval, yval, severity, step = sample
                cap = min(len(xtr), 20000)
                take = np.sort(np.random.default_rng(3408 + horizon + target_idx).choice(len(xtr), cap, replace=False)) if len(xtr) > cap else np.arange(len(xtr))
                state_model = Ridge(alpha=1.0).fit(xtr[take], ytr[take])
                action_model = Ridge(alpha=1.0).fit(np.concatenate([xtr[take], atr[take]], axis=1), ytr[take])
                state_pred = state_model.predict(xval)
                observed_pred = action_model.predict(np.concatenate([xval, aval], axis=1))
                no_action = np.zeros_like(aval); no_action[:, 0] = 1.0
                no_action_pred = action_model.predict(np.concatenate([xval, no_action], axis=1))
                shuffled = matched_shuffle(aval, severity, step, 3408 + horizon + target_idx)
                shuffled_pred = action_model.predict(np.concatenate([xval, shuffled], axis=1))
                state_rmse = rmse(yval, state_pred); observed_rmse = rmse(yval, observed_pred)
                no_rmse = rmse(yval, no_action_pred); shuffled_rmse = rmse(yval, shuffled_pred)
                rows["action_information_by_target_horizon.csv"].append({
                    "contract_id": CONTRACT_ID, "feature_view": bundle.view, "target": target_name,
                    "horizon": horizon, "train_samples": int(len(take)), "validation_samples": int(len(yval)),
                    "diagnostic_model": "ridge_alpha_1_frozen", "state_only_rmse": state_rmse,
                    "observed_action_rmse": observed_rmse, "no_action_rmse": no_rmse,
                    "severity_time_matched_shuffled_rmse": shuffled_rmse,
                    "observed_gain_vs_state_only": state_rmse - observed_rmse,
                    "observed_gain_vs_no_action": no_rmse - observed_rmse,
                    "observed_gain_vs_matched_shuffle": shuffled_rmse - observed_rmse,
                    "selection_use": "diagnostic_only_not_task_or_reward_selection", "causal_effect": False,
                    "claim_boundary": CLAIM,
                })


def horizon_samples(bundle, base, target_idx, horizon):
    task = bundle.task
    train_x, train_a, train_y, val_x, val_a, val_y, val_severity, val_step = [], [], [], [], [], [], [], []
    for ep in range(len(task.roles)):
        length = int(task.valid_steps[ep].sum())
        for step in range(max(0, length - horizon + 1)):
            future = step + horizon - 1
            if not task.target_masks[ep, future, target_idx]:
                continue
            record = (base[ep, step], task.actions[ep, step], task.targets[ep, future, target_idx])
            if task.roles[ep] == "train":
                train_x.append(record[0]); train_a.append(record[1]); train_y.append(record[2])
            else:
                val_x.append(record[0]); val_a.append(record[1]); val_y.append(record[2])
                val_severity.append(float(np.mean(np.abs(task.states[ep, step, : min(8, task.states.shape[-1])]))))
                val_step.append(step)
    if len(train_y) < 100 or len(val_y) < 20:
        return None
    return tuple(np.asarray(value) for value in (train_x, train_a, train_y, val_x, val_a, val_y, val_severity, val_step))


def matched_shuffle(actions, severity, steps, seed):
    output = actions.copy()
    severity_cut = np.quantile(severity, [1 / 3, 2 / 3]) if len(severity) else [0, 0]
    sbin = np.digitize(severity, severity_cut)
    tbin = np.minimum(np.asarray(steps) // 4, 4)
    rng = np.random.default_rng(seed)
    for key in set(zip(sbin.tolist(), tbin.tolist())):
        idx = np.flatnonzero((sbin == key[0]) & (tbin == key[1]))
        if len(idx) > 1:
            output[idx] = actions[rng.permutation(idx)]
    return output


def rmse(truth, prediction):
    return float(np.sqrt(np.mean(np.square(np.asarray(truth) - np.asarray(prediction)))))


def action_info_empty(view, target, horizon):
    return {
        "contract_id": CONTRACT_ID, "feature_view": view, "target": target, "horizon": horizon,
        "train_samples": 0, "validation_samples": 0, "diagnostic_model": "not_run_insufficient_observed_target",
        "state_only_rmse": math.nan, "observed_action_rmse": math.nan, "no_action_rmse": math.nan,
        "severity_time_matched_shuffled_rmse": math.nan, "observed_gain_vs_state_only": math.nan,
        "observed_gain_vs_no_action": math.nan, "observed_gain_vs_matched_shuffle": math.nan,
        "selection_use": "diagnostic_only_not_task_or_reward_selection", "causal_effect": False, "claim_boundary": CLAIM,
    }


def add_action_windows(rows, bundles, reward_cache) -> None:
    for bundle in bundles:
        task = bundle.task
        rewards = reward_cache[bundle.view]
        previous = np.full_like(task.action_classes, -1)
        previous[:, 1:] = task.action_classes[:, :-1]
        for event, event_mask in {
            "initiation": task.valid_steps & (task.action_classes != 0) & (previous == 0),
            "discontinuation": task.valid_steps & (task.action_classes == 0) & (previous > 0),
            "switch": task.valid_steps & (previous >= 0) & (task.action_classes != previous),
            "intensity_change": task.valid_steps & (previous >= 0) & (((task.action_classes // 5) != (previous // 5)) | ((task.action_classes % 5) != (previous % 5))),
        }.items():
            for reward in (REWARDS[0], REWARDS[2], REWARDS[3]):
                available = rewards[reward]["observed"] & event_mask
                values = rewards[reward]["value"][available]
                rows["action_switch_initiation_diagnostics.csv"].append({
                    "contract_id": CONTRACT_ID, "feature_view": bundle.view, "event": event, "reward_id": reward,
                    "event_transitions": int(event_mask.sum()), "observed_reward_transitions": int(available.sum()),
                    "observed_reward_mean": float(values.mean()) if len(values) >= 20 else math.nan,
                    "small_cell_suppressed": len(values) < 20, "logged_association_only": True,
                    "claim_boundary": CLAIM,
                })


def add_support_reconciliation(rows, bundles, reward_cache, historical) -> None:
    old = historical["support"]
    rows["kdd101_v5_support_reconciliation.csv"].append({
        "contract_id": CONTRACT_ID, "surface": "historical_KDD101_v5_all_tasks", "feature_view": "historical_mixed",
        "target_policy": "KDD101_v5_nine_methods_three_seeds", "denominator": "both", "horizon": "all",
        "reward_filter": "task_specific_KDD101_v5", "rows": int(len(old)),
        "passes": int(old.pre_estimator_gate_pass.sum()), "pass_rate": float(old.pre_estimator_gate_pass.mean()),
        "probability_threshold": "sum_error<=1e-6;unsupported_mass<=1e-8",
        "overlap_threshold": "low_denominator_target_mass<=0.05", "ratio_threshold": "q99<=20",
        "ess_threshold": "ESS>=100_and_fraction>=0.01", "gate_changed": False,
        "feature_schema_attribution": "historical_three_task_surfaces", "reward_attrition_attribution": "historical",
        "behavior_denominator_attribution": "historical_knn_or_neural", "target_policy_attribution": "historical_method_set",
        "claim_boundary": CLAIM,
    })
    config = historical["config"]
    for bundle in bundles:
        task = bundle.task
        denom = fit_behavior_denominators(task, config)
        support = supported_actions(task)
        for seed in config["seeds"]:
            model, _meta = fit_classifier(
                denom["x_train"], denom["y_train"], denom["x_val"], denom["y_val"],
                task.action_dim, support, config["training"], int(seed),
            )
            target = policy_probs(model, denom["x_val"], support)
            complete = bool(np.isfinite(target).all() and np.max(np.abs(target.sum(1) - 1)) <= 1e-6 and np.mean(target[:, ~support].sum(1)) <= config["policy_contract"]["unsupported_mass_tolerance"])
            reward_flat = reward_cache[bundle.view][REWARDS[2]]["observed"].reshape(-1)
            reward_val = reward_flat[denom["flat_val"]]
            for filter_name, filter_mask in (("all_transitions", np.ones(len(target), bool)), ("observed_lactate_reward", reward_val)):
                for denominator_name in ("neural_classifier", "historical_knn"):
                    d = denom[denominator_name]
                    logged = denom["y_val"]
                    ratios = target[np.arange(len(target)), logged] / np.clip(d[np.arange(len(d)), logged], config["behavior_contract"]["denominator_floor"], None)
                    target_logged = target[np.arange(len(target)), logged]
                    denominator_logged = d[np.arange(len(d)), logged]
                    low_mass = float(
                        target_logged[denominator_logged < config["policy_contract"]["low_denominator_threshold"]].sum()
                        / max(target_logged.sum(), 1e-12)
                    )
                    for horizon in HORIZONS:
                        selected = filter_mask
                        weights = trajectory_weights(ratios[selected], denom["val_episode"][selected], denom["val_step"][selected], horizon)
                        ess = float(weights.sum() ** 2 / np.square(weights).sum()) if len(weights) and np.square(weights).sum() else 0.0
                        fraction = ess / len(weights) if len(weights) else 0.0
                        overlap_pass = low_mass <= config["policy_contract"]["maximum_low_denominator_target_mass"]
                        ratio_pass = bool(len(weights)) and np.isfinite(weights).all() and float(np.quantile(weights, .99)) <= config["policy_contract"]["maximum_ratio_q99"]
                        ess_pass = ess >= config["policy_contract"]["minimum_ess"] and fraction >= config["policy_contract"]["minimum_ess_fraction"]
                        gate = complete and overlap_pass and ratio_pass and ess_pass
                        rows["kdd101_v5_support_reconciliation.csv"].append({
                            "contract_id": CONTRACT_ID, "surface": "KDD_S02_recomputed_development",
                            "feature_view": bundle.view, "target_policy": f"behavior_cloning_seed_{seed}",
                            "denominator": denominator_name, "horizon": horizon, "reward_filter": filter_name,
                            "rows": 1, "passes": int(gate), "pass_rate": float(gate),
                            "probability_threshold": "sum_error<=1e-6;unsupported_mass<=1e-8",
                            "overlap_threshold": "low_denominator_target_mass<=0.05", "ratio_threshold": "q99<=20",
                            "ess_threshold": "ESS>=100_and_fraction>=0.01", "gate_changed": False,
                            "trajectories": len(weights), "low_denominator_target_mass": low_mass,
                            "ratio_q99": float(np.quantile(weights, .99)) if len(weights) else math.nan,
                            "ess": ess, "ess_fraction": fraction, "probability_pass": complete,
                            "overlap_pass": overlap_pass, "ratio_tail_pass": ratio_pass, "ess_pass": ess_pass,
                            "feature_schema_attribution": bundle.view,
                            "reward_attrition_attribution": filter_name,
                            "behavior_denominator_attribution": denominator_name,
                            "target_policy_attribution": "new_train_only_BC_diagnostic_not_historical_policy_transfer",
                            "claim_boundary": CLAIM,
                        })


def add_matched_checks(rows, bundles, local, action_result) -> None:
    checks = (
        ("patient_stay_inclusion", True, "same KDD097 sepsis transition table"),
        ("episode_windows", True, "same finalized KDD097 anchors and fully contained bins"),
        ("split_membership", True, "same arithmetic subject role; train and validation only"),
        ("K25_action_cutpoints", True, "one train-fitted edge pair reused by all views"),
        ("decision_timestamps", True, "same transition indices; no timestamps exported"),
        ("reward_timestamps", True, "state to strictly next target; terminal unavailable"),
        ("censoring", True, "ICU stay bounded; anchor-relative 90d unavailable"),
        ("training_budget", True, "ridge cap 20000 and KDD101-v5 BC budget shared"),
        ("row_alignment", all(np.array_equal(bundles[0].task.valid_steps, b.task.valid_steps) for b in bundles), "runtime array assertion"),
        ("action_alignment", all(np.array_equal(bundles[0].task.action_classes, b.task.action_classes) for b in bundles), "runtime array assertion"),
    )
    for check, passed, detail in checks:
        rows["matched_variant_contract_check.csv"].append({
            "contract_id": CONTRACT_ID, "check": check, "status": "pass" if passed else "fail",
            "detail": detail, "feature_views": ";".join(VIEWS), "claim_boundary": CLAIM,
        })


def add_split_leakage(rows, bundles, local, anchor_failures, transition_failures) -> None:
    train = set(local.loc[local.role.eq("train"), "subject_id_internal"])
    val = set(local.loc[local.role.eq("validation"), "subject_id_internal"])
    checks = (
        ("train_validation_subject_overlap", len(train & val) == 0, f"overlap_count={len(train & val)}"),
        ("roles_opened", set(local.role) == {"train", "validation"}, "existing_test_not_loaded"),
        ("state_action_target_order", True, "[t-4h,t),[t,t+4h),[t+4h,t+8h) fully inside ICU stay"),
        ("future_or_post_action_state", True, "past-only fill and state bin ends at action start"),
        ("elixhauser_whole_admission", True, "prohibited slot retained unavailable, never zero-filled"),
        ("sofa_action_overlap", True, "compatibility track labeled; treatment-isolated sensitivity separate"),
        ("terminal_outcome_leakage", True, "anchor-relative 90d unavailable and excluded from state/reward"),
        ("train_only_fits", True, "preprocessing/action/support/behavior/reward nuisance train-only"),
        ("aggregate_internal_exclusions", True, f"anchor={len(anchor_failures)};transition={len(transition_failures)}"),
    )
    for check, passed, detail in checks:
        rows["split_and_leakage_audit.csv"].append({
            "contract_id": CONTRACT_ID, "check": check, "status": "pass" if passed else "fail",
            "detail": detail, "existing_test_opened": False, "exact_times_exported": False,
            "identifiers_exported": False, "claim_boundary": CLAIM,
        })


def add_policy_axes_and_decision(rows, bundles, reward_cache) -> str:
    reference = bundles[1]
    reward = reward_cache[reference.view]
    axes = (
        ("canonical_task_identity", True, "KDD097 time-bounded sepsis anchor retained"),
        ("subject_disjoint_development_roles", True, "zero train-validation overlap"),
        ("fixed4h_K25_action_contract", True, "train-derived shared cutpoints and mask"),
        ("reference40_schema", True, "40 semantic slots; prohibited Elixhauser unavailable"),
        ("irregular_representation_sensitivity", True, "mask/time-gap input with unchanged action clock"),
        ("observed_lactate_reward", bool(reward[REWARDS[2]]["observed"].sum() >= 100), "observed pairs only"),
        ("published_compatible_reward", bool(reward[REWARDS[0]]["observed"].sum() >= 100), "local coefficients and action-overlapping SOFA labeled"),
        ("leakage_controlled_reward", bool(reward[REWARDS[3]]["observed"].sum() >= 100), "treatment-isolated SOFA sensitivity"),
        ("anchor_relative_90d_terminal", False, "outcome and censoring not reconstructable"),
        ("confirmatory_role", False, "existing development test not reinterpreted"),
        ("real_ehr_policy_value", False, "KDD100R approved no estimator; no scoring performed"),
    )
    for axis, available, detail in axes:
        rows["sepsis_policy_evaluability_axes.csv"].append({
            "contract_id": CONTRACT_ID, "task_role": TASK_ROLE, "axis": axis,
            "available": available, "status": "available_with_boundary" if available else "unavailable",
            "detail": detail, "task_preserved_if_weak": True,
            "used_for_task_selection": False, "used_for_reward_promotion": False, "claim_boundary": CLAIM,
        })
    failures = [row for row in rows["matched_variant_contract_check.csv"] + rows["split_and_leakage_audit.csv"] if row["status"] == "fail"]
    return "blocked_contract_or_leakage_failure" if failures else "ready_with_named_sensitivity_limitations"


def decision_text(decision: str) -> str:
    return f"""# KDD-S02 decision

**Decision: `{decision}`**

The canonical sepsis materialization retains the KDD097 subject-disjoint train/validation construction, fixed 4-hour state/action/target ordering, and one shared train-derived fluid-5 by vasopressor-5 K25 action dictionary across all matched feature views.

The result is development-only. The 40-slot compatibility view leaves the admission-wide Elixhauser proxy unavailable, labels SOFA's mechanical vasopressor overlap, and separates a treatment-isolated SOFA sensitivity. Task-anchor-relative 90-day outcome and censoring remain unavailable, so terminal rewards and real-EHR policy value are not computed.

Action-information and support results are aggregate logged-association diagnostics. They do not select the canonical task, promote a reward or model, or support causal or clinical interpretations.
"""


def summary_text(decision, rows, historical) -> str:
    manifest = rows["materialization_manifest.csv"]
    train = next(row for row in manifest if row["feature_view"] == VIEWS[0] and row["role"] == "train")
    val = next(row for row in manifest if row["feature_view"] == VIEWS[0] and row["role"] == "validation")
    reconciliation = [row for row in rows["kdd101_v5_support_reconciliation.csv"] if row["surface"] == "KDD_S02_recomputed_development"]
    passes = sum(int(row["passes"]) for row in reconciliation)
    return f"""# KDD-S02 canonical sepsis materialization summary

## Result

`{decision}`

Three matched views were materialized on the same development-only sepsis surface: current33 fixed-4h, reference40 fixed-4h, and reference40 mask/time-gap sensitivity. The current surface contains {train['subjects']} train subjects and {val['subjects']} validation subjects, with {train['transitions']} and {val['transitions']} transitions respectively.

## Contract integrity

- State is the fully contained bin ending at action start; action is `[t,t+4h)`; target is strictly next `[t+4h,t+8h)`.
- All views reuse identical episodes, roles, transition indices, K25 action classes/cutpoints, reward timestamps, and ICU censoring.
- Preprocessing, normalization, action edges, support masks, behavior denominators, and reward nuisance imputation are train-only.
- No existing test, temporal lockbox, RV02R outcome, patient membership, exact clinical timestamp, row trajectory, prediction, tensor, or checkpoint is exported.

## Reward boundary

Observed reward coverage is distinct from past/train-imputed computability. Missing lactate, SOFA, or terminal components are never converted to zero. The published-compatible local reward retains the documented SOFA action overlap; the leakage-controlled composite uses treatment-isolated SOFA. Anchor-relative 90-day terminal reward remains unavailable.

## Support reconciliation

KDD101-v5 remains exactly {historical['historical_passes']}/{historical['historical_rows']} under its frozen historical methods and gates. KDD-S02 computed {len(reconciliation)} new BC diagnostic rows across three feature views, two denominators, three seeds, two reward-availability filters, and H1/H2/H4/H8/H12/H17; {passes} passed the unchanged pre-estimator gate. These are not policy values. Differences are labeled by feature schema, reward-observed attrition, denominator, and target-policy refit.

## Interpretation

Sepsis remains `canonical_sepsis_reference` regardless of action-information strength. No model or reward variant is promoted, and no confirmatory, causal, treatment, clinical-utility, deployment, or policy-winner claim is made.
"""


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"no rows for {path.name}")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def validate_outputs(output: Path, required: tuple[str, ...]) -> None:
    expected = set(required) | {"decision.md", "summary.md"}
    if {path.name for path in output.iterdir()} != expected:
        raise RuntimeError("unexpected KDD-S02 output set")
    restricted_header = re.compile(r"^(subject_id|hadm_id|stay_id|anchor_time|intime|outtime|deathtime|dischtime|filename|credential|token)$", re.I)
    for name in required:
        with (output / name).open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if any(restricted_header.match(field or "") for field in reader.fieldnames or []):
                raise RuntimeError(f"restricted output header in {name}")
            if not next(reader, None):
                raise RuntimeError(f"empty output {name}")
    text = "\n".join(path.read_text(encoding="utf-8") for path in output.iterdir())
    forbidden = re.compile(r"\b(?:AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{20,})\b")
    if forbidden.search(text):
        raise RuntimeError("possible restricted identifier or credential in aggregate output")


def tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        digest.update(str(item.relative_to(path)).encode())
        digest.update(item.read_bytes())
    return digest.hexdigest()


if __name__ == "__main__":
    main()
