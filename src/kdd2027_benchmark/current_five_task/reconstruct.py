from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator

from .authoritative_semantics import (
    GCS_ITEMIDS,
    clean_feature_values,
    compute_corrected_derived_features,
    gcs_total_rows,
    overlap_bin_amounts,
    overlap_bins,
)
from .runtime_config import RUNTIME_CONFIG

from .contracts import (
    ANTIBIOTIC_PATTERN,
    BIN_HOURS,
    CHART_ITEM_MAP,
    CREATININE_ITEMIDS,
    DIURETIC_PATTERN,
    FEATURE_INDEX,
    FEATURE_NAMES,
    FIO2_ITEMIDS,
    FLUID_ITEMIDS,
    LAB_ITEM_MAP,
    MECHVENT_ITEMIDS,
    MBP_ITEMIDS,
    PEEP_ITEMIDS,
    POST_ANCHOR_HOURS,
    PRE_ANCHOR_HOURS,
    RAW_EXTRACTION_BINS,
    RELEASE,
    RRT_PATTERN,
    SBP_ITEMIDS,
    TASKS,
    URINE_ITEMIDS,
    VASO_ITEMIDS,
    VASO_PATTERN,
    VASODILATOR_PATTERN,
    ContractError,
    aggregate_receipt,
    apply_kdd201_temporal_repair,
    assert_unique,
    compact_lineage_role,
    corrected_chart_value,
    eligible_transition_indices,
    encode_five_levels,
    extraction_post_hours,
    fit_train_positive_edges,
    joint_codes,
    parse_times,
    reward_components,
    subject_role,
    validate_layout,
)


def _read(path: Path, columns: Iterable[str] | None = None, dates: Iterable[str] = ()) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=list(columns) if columns else None, low_memory=False)
    return parse_times(frame, dates, path.name)


def _normal_icd(series: pd.Series) -> pd.Series:
    return series.astype("string").str.upper().str.replace(".", "", regex=False)


def _mortality_90d(frame: pd.DataFrame) -> pd.Series:
    dod = pd.to_datetime(frame["dod"], errors="coerce")
    discharge = pd.to_datetime(frame["dischtime"], errors="coerce")
    in_hospital = frame["hospital_expire_flag"].fillna(0).astype(int).eq(1)
    after = dod.notna() & discharge.notna() & dod.le(discharge + pd.Timedelta(days=90))
    return (in_hospital | after).astype(np.int8)


def load_core(paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    patients = _read(paths["hosp/patients"], dates=("dod",))
    admissions = _read(paths["hosp/admissions"], dates=("admittime", "dischtime", "deathtime"))
    stays = _read(paths["icu/icustays"], dates=("intime", "outtime"))
    assert_unique(patients, ("subject_id",), "patients")
    assert_unique(admissions, ("hadm_id",), "admissions")
    assert_unique(stays, ("stay_id",), "icustays")
    base = (
        stays.merge(admissions, on=("subject_id", "hadm_id"), how="inner")
        .merge(patients[["subject_id", "gender", "anchor_age", "dod"]], on="subject_id", how="inner")
        .sort_values(["subject_id", "intime", "stay_id"], kind="stable")
    )
    if base["intime"].isna().any() or base["outtime"].isna().any() or base["outtime"].le(base["intime"]).any():
        raise ContractError("malformed ICU time order")
    discharge = base["discharge_location"].fillna("").astype(str).str.upper()
    base = base[
        base["anchor_age"].ge(18)
        & base["gender"].notna()
        & base["dischtime"].notna()
        & ~discharge.str.contains("HOSPICE")
    ].copy()
    base["mortality_90d"] = _mortality_90d(base)
    base["role"] = [subject_role(value) for value in base["subject_id"]]
    diagnoses = _read(paths["hosp/diagnoses_icd"])
    return base.reset_index(drop=True), diagnoses


def _first_anchor(frame: pd.DataFrame, source: str | None = None) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["stay_id", "anchor_time", "anchor_source"])
    if source is not None:
        frame = frame.copy(); frame["anchor_source"] = source
    if "anchor_source" not in frame:
        raise ContractError("anchor source is required")
    output = frame.sort_values(["stay_id", "anchor_time"], kind="stable").groupby("stay_id", observed=True).first().reset_index()
    return output[["stay_id", "anchor_time", "anchor_source"]]


def suspected_infection_anchors(paths: dict[str, Path], stays: pd.DataFrame) -> pd.DataFrame:
    prescriptions = _read(paths["hosp/prescriptions"], dates=("starttime", "stoptime"))
    antibiotics = prescriptions[
        prescriptions["hadm_id"].isin(stays["hadm_id"])
        & prescriptions["starttime"].notna()
        & prescriptions["drug"].astype(str).str.contains(ANTIBIOTIC_PATTERN, case=False, regex=True, na=False)
    ].copy()
    antibiotics = antibiotics.merge(stays[["subject_id", "hadm_id", "stay_id"]], on=["subject_id", "hadm_id"], how="inner")
    cultures = _read(paths["hosp/microbiologyevents"], dates=("chartdate", "charttime"))
    cultures = cultures[cultures["subject_id"].isin(stays["subject_id"])].copy()
    cultures["culture_time"] = cultures["charttime"].fillna(cultures["chartdate"])
    cultures = cultures[cultures["culture_time"].notna()].sort_values(["subject_id", "culture_time"], kind="stable")
    rows: list[dict[str, Any]] = []
    for antibiotic in antibiotics.sort_values(["subject_id", "stay_id", "starttime"], kind="stable").itertuples(index=False):
        local = cultures[cultures["subject_id"].eq(antibiotic.subject_id)]
        prior = local[(local["culture_time"] >= antibiotic.starttime - pd.Timedelta(hours=72)) & (local["culture_time"] < antibiotic.starttime)]
        future = local[(local["culture_time"] > antibiotic.starttime) & (local["culture_time"] <= antibiotic.starttime + pd.Timedelta(hours=24))]
        if not prior.empty:
            onset = prior.iloc[0]["culture_time"]
        elif not future.empty:
            onset = antibiotic.starttime
        else:
            continue
        rows.append({"stay_id": int(antibiotic.stay_id), "anchor_time": pd.Timestamp(onset)})
    return _first_anchor(pd.DataFrame(rows), "antibiotic_culture_suspected_infection_pending_organ_dysfunction")


def _interval_filter(events: pd.DataFrame, stays: pd.DataFrame, time: str) -> pd.DataFrame:
    merged = events.merge(stays[["stay_id", "intime", "outtime"]], on="stay_id", how="inner")
    return merged[(merged[time] >= merged["intime"]) & (merged[time] < merged["outtime"])].copy()


def _prior_disease_stays(stays: pd.DataFrame, diagnoses: pd.DataFrame, *, icd9_prefix: tuple[str, ...], icd10_prefix: tuple[str, ...]) -> pd.DataFrame:
    code = _normal_icd(diagnoses["icd_code"]); version = diagnoses["icd_version"].astype(int)
    selected = diagnoses.loc[
        (version.eq(9) & code.str.startswith(icd9_prefix, na=False))
        | (version.eq(10) & code.str.startswith(icd10_prefix, na=False)),
        ["subject_id", "hadm_id"],
    ].drop_duplicates()
    completed = selected.merge(stays[["subject_id", "hadm_id", "dischtime"]].drop_duplicates(), on=["subject_id", "hadm_id"], how="inner")
    first = completed.dropna(subset=["dischtime"]).groupby("subject_id", observed=True)["dischtime"].min()
    available = stays["subject_id"].map(first)
    return stays[available.notna() & available.lt(stays["admittime"])].copy()


def build_anchors(paths: dict[str, Path], stays: pd.DataFrame, diagnoses: pd.DataFrame) -> pd.DataFrame:
    chart = _read(paths["icu/chartevents"], dates=("charttime",))
    inputs = _read(paths["icu/inputevents"], dates=("starttime", "endtime"))
    procedures = _read(paths["icu/procedureevents"], dates=("starttime", "endtime"))
    items = _read(paths["icu/d_items"])[["itemid", "label"]]
    label = items.set_index("itemid")["label"]
    frames: list[pd.DataFrame] = []

    sepsis = suspected_infection_anchors(paths, stays)
    if not sepsis.empty:
        local = stays.merge(sepsis, on="stay_id", how="inner")
        local = local[(local["anchor_time"] >= local["intime"]) & (local["anchor_time"] < local["outtime"])]
        local["task_id"] = "sepsis"
        frames.append(local)

    respiratory = chart[chart["itemid"].isin(MECHVENT_ITEMIDS)].dropna(subset=["charttime"])[["stay_id", "charttime"]].rename(columns={"charttime": "anchor_time"}); respiratory["anchor_source"] = "structured_ventilation_chart_event"
    p = procedures.copy(); p["label"] = p["itemid"].map(label).fillna("")
    support = p[p["label"].astype(str).str.contains(r"oxygen|ventilat|intubat|bipap|cpap", case=False, regex=True)][["stay_id", "starttime"]].rename(columns={"starttime": "anchor_time"}); support["anchor_source"] = "structured_respiratory_support_event"
    respiratory = _first_anchor(pd.concat([respiratory, support], ignore_index=True))
    if not respiratory.empty:
        local = stays.merge(respiratory, on="stay_id", how="inner"); local["task_id"] = "respiratory_support"; frames.append(local)

    bp = chart[chart["itemid"].isin(SBP_ITEMIDS + MBP_ITEMIDS)].dropna(subset=["charttime"]).copy()
    bp["value"] = pd.to_numeric(bp["valuenum"], errors="coerce")
    bp = _interval_filter(bp, stays, "charttime")
    bp["relative_bin"] = np.floor((bp["charttime"] - bp["intime"]).dt.total_seconds() / (BIN_HOURS * 3600)).astype(int)
    bp = bp[((bp["itemid"].isin(SBP_ITEMIDS)) & bp["value"].lt(90)) | ((bp["itemid"].isin(MBP_ITEMIDS)) & bp["value"].lt(65))]
    bp = bp.groupby(["stay_id", "relative_bin"], observed=True).agg(event_count=("charttime", "size"), anchor_time=("charttime", "min")).reset_index()
    bp = bp[bp["event_count"].ge(2)][["stay_id", "anchor_time"]]; bp["anchor_source"] = "sustained_hypotension"
    vaso = inputs[inputs["itemid"].isin(VASO_ITEMIDS)].copy()
    positive = np.maximum(pd.to_numeric(vaso["amount"], errors="coerce").fillna(0), pd.to_numeric(vaso["rate"], errors="coerce").fillna(0)).gt(0)
    vaso = vaso[positive][["stay_id", "starttime"]].rename(columns={"starttime": "anchor_time"}); vaso["anchor_source"] = "vasopressor_support"
    shock = _first_anchor(pd.concat([bp, vaso], ignore_index=True))
    if not shock.empty:
        local = stays.merge(shock, on="stay_id", how="inner"); local["task_id"] = "shock"; frames.append(local)

    labs = _read(paths["hosp/labevents"], dates=("charttime",))
    creat = labs[labs["itemid"].isin(CREATININE_ITEMIDS)].dropna(subset=["charttime"]).copy()
    creat["value"] = pd.to_numeric(creat["valuenum"], errors="coerce")
    creat = creat[creat["value"].gt(0)].sort_values(["subject_id", "charttime"], kind="stable")
    aki_rows: list[dict[str, Any]] = []
    for subject, group in creat.groupby("subject_id", observed=True):
        for row in group.itertuples(index=False):
            prior = group[(group["charttime"] < row.charttime) & (group["charttime"] >= row.charttime - pd.Timedelta(days=7))]
            if prior.empty:
                continue
            baseline = float(prior["value"].min())
            if row.value >= baseline + 0.3 or row.value >= 1.5 * baseline:
                aki_rows.append({"hadm_id": int(row.hadm_id), "anchor_time": row.charttime, "anchor_source": "creatinine_delta_48h" if row.value - baseline >= .3 else "creatinine_ratio_7d"})
    aki_lab = pd.DataFrame(aki_rows)
    if not aki_lab.empty:
        aki_lab = aki_lab.merge(stays[["hadm_id", "stay_id", "intime", "outtime"]], on="hadm_id", how="inner")
        aki_lab = aki_lab[(aki_lab["anchor_time"] >= aki_lab["intime"]) & (aki_lab["anchor_time"] < aki_lab["outtime"])][["stay_id", "anchor_time", "anchor_source"]]
    p["label"] = p["itemid"].map(label).fillna("")
    rrt = p[p["label"].astype(str).str.contains(RRT_PATTERN, case=False, regex=True)][["stay_id", "starttime"]].rename(columns={"starttime": "anchor_time"}); rrt["anchor_source"] = "time_stamped_rrt_start"
    aki = _first_anchor(pd.concat([aki_lab, rrt], ignore_index=True))
    if not aki.empty:
        local = stays.merge(aki, on="stay_id", how="inner"); local["task_id"] = "aki"; frames.append(local)

    sepsis_stay_ids = set(sepsis["stay_id"].astype(int)) if not sepsis.empty else set()
    hf_contract = RUNTIME_CONFIG["cohort_parameters"]["heart_failure"]
    hf_stays = _prior_disease_stays(stays, diagnoses, icd9_prefix=tuple(hf_contract["icd9_prefix"]), icd10_prefix=tuple(hf_contract["icd10_prefix"]))
    hf_stays = hf_stays[~hf_stays["stay_id"].isin(sepsis_stay_ids)].copy()
    rx = _read(paths["hosp/prescriptions"], dates=("starttime", "stoptime"))
    rx = rx[rx["drug"].astype(str).str.contains(f"{DIURETIC_PATTERN}|{VASODILATOR_PATTERN}", case=False, regex=True, na=False)]
    hf_event = rx.merge(hf_stays[["hadm_id", "stay_id", "intime", "outtime"]], on="hadm_id")
    hf_event = hf_event[(hf_event["starttime"] >= hf_event["intime"]) & (hf_event["starttime"] < hf_event["outtime"])][["stay_id", "starttime"]].rename(columns={"starttime": "anchor_time"}); hf_event["anchor_source"] = "current_stay_decongestion_prescription"
    i = inputs.copy(); i["label"] = i["itemid"].map(label).fillna("")
    hf_input = i[i["label"].astype(str).str.contains(f"{DIURETIC_PATTERN}|{VASODILATOR_PATTERN}", case=False, regex=True)][["stay_id", "starttime"]].rename(columns={"starttime": "anchor_time"}); hf_input["anchor_source"] = "current_stay_decongestion_input"
    hf = _first_anchor(pd.concat([hf_event, hf_input], ignore_index=True))
    if not hf.empty:
        local = hf_stays.merge(hf, on="stay_id", how="inner"); local["task_id"] = "heart_failure"; frames.append(local)

    if not frames:
        raise ContractError("no task anchors constructed")
    candidates = pd.concat(frames, ignore_index=True)
    candidates = candidates.sort_values(["task_id", "stay_id", "anchor_time"], kind="stable").groupby(["task_id", "stay_id"], observed=True).first().reset_index()
    compact = candidates["task_id"].isin({"respiratory_support", "shock", "aki"})
    candidates.loc[compact, "role"] = [compact_lineage_role(value) for value in candidates.loc[compact, "subject_id"]]
    candidates["base_anchor_time"] = candidates["anchor_time"]
    candidates["window_start"] = candidates["anchor_time"] - pd.Timedelta(hours=PRE_ANCHOR_HOURS)
    candidates["window_end"] = [
        anchor + pd.Timedelta(hours=extraction_post_hours(task))
        for task, anchor in zip(candidates["task_id"], candidates["anchor_time"])
    ]
    candidates["episode_idx"] = np.arange(len(candidates), dtype=np.int64)
    if candidates.groupby(["task_id", "subject_id"], observed=True)["role"].nunique().gt(1).any():
        raise ContractError("subject role overlap within task")
    return candidates


def _bin(events: pd.DataFrame, candidates: pd.DataFrame, key: str, time: str) -> pd.DataFrame:
    merged = events.merge(candidates[[key, "episode_idx", "window_start", "window_end"]], on=key, how="inner")
    merged = merged[(merged[time] >= merged["window_start"]) & (merged[time] < merged["window_end"])].copy()
    merged["bin"] = np.floor((merged[time] - merged["window_start"]).dt.total_seconds() / (BIN_HOURS * 3600)).astype(int)
    return merged


def build_arrays(paths: dict[str, Path], candidates: pd.DataFrame) -> dict[str, np.ndarray]:
    steps = RAW_EXTRACTION_BINS
    shape = (len(candidates), steps, len(FEATURE_NAMES))
    values = np.full(shape, np.nan, dtype=np.float32); masks = np.zeros(shape, dtype=bool)
    fluid = np.zeros(shape[:2], dtype=np.float32); vaso = np.zeros(shape[:2], dtype=np.float32)
    peep = np.full(shape[:2], np.nan, dtype=np.float32); peep_observed = np.zeros(shape[:2], dtype=bool)
    diuretic = np.zeros(shape[:2], dtype=np.float32); rrt = np.zeros(shape[:2], dtype=np.float32)
    for row in candidates.itertuples(index=False):
        values[row.episode_idx, :, FEATURE_INDEX["age"]] = float(row.anchor_age)
        values[row.episode_idx, :, FEATURE_INDEX["gender_male"]] = float(str(row.gender).upper() == "M")
        values[row.episode_idx, :, FEATURE_INDEX["step_id"]] = np.arange(steps)
        masks[row.episode_idx, :, [FEATURE_INDEX["age"], FEATURE_INDEX["gender_male"], FEATURE_INDEX["step_id"]]] = True

    chart = _bin(_read(paths["icu/chartevents"], dates=("charttime",)), candidates, "stay_id", "charttime")
    chart = chart[chart["itemid"].isin(set(CHART_ITEM_MAP) | set(PEEP_ITEMIDS))].copy()
    chart["value"] = [corrected_chart_value(item, value, unit) for item, value, unit in zip(chart["itemid"], chart["valuenum"], chart["valueuom"])]
    valid_chart = chart[np.isfinite(chart["value"])].copy()
    peep_rows = valid_chart[valid_chart["itemid"].isin(PEEP_ITEMIDS)]
    for (episode, step), group in peep_rows.groupby(["episode_idx", "bin"], observed=True):
        peep[int(episode), int(step)] = float(group["value"].median()); peep_observed[int(episode), int(step)] = True
    gcs = valid_chart[valid_chart["itemid"].isin(GCS_ITEMIDS)].copy()
    gcs["feature_value"] = gcs["value"]
    for row in gcs_total_rows(gcs).itertuples(index=False):
        index = FEATURE_INDEX["gcs_proxy"]; values[int(row.episode_idx), int(row.bin), index] = float(row.feature_value); masks[int(row.episode_idx), int(row.bin), index] = True
    generic = valid_chart[~valid_chart["itemid"].isin(set(PEEP_ITEMIDS) | GCS_ITEMIDS)].copy()
    generic["feature"] = generic["itemid"].astype(int).map(CHART_ITEM_MAP)
    for (episode, step, feature), group in generic.groupby(["episode_idx", "bin", "feature"], observed=True):
        selected = group
        if feature == "mbp":
            invasive = group[group["itemid"].eq(220052)]
            selected = invasive if not invasive.empty else group[group["itemid"].eq(220181)]
        cleaned = clean_feature_values(str(feature), selected["value"].to_numpy(float))
        finite = cleaned[np.isfinite(cleaned)]
        if finite.size:
            index = FEATURE_INDEX[str(feature)]; values[int(episode), int(step), index] = float(np.median(finite)); masks[int(episode), int(step), index] = True

    labs = _bin(_read(paths["hosp/labevents"], dates=("charttime",)), candidates, "hadm_id", "charttime")
    labs = labs[labs["itemid"].isin(LAB_ITEM_MAP)].copy(); labs["value"] = pd.to_numeric(labs["valuenum"], errors="coerce")
    labs = labs[np.isfinite(labs["value"]) & ~labs["value"].isin((-9999, -999, -99))]
    labs["feature"] = labs["itemid"].astype(int).map(LAB_ITEM_MAP)
    for (episode, step, feature), group in labs.groupby(["episode_idx", "bin", "feature"], observed=True):
        if feature in {"lactate", "ionized_calcium"}:
            group = group[group["valueuom"].astype("string").eq("mmol/L")]
        cleaned = clean_feature_values(str(feature), group["value"].to_numpy(float))
        finite = cleaned[np.isfinite(cleaned)]
        if finite.size:
            index = FEATURE_INDEX[str(feature)]; values[int(episode), int(step), index] = float(np.median(finite)); masks[int(episode), int(step), index] = True

    outputs = _bin(_read(paths["icu/outputevents"], dates=("charttime",)), candidates, "stay_id", "charttime")
    outputs = outputs[outputs["itemid"].isin(URINE_ITEMIDS)].copy(); outputs["numeric"] = pd.to_numeric(outputs["value"], errors="coerce")
    for (episode, step), group in outputs[outputs["numeric"].ge(0)].groupby(["episode_idx", "bin"], observed=True):
        cleaned = clean_feature_values("urine_output", np.asarray([group["numeric"].sum()], dtype=float))
        if np.isfinite(cleaned[0]):
            index = FEATURE_INDEX["urine_output"]; values[int(episode), int(step), index] = float(cleaned[0]); masks[int(episode), int(step), index] = True

    items = _read(paths["icu/d_items"])[["itemid", "label"]]; labels = items.set_index("itemid")["label"]
    inputs = _read(paths["icu/inputevents"], dates=("starttime", "endtime")); inputs["label"] = inputs["itemid"].map(labels).fillna("")
    ib = inputs.merge(candidates[["stay_id", "episode_idx", "window_start", "window_end"]], on="stay_id", how="inner")
    ib["endtime"] = ib["endtime"].fillna(ib["starttime"])
    ib = ib[(ib["endtime"] >= ib["window_start"]) & (ib["starttime"] < ib["window_end"])].copy()
    ib["amount_value"] = pd.to_numeric(ib["amount"], errors="coerce").fillna(0); ib["rate_value"] = pd.to_numeric(ib["rate"], errors="coerce").fillna(0)
    for row in ib.itertuples(index=False):
        episode = int(row.episode_idx)
        start = max(row.starttime, row.window_start); end = min(max(row.endtime, start + pd.Timedelta(minutes=1)), row.window_end)
        magnitude = max(float(row.amount_value), float(row.rate_value))
        if magnitude <= 0: continue
        if int(row.itemid) in FLUID_ITEMIDS and str(row.amountuom).lower() == "ml":
            for step, amount in overlap_bin_amounts(start, end, row.window_start, bin_hours=BIN_HOURS, n_steps=steps, amount=float(row.amount_value)):
                fluid[episode, step] += amount
        for step in overlap_bins(start, end, row.window_start, bin_hours=BIN_HOURS, n_steps=steps):
            if int(row.itemid) in VASO_ITEMIDS: vaso[episode, step] = max(vaso[episode, step], float(row.rate_value or row.amount_value))
            if re.search(DIURETIC_PATTERN, str(row.label), re.I): diuretic[episode, step] = 1
            if re.search(RRT_PATTERN, str(row.label), re.I): rrt[episode, step] = 1

    procedures = _read(paths["icu/procedureevents"], dates=("starttime", "endtime")); procedures["label"] = procedures["itemid"].map(labels).fillna("")
    pb = procedures.merge(candidates[["stay_id", "episode_idx", "window_start", "window_end"]], on="stay_id", how="inner")
    pb["endtime"] = pb["endtime"].fillna(pb["starttime"])
    pb = pb[(pb["endtime"] > pb["window_start"]) & (pb["starttime"] < pb["window_end"]) & (pb["endtime"] > pb["starttime"])].copy()
    for row in pb.itertuples(index=False):
        episode = int(row.episode_idx); start = max(row.starttime, row.window_start); end = min(row.endtime, row.window_end)
        for step in overlap_bins(start, end, row.window_start, bin_hours=BIN_HOURS, n_steps=steps):
            if int(row.itemid) in MECHVENT_ITEMIDS:
                index = FEATURE_INDEX["mechanical_ventilation"]; values[episode, step, index] = 1; masks[episode, step, index] = True
            if re.search(RRT_PATTERN, str(row.label), re.I): rrt[episode, step] = 1

    rx = _read(paths["hosp/prescriptions"], dates=("starttime", "stoptime")); rb = rx.merge(candidates[["hadm_id", "episode_idx", "window_start", "window_end"]], on="hadm_id", how="inner")
    rb["stoptime"] = rb["stoptime"].fillna(rb["starttime"])
    rb = rb[(rb["stoptime"] >= rb["window_start"]) & (rb["starttime"] < rb["window_end"])].copy()
    for row in rb.itertuples(index=False):
        if re.search(DIURETIC_PATTERN, str(row.drug), re.I):
            start = max(row.starttime, row.window_start); end = min(max(row.stoptime, start + pd.Timedelta(minutes=1)), row.window_end)
            for step in overlap_bins(start, end, row.window_start, bin_hours=BIN_HOURS, n_steps=steps): diuretic[int(row.episode_idx), step] = 1

    compute_corrected_derived_features(values, masks, vaso, feature_index=FEATURE_INDEX, bin_hours=BIN_HOURS)
    return {"values": values, "masks": masks, "fluid": fluid, "vaso": vaso, "peep": peep, "peep_observed": peep_observed, "diuretic": diuretic, "rrt": rrt}


def finalize_sepsis_anchors(candidates: pd.DataFrame, arrays: dict[str, np.ndarray]) -> pd.DataFrame:
    output = candidates.copy(); keep = np.ones(len(output), dtype=bool); step = pd.Timedelta(hours=BIN_HOURS)
    mask = arrays["masks"]
    sepsis = RUNTIME_CONFIG["cohort_parameters"]["sepsis"]
    valid_domains = int(sepsis["minimum_observed_sofa_domains"])
    domains = (
        (mask[..., FEATURE_INDEX["pao2"]] & mask[..., FEATURE_INDEX["fio2"]]).astype(np.int8)
        + mask[..., FEATURE_INDEX["platelet"]].astype(np.int8)
        + mask[..., FEATURE_INDEX["total_bilirubin"]].astype(np.int8)
        + (mask[..., FEATURE_INDEX["mbp"]] | (arrays["vaso"] > 0)).astype(np.int8)
        + mask[..., FEATURE_INDEX["gcs_proxy"]].astype(np.int8)
        + (mask[..., FEATURE_INDEX["creatinine"]] | mask[..., FEATURE_INDEX["urine_output"]]).astype(np.int8)
    )
    for row in output[output["task_id"].eq("sepsis")].itertuples(index=False):
        index = int(row.episode_idx); scores = arrays["values"][index, :, FEATURE_INDEX["sofa_proxy"]]; observed = mask[index, :, FEATURE_INDEX["sofa_proxy"]]
        pre: list[int] = []; post: list[int] = []
        for bin_index in range(len(scores)):
            start, end = row.window_start + bin_index * step, row.window_start + (bin_index + 1) * step
            valid = start >= row.intime and end <= row.outtime and observed[bin_index] and domains[index, bin_index] >= valid_domains
            if valid and end <= row.base_anchor_time: pre.append(bin_index)
            elif valid and start >= row.base_anchor_time and end <= row.base_anchor_time + pd.Timedelta(hours=48): post.append(bin_index)
        if not pre:
            keep[index] = False; continue
        baseline = float(np.nanmin(scores[pre])); qualifying = [item for item in post if float(scores[item]) - baseline >= float(sepsis["sofa_increase_min"])]
        if not qualifying:
            keep[index] = False; continue
        organ = min(qualifying); output.loc[output["episode_idx"].eq(index), "anchor_time"] = row.window_start + (organ + 1) * step
        output.loc[output["episode_idx"].eq(index), "anchor_source"] = "antibiotic_culture_plus_time_bounded_proxy_sofa_increase"
    return output.loc[keep].copy()


def build_transitions(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in candidates.itertuples(index=False):
        for relative in eligible_transition_indices(candidate.anchor_time, candidate.intime, candidate.outtime):
            if not apply_kdd201_temporal_repair(candidate.anchor_source, [relative])[0]:
                continue
            action_start = candidate.anchor_time + relative * pd.Timedelta(hours=BIN_HOURS)
            state = int((action_start - pd.Timedelta(hours=BIN_HOURS) - candidate.window_start) / pd.Timedelta(hours=BIN_HOURS))
            action = int((action_start - candidate.window_start) / pd.Timedelta(hours=BIN_HOURS))
            target = int((action_start + pd.Timedelta(hours=BIN_HOURS) - candidate.window_start) / pd.Timedelta(hours=BIN_HOURS))
            rows.append({"task": candidate.task_id, "role": candidate.role, "episode_idx": int(candidate.episode_idx), "relative_transition": relative, "state_idx": state, "action_idx": action, "target_idx": target})
    return pd.DataFrame(rows)


def encode_actions(task: str, transitions: pd.DataFrame, arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    episode = transitions["episode_idx"].to_numpy(int); step = transitions["action_idx"].to_numpy(int); roles = transitions["role"].astype(str).to_numpy()
    if task in {"sepsis", "shock"}:
        left, right = arrays["fluid"][episode, step], arrays["vaso"][episode, step]; lo = ro = np.ones(len(step), bool)
    elif task == "respiratory_support":
        left, lo = arrays["peep"][episode, step], arrays["peep_observed"][episode, step]
        index = FEATURE_INDEX["fio2"]; right, ro = arrays["values"][episode, step, index], arrays["masks"][episode, step, index]
    elif task == "aki":
        return (arrays["diuretic"][episode, step] > 0).astype(np.int16) + 2 * (arrays["rrt"][episode, step] > 0).astype(np.int16), {"K": 4, "edges": []}
    elif task == "heart_failure":
        return (arrays["diuretic"][episode, step] > 0).astype(np.int16), {"K": 2, "edges": []}
    else: raise ContractError(f"unknown task: {task}")
    le, re_ = fit_train_positive_edges(left, lo, roles), fit_train_positive_edges(right, ro, roles)
    valid = lo & ro
    return joint_codes(encode_five_levels(left, lo, le), encode_five_levels(right, ro, re_), valid), {"K": 25, "edges": [le, re_]}


def task_aggregate(task: str, candidates: pd.DataFrame, transitions: pd.DataFrame, actions: np.ndarray, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    episode_ids = transitions["episode_idx"].unique(); local = candidates[candidates["episode_idx"].isin(episode_ids)]
    counts = np.bincount(actions[actions >= 0], minlength={"sepsis":25,"respiratory_support":25,"shock":25,"aki":4,"heart_failure":2}[task])
    lengths = transitions.groupby("episode_idx", observed=True).size().to_numpy()
    if task in {"sepsis", "aki", "heart_failure"}:
        reward_contract = "terminal_discharge_origin_90d_proxy_once_at_final_valid_transition"
        reward_observed = int(len(episode_ids))
        terminal_rewards = int(len(episode_ids))
    elif task == "shock":
        target_episode = transitions["episode_idx"].to_numpy(int); target_step = transitions["target_idx"].to_numpy(int)
        reward_contract = "strictly_post_action_next_MBP_dense_component"
        reward_observed = int(arrays["masks"][target_episode, target_step, FEATURE_INDEX["mbp"]].sum())
        terminal_rewards = 0
    else:
        target_episode = transitions["episode_idx"].to_numpy(int); target_step = transitions["target_idx"].to_numpy(int)
        reward_contract = "strictly_post_action_next_SpO2_and_MBP_dense_reward"
        reward_observed = int((arrays["masks"][target_episode, target_step, FEATURE_INDEX["spo2"]] & arrays["masks"][target_episode, target_step, FEATURE_INDEX["mbp"]]).sum())
        terminal_rewards = 0
    return {"subjects": int(local["subject_id"].nunique()), "episodes": int(len(episode_ids)), "decisions": int(len(transitions)), "action_counts": counts.astype(int).tolist(), "minimum_horizon": int(lengths.min()) if len(lengths) else 0, "maximum_horizon": int(lengths.max()) if len(lengths) else 0, "reward_contract": reward_contract, "reward_observed_decisions": reward_observed, "terminal_reward_count": terminal_rewards}


def reconstruct(root: Path, output: Path, schema: Path, *, source_hashes: dict[str, str] | None = None, runtime_config: Path | None = None) -> dict[str, Any]:
    from .runtime_config import load_runtime_config

    load_runtime_config(runtime_config)
    if output.exists(): raise FileExistsError(output)
    paths = validate_layout(root)
    stays, diagnoses = load_core(paths); candidates = build_anchors(paths, stays, diagnoses); arrays = build_arrays(paths, candidates); candidates = finalize_sepsis_anchors(candidates, arrays); transitions = build_transitions(candidates)
    rows: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        local = transitions[transitions["task"].eq(task)].reset_index(drop=True)
        if local.empty: raise ContractError(f"no valid transitions for {task}")
        actions, contract = encode_actions(task, local, arrays)
        if np.any(actions < 0): raise ContractError(f"missing action observation for {task}")
        row = task_aggregate(task, candidates, local, actions, arrays); row["action_count"] = contract["K"]
        row["cutpoint_hash"] = hashlib.sha256(json.dumps(contract["edges"], sort_keys=True).encode()).hexdigest(); rows[task] = row
    receipt = aggregate_receipt(rows, source_hashes or {})
    schema_object = json.loads(schema.read_text(encoding="utf-8")); Draft202012Validator.check_schema(schema_object); Draft202012Validator(schema_object).validate(receipt)
    output.mkdir(parents=True); (output / "aggregate_receipt.json").write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the current five-task aggregate receipt from MIMIC-IV 3.1 flat files")
    parser.add_argument("--mimiciv-root", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path)
    args = parser.parse_args()
    os.environ.setdefault("TZ", "UTC")
    reconstruct(args.mimiciv_root, args.output, args.schema, runtime_config=args.runtime_config)


if __name__ == "__main__":
    main()
