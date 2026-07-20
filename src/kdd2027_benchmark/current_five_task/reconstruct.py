from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
from collections import deque
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


HIGH_VOLUME_TABLES = frozenset({
    "icu/chartevents",
    "hosp/labevents",
    "hosp/microbiologyevents",
    "hosp/prescriptions",
    "icu/inputevents",
    "icu/outputevents",
    "icu/procedureevents",
})
DEFAULT_CHUNK_ROWS = int(RUNTIME_CONFIG["runtime"]["high_volume_chunk_rows"])


def _audit_add(audit: dict[tuple[str, str, str], int], table: str, field: str, reason: str, count: int) -> None:
    if count:
        key = (table, field, reason)
        audit[key] = audit.get(key, 0) + int(count)


def _coerce_required_integer(
    frame: pd.DataFrame,
    column: str,
    table: str,
    audit: dict[tuple[str, str, str], int],
) -> pd.Series:
    raw = frame[column]
    text = raw.astype("string")
    missing = raw.isna() | text.str.strip().eq("")
    numeric = pd.to_numeric(raw, errors="coerce")
    malformed = ~missing & (~np.isfinite(numeric) | numeric.mod(1).ne(0) | numeric.le(0))
    if malformed.any():
        raise ContractError(f"malformed required identifier in {table}.{column}: count={int(malformed.sum())}")
    _audit_add(audit, table, column, "missing_required_identifier", int(missing.sum()))
    return ~missing


def _filter_required_ids(
    frame: pd.DataFrame,
    columns: Iterable[str],
    table: str,
    audit: dict[tuple[str, str, str], int],
) -> pd.DataFrame:
    keep = np.ones(len(frame), dtype=bool)
    columns = tuple(columns)
    for column in columns:
        keep &= _coerce_required_integer(frame, column, table, audit).to_numpy(bool)
    output = frame.loc[keep].copy()
    for column in columns:
        output[column] = pd.to_numeric(output[column], errors="raise").astype("int64")
    return output


def _stream_events(
    path: Path,
    *,
    table: str,
    columns: Iterable[str],
    dates: Iterable[str],
    required_ids: Iterable[str],
    required_times: Iterable[str],
    audit: dict[tuple[str, str, str], int],
    chunk_rows: int,
    itemids: set[int] | None = None,
    key: str | None = None,
    eligible_keys: set[int] | None = None,
    bounds: pd.DataFrame | None = None,
    point_time: str | None = None,
    interval_times: tuple[str, str] | None = None,
    sort_by: Iterable[str] = (),
) -> pd.DataFrame:
    if table not in HIGH_VOLUME_TABLES:
        raise ContractError(f"unregistered high-volume table: {table}")
    if chunk_rows <= 0:
        raise ContractError("chunk_rows must be positive")
    selected_frames: list[pd.DataFrame] = []
    source_offset = 0
    chunks: Any | None = None
    source: Any | None = None
    try:
        source = gzip.open(path, "rt", encoding="utf-8", newline="") if path.suffix == ".gz" else path.open("r", encoding="utf-8", newline="")
        chunks = pd.read_csv(source, usecols=list(columns), chunksize=chunk_rows, low_memory=False)
        try:
            for raw_chunk in chunks:
                chunk = raw_chunk.copy()
                chunk["__source_order"] = np.arange(source_offset, source_offset + len(chunk), dtype=np.int64)
                source_offset += len(chunk)
                required_id_columns = tuple(required_ids)
                if itemids is not None and "itemid" in required_id_columns:
                    valid_itemid = _coerce_required_integer(chunk, "itemid", table, audit)
                    chunk = chunk.loc[valid_itemid].copy()
                    if chunk.empty:
                        continue
                    chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="raise").astype("int64")
                    chunk = chunk[chunk["itemid"].isin(itemids)]
                    if chunk.empty:
                        continue
                chunk = parse_times(chunk, dates, table)
                keep = np.ones(len(chunk), dtype=bool)
                for column in required_id_columns:
                    if column == "itemid" and itemids is not None:
                        continue
                    keep &= _coerce_required_integer(chunk, column, table, audit).to_numpy(bool)
                for column in required_times:
                    missing = chunk[column].isna()
                    _audit_add(audit, table, column, "missing_required_timestamp", int(missing.sum()))
                    keep &= ~missing.to_numpy(bool)
                chunk = chunk.loc[keep].copy()
                if chunk.empty:
                    continue
                for column in required_id_columns:
                    chunk[column] = pd.to_numeric(chunk[column], errors="raise").astype("int64")
                if itemids is not None and "itemid" not in required_id_columns:
                    chunk = chunk[chunk["itemid"].isin(itemids)]
                if key is not None and eligible_keys is not None:
                    chunk = chunk[chunk[key].isin(eligible_keys)]
                if chunk.empty:
                    continue
                if bounds is not None:
                    if key is None:
                        raise ContractError("bounded stream requires a join key")
                    chunk = chunk.merge(bounds, on=key, how="inner", sort=False)
                    if point_time is not None:
                        chunk = chunk[(chunk[point_time] >= chunk["window_start"]) & (chunk[point_time] < chunk["window_end"])]
                    elif interval_times is not None:
                        start, end = interval_times
                        effective_end = chunk[end].fillna(chunk[start])
                        chunk = chunk[(effective_end >= chunk["window_start"]) & (chunk[start] < chunk["window_end"])]
                    else:
                        raise ContractError("bounded stream requires point or interval time semantics")
                if not chunk.empty:
                    selected_frames.append(chunk)
        finally:
            chunks.close()
            source.close()
    except (UnicodeDecodeError, ValueError, OSError) as exc:
        if isinstance(exc, ContractError):
            raise
        raise ContractError(f"unsupported or malformed high-volume source: {table}") from exc
    output_columns = list(columns) + ["__source_order"]
    if bounds is not None:
        output_columns += [column for column in bounds.columns if column != key]
    if not selected_frames:
        return pd.DataFrame(columns=list(dict.fromkeys(output_columns)))
    output = pd.concat(selected_frames, ignore_index=True)
    order = [column for column in sort_by if column in output] + ["__source_order"]
    if order:
        output = output.sort_values(order, kind="stable")
    return output.reset_index(drop=True)


def scan_creatinine(
    path: Path,
    *,
    chunk_rows: int,
    audit: dict[tuple[str, str, str], int],
) -> pd.DataFrame:
    creatinine = _stream_events(
        path,
        table="hosp/labevents",
        columns=("subject_id", "hadm_id", "charttime", "itemid", "valuenum"),
        dates=("charttime",),
        required_ids=("subject_id", "hadm_id", "itemid"),
        required_times=("charttime",),
        audit=audit,
        chunk_rows=chunk_rows,
        itemids=set(CREATININE_ITEMIDS),
        sort_by=("subject_id", "charttime"),
    )
    if creatinine.empty:
        return pd.DataFrame(columns=("subject_id", "hadm_id", "charttime", "value", "__source_order"))
    creatinine["value"] = pd.to_numeric(creatinine["valuenum"], errors="coerce")
    invalid_value = ~np.isfinite(creatinine["value"]) | creatinine["value"].le(0)
    _audit_add(audit, "hosp/labevents", "valuenum", "missing_nonfinite_or_nonpositive_creatinine", int(invalid_value.sum()))
    creatinine = creatinine.loc[~invalid_value, ["subject_id", "hadm_id", "charttime", "value", "__source_order"]]
    return creatinine.sort_values(["subject_id", "hadm_id", "charttime", "__source_order"], kind="stable").reset_index(drop=True)


def kdigo_creatinine_events(creatinine: pd.DataFrame) -> pd.DataFrame:
    qualifying: list[dict[str, Any]] = []
    h48 = pd.Timedelta(hours=48).value
    d7 = pd.Timedelta(days=7).value
    current_admission: tuple[int, int] | None = None
    q48: deque[tuple[int, float]] = deque()
    q7: deque[tuple[int, float]] = deque()
    for row in creatinine.itertuples(index=False):
        subject = int(row.subject_id)
        admission = int(row.hadm_id)
        when = int(pd.Timestamp(row.charttime).value)
        value = float(row.value)
        if (subject, admission) != current_admission:
            current_admission = (subject, admission)
            q48.clear()
            q7.clear()
        while q48 and q48[0][0] < when - h48:
            q48.popleft()
        while q7 and q7[0][0] < when - d7:
            q7.popleft()
        min48 = q48[0][1] if q48 else math.nan
        min7 = q7[0][1] if q7 else math.nan
        delta = np.isfinite(min48) and value - min48 >= 0.3
        ratio = np.isfinite(min7) and min7 > 0 and value / min7 >= 1.5
        if delta or ratio:
            qualifying.append({
                "subject_id": subject,
                "hadm_id": admission,
                "anchor_time": pd.Timestamp(row.charttime),
                "anchor_source": "creatinine_delta_48h" if delta else "creatinine_ratio_7d",
            })
        while q48 and q48[-1][1] >= value:
            q48.pop()
        while q7 and q7[-1][1] >= value:
            q7.pop()
        q48.append((when, value))
        q7.append((when, value))
    return pd.DataFrame(qualifying, columns=("subject_id", "hadm_id", "anchor_time", "anchor_source"))


def _normal_icd(series: pd.Series) -> pd.Series:
    return series.astype("string").str.upper().str.replace(".", "", regex=False)


def _mortality_90d(frame: pd.DataFrame) -> pd.Series:
    dod = pd.to_datetime(frame["dod"], errors="coerce")
    discharge = pd.to_datetime(frame["dischtime"], errors="coerce")
    in_hospital = frame["hospital_expire_flag"].fillna(0).astype(int).eq(1)
    after = dod.notna() & discharge.notna() & dod.le(discharge + pd.Timedelta(days=90))
    return (in_hospital | after).astype(np.int8)


def load_core(paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    base, diagnoses, _ = load_core_with_time_audit(paths)
    return base, diagnoses


def classify_icu_time_eligibility(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    missing_intime = frame["intime"].isna()
    missing_outtime = ~missing_intime & frame["outtime"].isna()
    both_observed = frame["intime"].notna() & frame["outtime"].notna()
    equal_time = both_observed & frame["outtime"].eq(frame["intime"])
    reversed_time = both_observed & frame["outtime"].lt(frame["intime"])
    valid_time = both_observed & frame["outtime"].gt(frame["intime"])
    audit = {
        "total_merged_icu_stays_considered": int(len(frame)),
        "valid_time_order_stays": int(valid_time.sum()),
        "missing_intime": int(missing_intime.sum()),
        "missing_outtime": int(missing_outtime.sum()),
        "equal_intime_outtime": int(equal_time.sum()),
        "reversed_time_order": int(reversed_time.sum()),
    }
    if sum(audit[key] for key in audit if key != "total_merged_icu_stays_considered") != audit["total_merged_icu_stays_considered"]:
        raise ContractError("ICU time-order eligibility categories are not exhaustive and mutually exclusive")
    retained = frame.loc[valid_time].copy()
    if retained["intime"].isna().any() or retained["outtime"].isna().any() or retained["outtime"].le(retained["intime"]).any():
        raise ContractError("invalid ICU time order remained after eligibility filtering")
    return retained, audit


def load_core_with_time_audit(
    paths: dict[str, Path],
    ingestion_audit: dict[tuple[str, str, str], int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    ingestion_audit = ingestion_audit if ingestion_audit is not None else {}
    patients = _read(paths["hosp/patients"], dates=("dod",))
    admissions = _read(paths["hosp/admissions"], dates=("admittime", "dischtime", "deathtime"))
    stays = _read(paths["icu/icustays"], dates=("intime", "outtime"))
    patients = _filter_required_ids(patients, ("subject_id",), "hosp/patients", ingestion_audit)
    admissions = _filter_required_ids(admissions, ("subject_id", "hadm_id"), "hosp/admissions", ingestion_audit)
    stays = _filter_required_ids(stays, ("subject_id", "hadm_id", "stay_id"), "icu/icustays", ingestion_audit)
    assert_unique(patients, ("subject_id",), "patients")
    assert_unique(admissions, ("hadm_id",), "admissions")
    assert_unique(stays, ("stay_id",), "icustays")
    base = (
        stays.merge(admissions, on=("subject_id", "hadm_id"), how="inner")
        .merge(patients[["subject_id", "gender", "anchor_age", "dod"]], on="subject_id", how="inner")
        .sort_values(["subject_id", "intime", "stay_id"], kind="stable")
    )
    base, time_audit = classify_icu_time_eligibility(base)
    discharge = base["discharge_location"].fillna("").astype(str).str.upper()
    base = base[
        base["anchor_age"].ge(18)
        & base["gender"].notna()
        & base["dischtime"].notna()
        & ~discharge.str.contains("HOSPICE")
    ].copy()
    time_audit["retained_after_existing_public_base_eligibility"] = int(len(base))
    base["mortality_90d"] = _mortality_90d(base)
    base["role"] = [subject_role(value) for value in base["subject_id"]]
    diagnoses = _read(paths["hosp/diagnoses_icd"])
    return base.reset_index(drop=True), diagnoses, time_audit


def _first_anchor(frame: pd.DataFrame, source: str | None = None) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["stay_id", "anchor_time", "anchor_source"])
    if source is not None:
        frame = frame.copy(); frame["anchor_source"] = source
    if "anchor_source" not in frame:
        raise ContractError("anchor source is required")
    output = frame.sort_values(["stay_id", "anchor_time"], kind="stable").groupby("stay_id", observed=True).first().reset_index()
    return output[["stay_id", "anchor_time", "anchor_source"]]


def suspected_infection_anchors(
    paths: dict[str, Path],
    stays: pd.DataFrame,
    *,
    chunk_rows: int,
    audit: dict[tuple[str, str, str], int],
) -> pd.DataFrame:
    prescriptions = _stream_events(
        paths["hosp/prescriptions"], table="hosp/prescriptions",
        columns=("subject_id", "hadm_id", "starttime", "stoptime", "drug"),
        dates=("starttime", "stoptime"), required_ids=("subject_id", "hadm_id"), required_times=("starttime",),
        audit=audit, chunk_rows=chunk_rows,
        key="hadm_id", eligible_keys=set(stays["hadm_id"].astype(int)), sort_by=("subject_id", "starttime"),
    )
    antibiotics = prescriptions[
        prescriptions["hadm_id"].isin(stays["hadm_id"])
        & prescriptions["starttime"].notna()
        & prescriptions["drug"].astype(str).str.contains(ANTIBIOTIC_PATTERN, case=False, regex=True, na=False)
    ].copy()
    antibiotics = antibiotics.merge(stays[["subject_id", "hadm_id", "stay_id"]], on=["subject_id", "hadm_id"], how="inner")
    cultures = _stream_events(
        paths["hosp/microbiologyevents"], table="hosp/microbiologyevents",
        columns=("subject_id", "hadm_id", "chartdate", "charttime"),
        dates=("chartdate", "charttime"), required_ids=("subject_id",), required_times=(),
        audit=audit, chunk_rows=chunk_rows,
        key="subject_id", eligible_keys=set(stays["subject_id"].astype(int)), sort_by=("subject_id", "charttime", "chartdate"),
    )
    cultures["culture_time"] = cultures["charttime"].fillna(cultures["chartdate"])
    missing_culture_time = cultures["culture_time"].isna()
    _audit_add(audit, "hosp/microbiologyevents", "charttime_or_chartdate", "missing_required_event_timestamp", int(missing_culture_time.sum()))
    cultures = cultures[~missing_culture_time].sort_values(["subject_id", "culture_time", "__source_order"], kind="stable")
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


def _prior_disease_stays(
    stays: pd.DataFrame,
    diagnoses: pd.DataFrame,
    *,
    icd9_prefix: tuple[str, ...],
    icd10_prefix: tuple[str, ...],
    audit: dict[tuple[str, str, str], int],
) -> pd.DataFrame:
    diagnoses = _filter_required_ids(diagnoses, ("subject_id", "hadm_id", "icd_version"), "hosp/diagnoses_icd", audit)
    missing_code = diagnoses["icd_code"].isna() | diagnoses["icd_code"].astype("string").str.strip().eq("")
    _audit_add(audit, "hosp/diagnoses_icd", "icd_code", "missing_required_code", int(missing_code.sum()))
    diagnoses = diagnoses.loc[~missing_code].copy()
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


def build_anchors(
    paths: dict[str, Path],
    stays: pd.DataFrame,
    diagnoses: pd.DataFrame,
    *,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    audit: dict[tuple[str, str, str], int] | None = None,
) -> pd.DataFrame:
    audit = audit if audit is not None else {}
    items = _read(paths["icu/d_items"])[["itemid", "label"]]
    items = _filter_required_ids(items, ("itemid",), "icu/d_items", audit)
    label = items.set_index("itemid")["label"]
    stay_ids = set(stays["stay_id"].astype(int))
    chart = _stream_events(
        paths["icu/chartevents"], table="icu/chartevents",
        columns=("stay_id", "charttime", "itemid", "valuenum", "valueuom"),
        dates=("charttime",), required_ids=("stay_id", "itemid"), required_times=("charttime",),
        audit=audit, chunk_rows=chunk_rows,
        itemids=set(MECHVENT_ITEMIDS + SBP_ITEMIDS + MBP_ITEMIDS),
        key="stay_id", eligible_keys=stay_ids, sort_by=("stay_id", "charttime", "itemid"),
    )
    input_label_ids = set(label[label.astype(str).str.contains(f"{DIURETIC_PATTERN}|{VASODILATOR_PATTERN}|{RRT_PATTERN}", case=False, regex=True, na=False)].index.astype(int))
    inputs = _stream_events(
        paths["icu/inputevents"], table="icu/inputevents",
        columns=("stay_id", "starttime", "endtime", "itemid", "amount", "rate"),
        dates=("starttime", "endtime"), required_ids=("stay_id", "itemid"), required_times=("starttime",),
        audit=audit, chunk_rows=chunk_rows,
        itemids=set(VASO_ITEMIDS) | input_label_ids,
        key="stay_id", eligible_keys=stay_ids, sort_by=("stay_id", "starttime", "itemid"),
    )
    procedure_label_ids = set(label[label.astype(str).str.contains(f"oxygen|ventilat|intubat|bipap|cpap|{RRT_PATTERN}", case=False, regex=True, na=False)].index.astype(int))
    procedures = _stream_events(
        paths["icu/procedureevents"], table="icu/procedureevents",
        columns=("stay_id", "starttime", "endtime", "itemid"),
        dates=("starttime", "endtime"), required_ids=("stay_id", "itemid"), required_times=("starttime",),
        audit=audit, chunk_rows=chunk_rows,
        itemids=set(MECHVENT_ITEMIDS) | procedure_label_ids,
        key="stay_id", eligible_keys=stay_ids, sort_by=("stay_id", "starttime", "itemid"),
    )
    frames: list[pd.DataFrame] = []

    sepsis = suspected_infection_anchors(paths, stays, chunk_rows=chunk_rows, audit=audit)
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

    creat = scan_creatinine(paths["hosp/labevents"], chunk_rows=chunk_rows, audit=audit)
    aki_lab = kdigo_creatinine_events(creat)
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
    hf_stays = _prior_disease_stays(stays, diagnoses, icd9_prefix=tuple(hf_contract["icd9_prefix"]), icd10_prefix=tuple(hf_contract["icd10_prefix"]), audit=audit)
    hf_stays = hf_stays[~hf_stays["stay_id"].isin(sepsis_stay_ids)].copy()
    rx = _stream_events(
        paths["hosp/prescriptions"], table="hosp/prescriptions",
        columns=("subject_id", "hadm_id", "starttime", "stoptime", "drug"),
        dates=("starttime", "stoptime"), required_ids=("subject_id", "hadm_id"), required_times=("starttime",),
        audit=audit, chunk_rows=chunk_rows,
        key="hadm_id", eligible_keys=set(hf_stays["hadm_id"].astype(int)), sort_by=("hadm_id", "starttime"),
    )
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


def build_arrays(
    paths: dict[str, Path],
    candidates: pd.DataFrame,
    *,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    audit: dict[tuple[str, str, str], int] | None = None,
) -> dict[str, np.ndarray]:
    audit = audit if audit is not None else {}
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

    stay_bounds = candidates[["stay_id", "episode_idx", "window_start", "window_end"]]
    chart = _stream_events(
        paths["icu/chartevents"], table="icu/chartevents",
        columns=("stay_id", "charttime", "itemid", "valuenum", "valueuom"),
        dates=("charttime",), required_ids=("stay_id", "itemid"), required_times=("charttime",),
        audit=audit, chunk_rows=chunk_rows, itemids=set(CHART_ITEM_MAP) | set(PEEP_ITEMIDS),
        key="stay_id", bounds=stay_bounds, point_time="charttime",
        sort_by=("episode_idx", "charttime", "itemid"),
    )
    chart["bin"] = np.floor((chart["charttime"] - chart["window_start"]).dt.total_seconds() / (BIN_HOURS * 3600)).astype(int)
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

    hadm_bounds = candidates[["hadm_id", "episode_idx", "window_start", "window_end"]]
    labs = _stream_events(
        paths["hosp/labevents"], table="hosp/labevents",
        columns=("hadm_id", "charttime", "itemid", "valuenum", "valueuom"),
        dates=("charttime",), required_ids=("hadm_id", "itemid"), required_times=("charttime",),
        audit=audit, chunk_rows=chunk_rows, itemids=set(LAB_ITEM_MAP),
        key="hadm_id", bounds=hadm_bounds, point_time="charttime",
        sort_by=("episode_idx", "charttime", "itemid"),
    )
    labs["bin"] = np.floor((labs["charttime"] - labs["window_start"]).dt.total_seconds() / (BIN_HOURS * 3600)).astype(int)
    labs["value"] = pd.to_numeric(labs["valuenum"], errors="coerce")
    labs = labs[np.isfinite(labs["value"]) & ~labs["value"].isin((-9999, -999, -99))]
    labs["feature"] = labs["itemid"].astype(int).map(LAB_ITEM_MAP)
    for (episode, step, feature), group in labs.groupby(["episode_idx", "bin", "feature"], observed=True):
        if feature in {"lactate", "ionized_calcium"}:
            group = group[group["valueuom"].astype("string").eq("mmol/L")]
        cleaned = clean_feature_values(str(feature), group["value"].to_numpy(float))
        finite = cleaned[np.isfinite(cleaned)]
        if finite.size:
            index = FEATURE_INDEX[str(feature)]; values[int(episode), int(step), index] = float(np.median(finite)); masks[int(episode), int(step), index] = True

    outputs = _stream_events(
        paths["icu/outputevents"], table="icu/outputevents",
        columns=("stay_id", "charttime", "itemid", "value"),
        dates=("charttime",), required_ids=("stay_id", "itemid"), required_times=("charttime",),
        audit=audit, chunk_rows=chunk_rows, itemids=set(URINE_ITEMIDS),
        key="stay_id", bounds=stay_bounds, point_time="charttime",
        sort_by=("episode_idx", "charttime", "itemid"),
    )
    outputs["bin"] = np.floor((outputs["charttime"] - outputs["window_start"]).dt.total_seconds() / (BIN_HOURS * 3600)).astype(int)
    outputs["numeric"] = pd.to_numeric(outputs["value"], errors="coerce")
    for (episode, step), group in outputs[outputs["numeric"].ge(0)].groupby(["episode_idx", "bin"], observed=True):
        cleaned = clean_feature_values("urine_output", np.asarray([group["numeric"].sum()], dtype=float))
        if np.isfinite(cleaned[0]):
            index = FEATURE_INDEX["urine_output"]; values[int(episode), int(step), index] = float(cleaned[0]); masks[int(episode), int(step), index] = True

    items = _read(paths["icu/d_items"])[["itemid", "label"]]
    items = _filter_required_ids(items, ("itemid",), "icu/d_items", audit)
    labels = items.set_index("itemid")["label"]
    label_ids = set(labels[labels.astype(str).str.contains(f"{DIURETIC_PATTERN}|{RRT_PATTERN}", case=False, regex=True, na=False)].index.astype(int))
    ib = _stream_events(
        paths["icu/inputevents"], table="icu/inputevents",
        columns=("stay_id", "starttime", "endtime", "itemid", "amount", "amountuom", "rate", "rateuom"),
        dates=("starttime", "endtime"), required_ids=("stay_id", "itemid"), required_times=("starttime",),
        audit=audit, chunk_rows=chunk_rows,
        itemids=set(FLUID_ITEMIDS) | set(VASO_ITEMIDS) | label_ids,
        key="stay_id", bounds=stay_bounds, interval_times=("starttime", "endtime"),
        sort_by=("episode_idx", "starttime", "itemid"),
    )
    ib["label"] = ib["itemid"].map(labels).fillna("")
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

    procedure_label_ids = set(labels[labels.astype(str).str.contains(RRT_PATTERN, case=False, regex=True, na=False)].index.astype(int))
    pb = _stream_events(
        paths["icu/procedureevents"], table="icu/procedureevents",
        columns=("stay_id", "starttime", "endtime", "itemid"),
        dates=("starttime", "endtime"), required_ids=("stay_id", "itemid"), required_times=("starttime",),
        audit=audit, chunk_rows=chunk_rows, itemids=set(MECHVENT_ITEMIDS) | procedure_label_ids,
        key="stay_id", bounds=stay_bounds, interval_times=("starttime", "endtime"),
        sort_by=("episode_idx", "starttime", "itemid"),
    )
    pb["label"] = pb["itemid"].map(labels).fillna("")
    pb["endtime"] = pb["endtime"].fillna(pb["starttime"])
    pb = pb[(pb["endtime"] > pb["window_start"]) & (pb["starttime"] < pb["window_end"]) & (pb["endtime"] > pb["starttime"])].copy()
    for row in pb.itertuples(index=False):
        episode = int(row.episode_idx); start = max(row.starttime, row.window_start); end = min(row.endtime, row.window_end)
        for step in overlap_bins(start, end, row.window_start, bin_hours=BIN_HOURS, n_steps=steps):
            if int(row.itemid) in MECHVENT_ITEMIDS:
                index = FEATURE_INDEX["mechanical_ventilation"]; values[episode, step, index] = 1; masks[episode, step, index] = True
            if re.search(RRT_PATTERN, str(row.label), re.I): rrt[episode, step] = 1

    rb = _stream_events(
        paths["hosp/prescriptions"], table="hosp/prescriptions",
        columns=("hadm_id", "starttime", "stoptime", "drug"),
        dates=("starttime", "stoptime"), required_ids=("hadm_id",), required_times=("starttime",),
        audit=audit, chunk_rows=chunk_rows,
        key="hadm_id", bounds=hadm_bounds, interval_times=("starttime", "stoptime"),
        sort_by=("episode_idx", "starttime"),
    )
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


def reconstruct(
    root: Path,
    output: Path,
    schema: Path,
    *,
    source_hashes: dict[str, str] | None = None,
    runtime_config: Path | None = None,
    chunk_rows: int | None = None,
) -> dict[str, Any]:
    from .runtime_config import load_runtime_config

    config = load_runtime_config(runtime_config)
    effective_chunk_rows = int(chunk_rows or config["runtime"]["high_volume_chunk_rows"])
    if effective_chunk_rows <= 0:
        raise ContractError("chunk_rows must be positive")
    if output.exists(): raise FileExistsError(output)
    paths = validate_layout(root)
    ingestion_audit: dict[tuple[str, str, str], int] = {}
    stays, diagnoses, time_audit = load_core_with_time_audit(paths, ingestion_audit)
    candidates = build_anchors(paths, stays, diagnoses, chunk_rows=effective_chunk_rows, audit=ingestion_audit)
    arrays = build_arrays(paths, candidates, chunk_rows=effective_chunk_rows, audit=ingestion_audit)
    candidates = finalize_sepsis_anchors(candidates, arrays); transitions = build_transitions(candidates)
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
    output.mkdir(parents=True)
    (output / "aggregate_receipt.json").write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    with (output / "icu_time_order_eligibility_aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("category", "count", "precedence"), lineterminator="\n")
        writer.writeheader()
        precedence = {
            "total_merged_icu_stays_considered": "denominator",
            "valid_time_order_stays": "after_all_invalid_categories",
            "missing_intime": "first",
            "missing_outtime": "second_only_when_intime_present",
            "equal_intime_outtime": "third_only_when_both_present",
            "reversed_time_order": "fourth_only_when_both_present",
            "retained_after_existing_public_base_eligibility": "after_time_filter_and_unchanged_existing_base_filters",
        }
        writer.writerows({"category": key, "count": value, "precedence": precedence[key]} for key, value in time_audit.items())
    with (output / "nullable_key_and_timestamp_exclusion_aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("table", "field", "reason", "excluded_count"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {"table": table, "field": field, "reason": reason, "excluded_count": count}
            for (table, field, reason), count in sorted(ingestion_audit.items())
        )
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
