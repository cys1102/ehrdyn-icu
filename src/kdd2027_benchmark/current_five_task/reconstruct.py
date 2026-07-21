from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import resource
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator
from kdd2027_benchmark.canonical import write_canonical_json

from .authoritative_semantics import (
    GCS_ITEMIDS,
    clean_feature_values,
    compute_corrected_derived_features,
    gcs_total_rows,
    overlap_bin_amounts,
)
from .runtime_config import RUNTIME_CONFIG
from .lineage_source_port import (
    blood_culture_events,
    compact_lineage_stays,
    sepsis_sofa_filter,
    kdd097_interval_bins,
    large_lineage_stays,
    match_suspected_infections,
)

from .contracts import (
    ANTIBIOTIC_PATTERN,
    BIN_HOURS,
    CHART_ITEM_MAP,
    CREATININE_ITEMIDS,
    DIURETIC_PATTERN,
    FEATURE_INDEX,
    FEATURE_NAMES,
    SAFE_FEATURE_INDICES,
    LEGACY_ACTION_FIO2_ITEMIDS,
    SAFE_STATE_FIO2_ITEMIDS,
    FLUID_ITEMIDS,
    LAB_ITEM_MAP,
    MECHVENT_ITEMIDS,
    MBP_ITEMIDS,
    PEEP_ITEMIDS,
    POST_ANCHOR_HOURS,
    PRE_ANCHOR_HOURS,
    RAW_EXTRACTION_BINS,
    RELEASE,
    ROLE_ORDER,
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
    fit_train_positive_edges,
    joint_codes,
    legacy_action_fio2_value,
    legacy_action_peep_value,
    parse_times,
    pre_repair_chart_value,
    reward_components,
    subject_role,
    validate_layout,
)


def _read(path: Path, columns: Iterable[str] | None = None, dates: Iterable[str] = ()) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=list(columns) if columns else None, low_memory=False)
    return parse_times(frame, dates, path.name)


HIGH_VOLUME_TABLE_ORDER = (
    "hosp/labevents",
    "hosp/microbiologyevents",
    "hosp/prescriptions",
    "icu/chartevents",
    "icu/inputevents",
    "icu/outputevents",
    "icu/procedureevents",
)
HIGH_VOLUME_TABLES = frozenset(HIGH_VOLUME_TABLE_ORDER)
DEFAULT_CHUNK_ROWS = int(RUNTIME_CONFIG["runtime"]["high_volume_chunk_rows"])


def _streaming_rows(
    audit: dict[str, dict[str, int | str]],
    *,
    chunk_rows: int,
    paths: dict[str, Path] | None = None,
) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for table in HIGH_VOLUME_TABLE_ORDER:
        observed = audit.get(table, {})
        encoding = observed.get("compression_encoding")
        if encoding is None and paths is not None:
            encoding = "csv_gz" if paths[table].suffix == ".gz" else "csv"
        rows.append({
            "table": table,
            "rows_read": int(observed.get("rows_read", 0)),
            "rows_retained": int(observed.get("rows_retained", 0)),
            "chunks_processed": int(observed.get("chunks_processed", 0)),
            "maximum_retained_rows_per_chunk": int(observed.get("maximum_retained_rows_per_chunk", 0)),
            "effective_chunk_size": int(observed.get("effective_chunk_size", chunk_rows)),
            "compression_encoding": str(encoding or "not_scanned"),
            "scan_count": int(observed.get("scan_count", 0)),
        })
    return rows


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
    streaming_audit: dict[str, dict[str, int | str]] | None = None,
) -> pd.DataFrame:
    if table not in HIGH_VOLUME_TABLES:
        raise ContractError(f"unregistered high-volume table: {table}")
    if chunk_rows <= 0:
        raise ContractError("chunk_rows must be positive")
    stream_row: dict[str, int | str] | None = None
    if streaming_audit is not None:
        encoding = "csv_gz" if path.suffix == ".gz" else "csv"
        stream_row = streaming_audit.setdefault(table, {
            "rows_read": 0,
            "rows_retained": 0,
            "chunks_processed": 0,
            "maximum_retained_rows_per_chunk": 0,
            "effective_chunk_size": int(chunk_rows),
            "compression_encoding": encoding,
            "scan_count": 0,
        })
        if stream_row["effective_chunk_size"] != int(chunk_rows) or stream_row["compression_encoding"] != encoding:
            raise ContractError(f"streaming contract drift for {table}")
        stream_row["scan_count"] = int(stream_row["scan_count"]) + 1
    selected_frames: list[pd.DataFrame] = []
    source_offset = 0
    chunks: Any | None = None
    source: Any | None = None
    try:
        source = gzip.open(path, "rt", encoding="utf-8", newline="") if path.suffix == ".gz" else path.open("r", encoding="utf-8", newline="")
        chunks = pd.read_csv(source, usecols=list(columns), chunksize=chunk_rows, low_memory=False)
        try:
            for raw_chunk in chunks:
                if stream_row is not None:
                    stream_row["chunks_processed"] = int(stream_row["chunks_processed"]) + 1
                    stream_row["rows_read"] = int(stream_row["rows_read"]) + len(raw_chunk)
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
                    if stream_row is not None:
                        stream_row["rows_retained"] = int(stream_row["rows_retained"]) + len(chunk)
                        stream_row["maximum_retained_rows_per_chunk"] = max(
                            int(stream_row["maximum_retained_rows_per_chunk"]), len(chunk)
                        )
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
    streaming_audit: dict[str, dict[str, int | str]] | None = None,
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
        streaming_audit=streaming_audit,
    )
    if creatinine.empty:
        return pd.DataFrame(columns=("subject_id", "hadm_id", "charttime", "value", "__source_order"))
    creatinine["value"] = pd.to_numeric(creatinine["valuenum"], errors="coerce")
    invalid_value = ~np.isfinite(creatinine["value"]) | creatinine["value"].le(0)
    _audit_add(audit, "hosp/labevents", "valuenum", "missing_nonfinite_or_nonpositive_creatinine", int(invalid_value.sum()))
    creatinine = creatinine.loc[~invalid_value, ["subject_id", "hadm_id", "charttime", "value", "__source_order"]]
    return creatinine.sort_values(["subject_id", "charttime", "__source_order"], kind="stable").reset_index(drop=True)


def kdigo_creatinine_events(creatinine: pd.DataFrame) -> pd.DataFrame:
    qualifying: list[dict[str, Any]] = []
    h48 = pd.Timedelta(hours=48).value
    d7 = pd.Timedelta(days=7).value
    current_subject: int | None = None
    q48: deque[tuple[int, float]] = deque()
    q7: deque[tuple[int, float]] = deque()
    for row in creatinine.itertuples(index=False):
        subject = int(row.subject_id)
        admission = int(row.hadm_id)
        when = int(pd.Timestamp(row.charttime).value)
        value = float(row.value)
        if subject != current_subject:
            current_subject = subject
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
    base = compact_lineage_stays(base)
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
    streaming_audit: dict[str, dict[str, int | str]] | None = None,
) -> pd.DataFrame:
    prescriptions = _stream_events(
        paths["hosp/prescriptions"], table="hosp/prescriptions",
        columns=("subject_id", "hadm_id", "starttime", "stoptime", "drug"),
        dates=("starttime", "stoptime"), required_ids=("subject_id", "hadm_id"), required_times=("starttime",),
        audit=audit, chunk_rows=chunk_rows,
        key="hadm_id", eligible_keys=set(stays["hadm_id"].astype(int)), sort_by=("subject_id", "starttime"),
        streaming_audit=streaming_audit,
    )
    antibiotics = prescriptions[
        prescriptions["hadm_id"].isin(stays["hadm_id"])
        & prescriptions["starttime"].notna()
        & prescriptions["drug"].astype(str).str.contains(ANTIBIOTIC_PATTERN, case=False, regex=True, na=False)
    ].copy()
    antibiotics = antibiotics.merge(stays[["subject_id", "hadm_id", "stay_id"]], on=["subject_id", "hadm_id"], how="inner")
    antibiotics = antibiotics.rename(columns={"starttime": "antibiotic_time"})
    cultures = _stream_events(
        paths["hosp/microbiologyevents"], table="hosp/microbiologyevents",
        columns=("micro_specimen_id", "subject_id", "hadm_id", "chartdate", "charttime", "spec_type_desc", "org_itemid", "org_name"),
        dates=("chartdate", "charttime"), required_ids=("subject_id",), required_times=(),
        audit=audit, chunk_rows=chunk_rows,
        key="subject_id", eligible_keys=set(stays["subject_id"].astype(int)), sort_by=("subject_id", "charttime", "chartdate"),
        streaming_audit=streaming_audit,
    )
    cultures = blood_culture_events(cultures)
    missing_culture_time = cultures["culture_time"].isna()
    _audit_add(audit, "hosp/microbiologyevents", "charttime_or_chartdate", "missing_required_event_timestamp", int(missing_culture_time.sum()))
    cultures = cultures[~missing_culture_time].sort_values(
        ["subject_id", "culture_time", "micro_specimen_id"], kind="stable"
    )
    matches = match_suspected_infections(antibiotics, cultures)
    if matches.empty:
        return _first_anchor(pd.DataFrame(columns=("stay_id", "anchor_time")), "E060_suspected_infection_plus_SOFA_ge2_scaffold")
    matches = matches.rename(columns={"suspected_infection_time": "anchor_time"})
    return _first_anchor(matches[["stay_id", "anchor_time"]], "E060_suspected_infection_plus_SOFA_ge2_scaffold")


def _interval_filter(events: pd.DataFrame, stays: pd.DataFrame, time: str) -> pd.DataFrame:
    merged = events.merge(stays[["stay_id", "intime", "outtime"]], on="stay_id", how="inner")
    return merged[(merged[time] >= merged["intime"]) & (merged[time] < merged["outtime"])].copy()


def sustained_hypotension_from_chart(
    chart: pd.DataFrame,
    stays: pd.DataFrame,
    *,
    authoritative_chunk_rows: int,
) -> pd.DataFrame:
    """Reproduce the frozen KDD152 logical source-chunk grouping."""
    bp = chart[chart["itemid"].isin(SBP_ITEMIDS + MBP_ITEMIDS)].dropna(subset=["charttime"]).copy()
    bp["value"] = pd.to_numeric(bp["valuenum"], errors="coerce")
    bp = _interval_filter(bp, stays, "charttime")
    bp["relative_bin"] = np.floor((bp["charttime"] - bp["intime"]).dt.total_seconds() / (BIN_HOURS * 3600)).astype(int)
    bp["authoritative_source_chunk"] = bp["__source_order"].astype(np.int64) // int(authoritative_chunk_rows)
    low = ((bp["itemid"].isin(SBP_ITEMIDS)) & bp["value"].lt(90)) | ((bp["itemid"].isin(MBP_ITEMIDS)) & bp["value"].lt(65))
    bp = bp[low]
    grouped = bp.groupby(
        ["authoritative_source_chunk", "stay_id", "relative_bin"], observed=True
    ).agg(event_count=("charttime", "size"), anchor_time=("charttime", "min")).reset_index()
    grouped = grouped[grouped["event_count"].ge(2)][["stay_id", "anchor_time"]]
    grouped["anchor_source"] = "sustained_hypotension"
    return _first_anchor(grouped)


def build_anchors(
    paths: dict[str, Path],
    stays: pd.DataFrame,
    diagnoses: pd.DataFrame,
    *,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    audit: dict[tuple[str, str, str], int] | None = None,
    streaming_audit: dict[str, dict[str, int | str]] | None = None,
) -> pd.DataFrame:
    audit = audit if audit is not None else {}
    compact_stays = compact_lineage_stays(stays)
    large_stays = large_lineage_stays(stays)
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
        streaming_audit=streaming_audit,
    )
    input_label_ids = set(label[label.astype(str).str.contains(f"{DIURETIC_PATTERN}|{VASODILATOR_PATTERN}|{RRT_PATTERN}", case=False, regex=True, na=False)].index.astype(int))
    inputs = _stream_events(
        paths["icu/inputevents"], table="icu/inputevents",
        columns=("stay_id", "starttime", "endtime", "itemid", "amount", "rate"),
        dates=("starttime", "endtime"), required_ids=("stay_id", "itemid"), required_times=("starttime",),
        audit=audit, chunk_rows=chunk_rows,
        itemids=set(VASO_ITEMIDS) | input_label_ids,
        key="stay_id", eligible_keys=stay_ids, sort_by=("stay_id", "starttime", "itemid"),
        streaming_audit=streaming_audit,
    )
    procedure_label_ids = set(label[label.astype(str).str.contains(f"oxygen|ventilat|intubat|bipap|cpap|{RRT_PATTERN}", case=False, regex=True, na=False)].index.astype(int))
    procedures = _stream_events(
        paths["icu/procedureevents"], table="icu/procedureevents",
        columns=("stay_id", "starttime", "endtime", "itemid"),
        dates=("starttime", "endtime"), required_ids=("stay_id", "itemid"), required_times=("starttime",),
        audit=audit, chunk_rows=chunk_rows,
        itemids=set(MECHVENT_ITEMIDS) | procedure_label_ids,
        key="stay_id", eligible_keys=stay_ids, sort_by=("stay_id", "starttime", "itemid"),
        streaming_audit=streaming_audit,
    )
    frames: list[pd.DataFrame] = []

    sepsis = suspected_infection_anchors(
        paths, large_stays, chunk_rows=chunk_rows, audit=audit, streaming_audit=streaming_audit
    )
    if not sepsis.empty:
        local = large_stays.merge(sepsis, on="stay_id", how="inner")
        local = local[
            (local["anchor_time"] + pd.Timedelta(hours=POST_ANCHOR_HOURS) > local["intime"])
            & (local["anchor_time"] - pd.Timedelta(hours=PRE_ANCHOR_HOURS) < local["outtime"])
        ]
        local["task_id"] = "sepsis"
        frames.append(local)

    respiratory = _interval_filter(
        chart[chart["itemid"].isin(MECHVENT_ITEMIDS)].dropna(subset=["charttime"]), compact_stays, "charttime"
    )[["stay_id", "charttime"]].rename(columns={"charttime": "anchor_time"}); respiratory["anchor_source"] = "structured_ventilation_chart_event"
    p = procedures.copy(); p["label"] = p["itemid"].map(label).fillna("")
    support = p[p["label"].astype(str).str.contains(r"oxygen|ventilat|intubat|bipap|cpap", case=False, regex=True)]
    support = _interval_filter(support, compact_stays, "starttime")[["stay_id", "starttime"]].rename(columns={"starttime": "anchor_time"}); support["anchor_source"] = "structured_respiratory_support_event"
    respiratory = _first_anchor(pd.concat([respiratory, support], ignore_index=True))
    if not respiratory.empty:
        local = compact_stays.merge(respiratory, on="stay_id", how="inner"); local["task_id"] = "respiratory_support"; frames.append(local)

    bp = sustained_hypotension_from_chart(
        chart, stays,
        authoritative_chunk_rows=int(RUNTIME_CONFIG["runtime"]["authoritative_kdd152_source_chunk_rows"]),
    )
    vaso = inputs[inputs["itemid"].isin(VASO_ITEMIDS)].copy()
    positive = np.maximum(pd.to_numeric(vaso["amount"], errors="coerce").fillna(0), pd.to_numeric(vaso["rate"], errors="coerce").fillna(0)).gt(0)
    vaso = _interval_filter(vaso[positive], compact_stays, "starttime")[["stay_id", "starttime"]].rename(columns={"starttime": "anchor_time"}); vaso["anchor_source"] = "vasopressor_support"
    shock = _first_anchor(pd.concat([bp, vaso], ignore_index=True))
    if not shock.empty:
        local = compact_stays.merge(shock, on="stay_id", how="inner"); local["task_id"] = "shock"; frames.append(local)

    creat = scan_creatinine(
        paths["hosp/labevents"], chunk_rows=chunk_rows, audit=audit, streaming_audit=streaming_audit
    )
    aki_lab = kdigo_creatinine_events(creat)
    if not aki_lab.empty:
        aki_lab = aki_lab.merge(compact_stays[["hadm_id", "stay_id", "intime", "outtime"]], on="hadm_id", how="inner")
        aki_lab = aki_lab[(aki_lab["anchor_time"] >= aki_lab["intime"]) & (aki_lab["anchor_time"] < aki_lab["outtime"])][["stay_id", "anchor_time", "anchor_source"]]
    p["label"] = p["itemid"].map(label).fillna("")
    rrt = p[p["label"].astype(str).str.contains(RRT_PATTERN, case=False, regex=True)]
    rrt = _interval_filter(rrt, compact_stays, "starttime")[["stay_id", "starttime"]].rename(columns={"starttime": "anchor_time"}); rrt["anchor_source"] = "time_stamped_rrt_start"
    aki = _first_anchor(pd.concat([aki_lab, rrt], ignore_index=True))
    if not aki.empty:
        local = compact_stays.merge(aki, on="stay_id", how="inner"); local["task_id"] = "aki"; frames.append(local)

    hf_stays = large_stays.copy()
    rx = _stream_events(
        paths["hosp/prescriptions"], table="hosp/prescriptions",
        columns=("subject_id", "hadm_id", "starttime", "stoptime", "drug"),
        dates=("starttime", "stoptime"), required_ids=("subject_id", "hadm_id"), required_times=("starttime",),
        audit=audit, chunk_rows=chunk_rows,
        key="hadm_id", eligible_keys=set(hf_stays["hadm_id"].astype(int)), sort_by=("hadm_id", "starttime"),
        streaming_audit=streaming_audit,
    )
    rx = rx[rx["drug"].astype(str).str.contains(f"{DIURETIC_PATTERN}|{VASODILATOR_PATTERN}", case=False, regex=True, na=False)]
    hf_event = rx.merge(hf_stays[["hadm_id", "stay_id", "intime", "outtime"]], on="hadm_id")
    hf_event = hf_event[(hf_event["starttime"] >= hf_event["intime"]) & (hf_event["starttime"] < hf_event["outtime"])][["stay_id", "starttime"]].rename(columns={"starttime": "anchor_time"}); hf_event["anchor_source"] = "current_stay_decongestion_prescription"
    i = inputs.copy(); i["label"] = i["itemid"].map(label).fillna("")
    positive = np.maximum(pd.to_numeric(i["amount"], errors="coerce").fillna(0), pd.to_numeric(i["rate"], errors="coerce").fillna(0)).gt(0)
    hf_input = i[positive & i["label"].astype(str).str.contains(f"{DIURETIC_PATTERN}|{VASODILATOR_PATTERN}", case=False, regex=True)]
    hf_input = _interval_filter(hf_input, hf_stays, "starttime")[["stay_id", "starttime"]].rename(columns={"starttime": "anchor_time"}); hf_input["anchor_source"] = "current_stay_decongestion_input"
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
        anchor + pd.Timedelta(hours=(RAW_EXTRACTION_BINS * BIN_HOURS - PRE_ANCHOR_HOURS) if task in {"respiratory_support", "shock", "aki"} else POST_ANCHOR_HOURS)
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


def respiratory_chart_surfaces(chart: pd.DataFrame) -> pd.DataFrame:
    """Return independent legacy-action and repaired SAFE-state chart values."""
    required = {"itemid", "valuenum", "valueuom"}
    if not required.issubset(chart.columns):
        raise ContractError("respiratory chart fixture lacks itemid, valuenum, or valueuom")
    output = chart.copy()
    output["legacy_peep_value"] = [
        legacy_action_peep_value(item, value)
        for item, value in zip(output["itemid"], output["valuenum"])
    ]
    output["legacy_fio2_value"] = [
        legacy_action_fio2_value(item, value)
        for item, value in zip(output["itemid"], output["valuenum"])
    ]
    output["safe_fio2_value"] = [
        corrected_chart_value(item, value, unit) if int(item) in SAFE_STATE_FIO2_ITEMIDS else math.nan
        for item, value, unit in zip(output["itemid"], output["valuenum"], output["valueuom"])
    ]
    return output


def aggregate_respiratory_action_bins(chart: pd.DataFrame) -> pd.DataFrame:
    parsed = respiratory_chart_surfaces(chart)
    required = {"episode_idx", "bin"}
    if not required.issubset(parsed.columns):
        raise ContractError("respiratory chart aggregation lacks episode_idx or bin")
    rows: list[dict[str, Any]] = []
    for (episode, step), group in parsed.groupby(["episode_idx", "bin"], sort=True, observed=True):
        peep = group["legacy_peep_value"].to_numpy(float)
        fio2 = group["legacy_fio2_value"].to_numpy(float)
        safe = group["safe_fio2_value"].to_numpy(float)
        rows.append({
            "episode_idx": int(episode), "bin": int(step),
            "legacy_peep": float(np.median(peep[np.isfinite(peep)])) if np.isfinite(peep).any() else math.nan,
            "legacy_peep_observed": bool(np.isfinite(peep).any()),
            "legacy_fio2": float(np.median(fio2[np.isfinite(fio2)])) if np.isfinite(fio2).any() else math.nan,
            "legacy_fio2_observed": bool(np.isfinite(fio2).any()),
            "safe_fio2": float(np.median(safe[np.isfinite(safe)])) if np.isfinite(safe).any() else math.nan,
            "safe_fio2_observed": bool(np.isfinite(safe).any()),
        })
    return pd.DataFrame(rows).sort_values(["episode_idx", "bin"], kind="stable").reset_index(drop=True)


def build_arrays(
    paths: dict[str, Path],
    candidates: pd.DataFrame,
    *,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    audit: dict[tuple[str, str, str], int] | None = None,
    streaming_audit: dict[str, dict[str, int | str]] | None = None,
) -> dict[str, np.ndarray]:
    audit = audit if audit is not None else {}
    steps = RAW_EXTRACTION_BINS
    shape = (len(candidates), steps, len(FEATURE_NAMES))
    values = np.full(shape, np.nan, dtype=np.float32); masks = np.zeros(shape, dtype=bool)
    pre_repair_values = np.full(shape, np.nan, dtype=np.float32)
    pre_repair_masks = np.zeros(shape, dtype=bool)
    fluid = np.zeros(shape[:2], dtype=np.float32); vaso = np.zeros(shape[:2], dtype=np.float32)
    peep = np.full(shape[:2], np.nan, dtype=np.float32); peep_observed = np.zeros(shape[:2], dtype=bool)
    fio2_action = np.full(shape[:2], np.nan, dtype=np.float32); fio2_action_observed = np.zeros(shape[:2], dtype=bool)
    diuretic = np.zeros(shape[:2], dtype=np.float32); rrt = np.zeros(shape[:2], dtype=np.float32)
    for row in candidates.itertuples(index=False):
        values[row.episode_idx, :, FEATURE_INDEX["age"]] = float(row.anchor_age)
        values[row.episode_idx, :, FEATURE_INDEX["gender_male"]] = float(str(row.gender).upper() == "M")
        values[row.episode_idx, :, FEATURE_INDEX["step_id"]] = np.arange(steps)
        masks[row.episode_idx, :, [FEATURE_INDEX["age"], FEATURE_INDEX["gender_male"], FEATURE_INDEX["step_id"]]] = True
        pre_repair_values[row.episode_idx, :, :] = values[row.episode_idx, :, :]
        pre_repair_masks[row.episode_idx, :, :] = masks[row.episode_idx, :, :]

    stay_bounds = candidates[["stay_id", "episode_idx", "window_start", "window_end"]]
    chart = _stream_events(
        paths["icu/chartevents"], table="icu/chartevents",
        columns=("stay_id", "charttime", "itemid", "valuenum", "valueuom"),
        dates=("charttime",), required_ids=("stay_id", "itemid"), required_times=("charttime",),
        audit=audit, chunk_rows=chunk_rows,
        itemids=set(CHART_ITEM_MAP) | set(PEEP_ITEMIDS) | set(LEGACY_ACTION_FIO2_ITEMIDS) | set(MECHVENT_ITEMIDS),
        key="stay_id", bounds=stay_bounds, point_time="charttime",
        sort_by=("episode_idx", "charttime", "itemid"),
        streaming_audit=streaming_audit,
    )
    chart["bin"] = np.floor((chart["charttime"] - chart["window_start"]).dt.total_seconds() / (BIN_HOURS * 3600)).astype(int)
    chart = respiratory_chart_surfaces(chart)
    chart["value"] = [corrected_chart_value(item, value, unit) for item, value, unit in zip(chart["itemid"], chart["valuenum"], chart["valueuom"])]
    chart["pre_repair_value"] = [pre_repair_chart_value(item, value) for item, value in zip(chart["itemid"], chart["valuenum"])]
    valid_chart = chart[np.isfinite(chart["value"])].copy()
    action_bins = aggregate_respiratory_action_bins(chart)
    for row in action_bins.itertuples(index=False):
        episode, step = int(row.episode_idx), int(row.bin)
        if row.legacy_peep_observed:
            peep[episode, step] = float(row.legacy_peep); peep_observed[episode, step] = True
        if row.legacy_fio2_observed:
            fio2_action[episode, step] = float(row.legacy_fio2); fio2_action_observed[episode, step] = True
    gcs = valid_chart[valid_chart["itemid"].isin(GCS_ITEMIDS)].copy()
    gcs["feature_value"] = gcs["value"]
    for row in gcs_total_rows(gcs).itertuples(index=False):
        index = FEATURE_INDEX["gcs_proxy"]; values[int(row.episode_idx), int(row.bin), index] = float(row.feature_value); masks[int(row.episode_idx), int(row.bin), index] = True
    pre_gcs = chart[np.isfinite(chart["pre_repair_value"]) & chart["itemid"].isin(GCS_ITEMIDS)].copy()
    pre_gcs["feature_value"] = pre_gcs["pre_repair_value"]
    for row in gcs_total_rows(pre_gcs).itertuples(index=False):
        index = FEATURE_INDEX["gcs_proxy"]; pre_repair_values[int(row.episode_idx), int(row.bin), index] = float(row.feature_value); pre_repair_masks[int(row.episode_idx), int(row.bin), index] = True
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
    pre_generic = chart[np.isfinite(chart["pre_repair_value"]) & ~chart["itemid"].isin(set(PEEP_ITEMIDS) | GCS_ITEMIDS)].copy()
    pre_generic["feature"] = pre_generic["itemid"].astype(int).map(CHART_ITEM_MAP).fillna(
        pre_generic["itemid"].astype(int).map({item: "fio2" for item in LEGACY_ACTION_FIO2_ITEMIDS})
    )
    pre_generic["feature"] = pre_generic["feature"].fillna(
        pre_generic["itemid"].astype(int).map({item: "mechanical_ventilation" for item in MECHVENT_ITEMIDS})
    )
    pre_generic = pre_generic[pre_generic["feature"].notna()]
    for (episode, step, feature), group in pre_generic.groupby(["episode_idx", "bin", "feature"], observed=True):
        finite = group["pre_repair_value"].to_numpy(float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            index = FEATURE_INDEX[str(feature)]
            pre_repair_values[int(episode), int(step), index] = float(np.median(finite))
            pre_repair_masks[int(episode), int(step), index] = True

    hadm_bounds = candidates[["hadm_id", "episode_idx", "window_start", "window_end"]]
    labs = _stream_events(
        paths["hosp/labevents"], table="hosp/labevents",
        columns=("hadm_id", "charttime", "itemid", "valuenum", "valueuom"),
        dates=("charttime",), required_ids=("hadm_id", "itemid"), required_times=("charttime",),
        audit=audit, chunk_rows=chunk_rows, itemids=set(LAB_ITEM_MAP),
        key="hadm_id", bounds=hadm_bounds, point_time="charttime",
        sort_by=("episode_idx", "charttime", "itemid"),
        streaming_audit=streaming_audit,
    )
    labs["bin"] = np.floor((labs["charttime"] - labs["window_start"]).dt.total_seconds() / (BIN_HOURS * 3600)).astype(int)
    labs["value"] = pd.to_numeric(labs["valuenum"], errors="coerce")
    labs = labs[np.isfinite(labs["value"]) & ~labs["value"].isin((-9999, -999, -99))]
    labs["feature"] = labs["itemid"].astype(int).map(LAB_ITEM_MAP)
    for (episode, step, feature), group in labs.groupby(["episode_idx", "bin", "feature"], observed=True):
        pre_group = group
        if feature in {"lactate", "ionized_calcium"}:
            group = group[group["valueuom"].astype("string").eq("mmol/L")]
        cleaned = clean_feature_values(str(feature), group["value"].to_numpy(float))
        finite = cleaned[np.isfinite(cleaned)]
        if finite.size:
            index = FEATURE_INDEX[str(feature)]; values[int(episode), int(step), index] = float(np.median(finite)); masks[int(episode), int(step), index] = True
        pre_cleaned = clean_feature_values(str(feature), pre_group["value"].to_numpy(float))
        pre_finite = pre_cleaned[np.isfinite(pre_cleaned)]
        if pre_finite.size:
            index = FEATURE_INDEX[str(feature)]
            pre_repair_values[int(episode), int(step), index] = float(np.median(pre_finite))
            pre_repair_masks[int(episode), int(step), index] = True

    outputs = _stream_events(
        paths["icu/outputevents"], table="icu/outputevents",
        columns=("stay_id", "charttime", "itemid", "value"),
        dates=("charttime",), required_ids=("stay_id", "itemid"), required_times=("charttime",),
        audit=audit, chunk_rows=chunk_rows, itemids=set(URINE_ITEMIDS),
        key="stay_id", bounds=stay_bounds, point_time="charttime",
        sort_by=("episode_idx", "charttime", "itemid"),
        streaming_audit=streaming_audit,
    )
    outputs["bin"] = np.floor((outputs["charttime"] - outputs["window_start"]).dt.total_seconds() / (BIN_HOURS * 3600)).astype(int)
    outputs["numeric"] = pd.to_numeric(outputs["value"], errors="coerce")
    for (episode, step), group in outputs[outputs["numeric"].ge(0)].groupby(["episode_idx", "bin"], observed=True):
        cleaned = clean_feature_values("urine_output", np.asarray([group["numeric"].sum()], dtype=float))
        if np.isfinite(cleaned[0]):
            index = FEATURE_INDEX["urine_output"]; values[int(episode), int(step), index] = float(cleaned[0]); masks[int(episode), int(step), index] = True
            pre_repair_values[int(episode), int(step), index] = float(cleaned[0]); pre_repair_masks[int(episode), int(step), index] = True

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
        streaming_audit=streaming_audit,
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
        for step in kdd097_interval_bins(start, end, row.window_start, bin_hours=BIN_HOURS, n_steps=steps):
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
        streaming_audit=streaming_audit,
    )
    pb["label"] = pb["itemid"].map(labels).fillna("")
    pb["endtime"] = pb["endtime"].fillna(pb["starttime"])
    pb = pb[(pb["endtime"] > pb["window_start"]) & (pb["starttime"] < pb["window_end"]) & (pb["endtime"] > pb["starttime"])].copy()
    for row in pb.itertuples(index=False):
        episode = int(row.episode_idx); start = max(row.starttime, row.window_start); end = min(row.endtime, row.window_end)
        for step in kdd097_interval_bins(start, end, row.window_start, bin_hours=BIN_HOURS, n_steps=steps):
            if int(row.itemid) in MECHVENT_ITEMIDS:
                index = FEATURE_INDEX["mechanical_ventilation"]; values[episode, step, index] = 1; masks[episode, step, index] = True
                pre_repair_values[episode, step, index] = 1; pre_repair_masks[episode, step, index] = True
            if re.search(RRT_PATTERN, str(row.label), re.I): rrt[episode, step] = 1

    rb = _stream_events(
        paths["hosp/prescriptions"], table="hosp/prescriptions",
        columns=("hadm_id", "starttime", "stoptime", "drug"),
        dates=("starttime", "stoptime"), required_ids=("hadm_id",), required_times=("starttime",),
        audit=audit, chunk_rows=chunk_rows,
        key="hadm_id", bounds=hadm_bounds, interval_times=("starttime", "stoptime"),
        sort_by=("episode_idx", "starttime"),
        streaming_audit=streaming_audit,
    )
    rb["stoptime"] = rb["stoptime"].fillna(rb["starttime"])
    rb = rb[(rb["stoptime"] >= rb["window_start"]) & (rb["starttime"] < rb["window_end"])].copy()
    for row in rb.itertuples(index=False):
        if re.search(DIURETIC_PATTERN, str(row.drug), re.I):
            start = max(row.starttime, row.window_start); end = min(max(row.stoptime, start + pd.Timedelta(minutes=1)), row.window_end)
            for step in kdd097_interval_bins(start, end, row.window_start, bin_hours=BIN_HOURS, n_steps=steps): diuretic[int(row.episode_idx), step] = 1

    compute_corrected_derived_features(values, masks, vaso, feature_index=FEATURE_INDEX, bin_hours=BIN_HOURS)
    compute_corrected_derived_features(pre_repair_values, pre_repair_masks, vaso, feature_index=FEATURE_INDEX, bin_hours=BIN_HOURS)
    return {"values": values, "masks": masks, "pre_repair_values": pre_repair_values, "pre_repair_masks": pre_repair_masks, "fluid": fluid, "vaso": vaso, "peep": peep, "peep_observed": peep_observed, "fio2_action": fio2_action, "fio2_action_observed": fio2_action_observed, "diuretic": diuretic, "rrt": rrt}


def finalize_sepsis_anchors(candidates: pd.DataFrame, arrays: dict[str, np.ndarray]) -> pd.DataFrame:
    return sepsis_sofa_filter(
        candidates,
        arrays["pre_repair_values"],
        arrays["pre_repair_masks"],
        sofa_index=FEATURE_INDEX["sofa_proxy"],
        minimum=float(RUNTIME_CONFIG["cohort_parameters"]["sepsis"]["maximum_observed_sofa_min"]),
    )


def build_transitions(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in candidates.itertuples(index=False):
        for relative in eligible_transition_indices(candidate.anchor_time, candidate.intime, candidate.outtime):
            action_start = candidate.anchor_time + relative * pd.Timedelta(hours=BIN_HOURS)
            state = int((action_start - pd.Timedelta(hours=BIN_HOURS) - candidate.window_start) / pd.Timedelta(hours=BIN_HOURS))
            action = int((action_start - candidate.window_start) / pd.Timedelta(hours=BIN_HOURS))
            target = int((action_start + pd.Timedelta(hours=BIN_HOURS) - candidate.window_start) / pd.Timedelta(hours=BIN_HOURS))
            rows.append({"task": candidate.task_id, "role": candidate.role, "episode_idx": int(candidate.episode_idx), "anchor_source": candidate.anchor_source, "relative_transition": relative, "state_idx": state, "action_idx": action, "target_idx": target})
    return pd.DataFrame(rows)


def encode_actions(task: str, transitions: pd.DataFrame, arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    episode = transitions["episode_idx"].to_numpy(int); step = transitions["action_idx"].to_numpy(int); roles = transitions["role"].astype(str).to_numpy()
    if task in {"sepsis", "shock"}:
        left, right = arrays["fluid"][episode, step], arrays["vaso"][episode, step]; lo = ro = np.ones(len(step), bool)
    elif task == "respiratory_support":
        left, lo = arrays["peep"][episode, step], arrays["peep_observed"][episode, step]
        right, ro = arrays["fio2_action"][episode, step], arrays["fio2_action_observed"][episode, step]
    elif task == "aki":
        return (arrays["diuretic"][episode, step] > 0).astype(np.int16) + 2 * (arrays["rrt"][episode, step] > 0).astype(np.int16), {"K": 4, "edges": []}
    elif task == "heart_failure":
        return (arrays["diuretic"][episode, step] > 0).astype(np.int16), {"K": 2, "edges": []}
    else: raise ContractError(f"unknown task: {task}")
    le, re_ = fit_train_positive_edges(left, lo, roles), fit_train_positive_edges(right, ro, roles)
    valid = lo & ro
    return joint_codes(encode_five_levels(left, lo, le), encode_five_levels(right, ro, re_), valid), {
        "K": 25,
        "edges": [le, re_],
        "valid_action_mask": valid,
    }


def filter_respiratory_action_transitions(
    transitions: pd.DataFrame,
    actions: np.ndarray,
    valid_action_mask: np.ndarray,
    *,
    action_count: int = 25,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, int]]:
    actions = np.asarray(actions, dtype=np.int16)
    valid_action_mask = np.asarray(valid_action_mask, dtype=bool)
    if len(transitions) != len(actions) or actions.shape != valid_action_mask.shape:
        raise ContractError("respiratory transition and action arrays differ in shape")
    if np.any(actions < -1) or np.any(actions >= action_count):
        raise ContractError("respiratory action class outside frozen K=25 range")
    if not np.array_equal(valid_action_mask, actions >= 0):
        raise ContractError("respiratory action mask and encoded missingness disagree")
    retained = transitions.loc[valid_action_mask].copy().reset_index(drop=True)
    retained_actions = actions[valid_action_mask]
    if retained.empty:
        raise ContractError("no valid transitions for respiratory_support")
    if np.any(retained_actions < 0) or np.any(retained_actions >= action_count):
        raise ContractError("retained respiratory action class outside frozen K=25 range")
    before_episodes = transitions["episode_idx"].nunique()
    after_episodes = retained["episode_idx"].nunique()
    return retained, retained_actions, {
        "candidate_transitions": int(len(transitions)),
        "retained_transitions": int(len(retained)),
        "excluded_missing_action_transitions": int((~valid_action_mask).sum()),
        "candidate_episodes": int(before_episodes),
        "retained_episodes": int(after_episodes),
        "excluded_empty_episodes": int(before_episodes - after_episodes),
    }


def filter_target_observed_transitions(
    transitions: pd.DataFrame,
    arrays: dict[str, np.ndarray],
) -> pd.DataFrame:
    if transitions.empty:
        return transitions.copy()
    episode = transitions["episode_idx"].to_numpy(dtype=int)
    target = transitions["target_idx"].to_numpy(dtype=int)
    observed = arrays["pre_repair_masks"][episode, target][:, SAFE_FEATURE_INDICES].astype(bool).any(axis=1)
    return transitions.loc[observed].reset_index(drop=True)


def apply_kdd201_to_frozen_actions(
    transitions: pd.DataFrame,
    actions: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, int]:
    """Subset rows and already-encoded actions with one identical frozen mask."""
    actions = np.asarray(actions, dtype=np.int16)
    if len(transitions) != len(actions):
        raise ContractError("KDD201 transition/action length mismatch")
    keep = np.concatenate([
        apply_kdd201_temporal_repair(source, [relative])
        for source, relative in zip(transitions["anchor_source"], transitions["relative_transition"])
    ]) if len(transitions) else np.zeros(0, dtype=bool)
    retained = transitions.loc[keep].reset_index(drop=True)
    retained_actions = actions[keep].copy()
    if not np.array_equal(retained_actions, actions[np.flatnonzero(keep)]):
        raise ContractError("KDD201 changed a frozen action class")
    return retained, retained_actions, int((~keep).sum())


def frozen_action_and_membership_pipeline(
    task: str,
    candidates: pd.DataFrame,
    arrays: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any], dict[str, Any]]:
    """KDD097 -> KDD152 -> KDD201 ordering without cutpoint refitting."""
    candidate = build_transitions(candidates)
    candidate = candidate[candidate["task"].eq(task)].reset_index(drop=True)
    if candidate.empty:
        raise ContractError(f"no candidate transitions for {task}")
    frozen_actions, action_contract = encode_actions(task, candidate, arrays)
    target_episode = candidate["episode_idx"].to_numpy(int)
    target_step = candidate["target_idx"].to_numpy(int)
    original_target = arrays["pre_repair_masks"][target_episode, target_step][:, SAFE_FEATURE_INDICES].any(axis=1)
    action_observed = frozen_actions >= 0
    if task == "respiratory_support":
        if not np.array_equal(action_observed, np.asarray(action_contract["valid_action_mask"], dtype=bool)):
            raise ContractError("respiratory action mask differs from frozen action classes")
    keep_kdd152 = original_target & action_observed
    membership = candidate.loc[keep_kdd152].reset_index(drop=True)
    actions = frozen_actions[keep_kdd152]
    pre_kdd201 = membership.copy()
    membership, actions, kdd201_removed = apply_kdd201_to_frozen_actions(membership, actions)
    if membership.empty:
        raise ContractError(f"no retained transitions for {task}")
    if np.any(actions < 0):
        raise ContractError(f"missing action observation for {task}")
    stages = {
        "candidate_transitions": int(len(candidate)),
        "original_target_membership_transitions": int(original_target.sum()),
        "action_observed_transitions": int(action_observed.sum()),
        "kdd152_joint_retained_transitions": int(keep_kdd152.sum()),
        "missing_action_exclusions": int((original_target & ~action_observed).sum()),
        "original_target_exclusions": int((~original_target).sum()),
        "kdd201_removed_transitions": kdd201_removed,
        "final_transitions": int(len(membership)),
    }
    if task == "respiratory_support":
        episode = candidate["episode_idx"].to_numpy(int)
        step = candidate["action_idx"].to_numpy(int)
        peep = arrays["peep_observed"][episode, step]
        fio2 = arrays["fio2_action_observed"][episode, step]
        stages.update({
            "peep_observed_transitions": int(peep.sum()),
            "legacy_fio2_observed_transitions": int(fio2.sum()),
            "joint_action_observed_transitions": int((peep & fio2).sum()),
        })
    role_counts: list[dict[str, Any]] = []
    for role in (*ROLE_ORDER, "all_roles"):
        role_candidate = np.ones(len(candidate), dtype=bool) if role == "all_roles" else candidate["role"].eq(role).to_numpy(bool)
        role_pre = np.ones(len(pre_kdd201), dtype=bool) if role == "all_roles" else pre_kdd201["role"].eq(role).to_numpy(bool)
        role_final = np.ones(len(membership), dtype=bool) if role == "all_roles" else membership["role"].eq(role).to_numpy(bool)
        role_row: dict[str, Any] = {
            "role": role,
            "candidate_transitions": int(role_candidate.sum()),
            "original_target_membership_transitions": int((role_candidate & original_target).sum()),
            "action_observed_transitions": int((role_candidate & action_observed).sum()),
            "kdd152_joint_retained_transitions": int((role_candidate & keep_kdd152).sum()),
            "kdd201_removed_transitions": int(role_pre.sum() - role_final.sum()),
            "final_transitions": int(role_final.sum()),
        }
        if task == "respiratory_support":
            role_row.update({
                "peep_observed_transitions": int((role_candidate & peep).sum()),
                "legacy_fio2_observed_transitions": int((role_candidate & fio2).sum()),
                "joint_action_observed_transitions": int((role_candidate & peep & fio2).sum()),
                "missing_action_transitions": int((role_candidate & ~(peep & fio2)).sum()),
            })
        role_counts.append(role_row)
    stages["role_counts"] = role_counts
    return membership, actions, action_contract, stages


def _array_digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    header = hashlib.sha256(json.dumps(
        {"shape": array.shape, "dtype": array.dtype.str},
        sort_keys=True, separators=(",", ":"), default=str,
    ).encode()).hexdigest().encode()
    digest = hashlib.sha256(header)
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _full_past_only_arrays(
    arrays: dict[str, np.ndarray], candidates: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    values = arrays["values"][:, :, SAFE_FEATURE_INDICES].astype(np.float32).copy()
    masks = arrays["masks"][:, :, SAFE_FEATURE_INDICES].astype(bool)
    filled = np.full_like(values, np.nan)
    recency = np.ones_like(values, dtype=np.float32)
    candidate = candidates.set_index("episode_idx")
    bin_step = pd.Timedelta(hours=BIN_HOURS)
    for episode in range(values.shape[0]):
        if episode not in candidate.index:
            continue
        row = candidate.loc[episode]
        last = np.full(values.shape[2], np.nan, dtype=np.float32)
        since = np.full(values.shape[2], 6, dtype=np.int16)
        for index in range(values.shape[1]):
            bin_start = row.window_start + index * bin_step
            if not (bin_start >= row.intime and (bin_start + bin_step) <= row.outtime):
                continue
            observed = masks[episode, index] & np.isfinite(values[episode, index])
            since = np.minimum(since + 1, 6)
            last[observed] = values[episode, index, observed]
            since[observed] = 0
            filled[episode, index] = last
            recency[episode, index] = since.astype(np.float32) / 6.0
    return filled, recency


def _role_surface(
    task: str,
    role: str,
    candidates: pd.DataFrame,
    transitions: pd.DataFrame,
    actions: np.ndarray,
    arrays: dict[str, np.ndarray],
    train_mean: np.ndarray,
    train_scale: np.ndarray,
    filled: np.ndarray,
    full_recency: np.ndarray,
) -> dict[str, Any]:
    select = np.ones(len(transitions), dtype=bool) if role == "all_roles" else transitions["role"].eq(role).to_numpy(bool)
    local_transitions = transitions.loc[select].reset_index(drop=True)
    local_actions = np.asarray(actions)[select]
    episode_ids = local_transitions["episode_idx"].drop_duplicates().to_numpy(int)
    local_candidates = candidates[candidates["episode_idx"].isin(episode_ids)]
    count = {"sepsis": 25, "respiratory_support": 25, "shock": 25, "aki": 4, "heart_failure": 2}[task]
    action_counts = np.bincount(local_actions, minlength=count).astype(int) if len(local_actions) else np.zeros(count, dtype=int)
    lengths = local_transitions.groupby("episode_idx", observed=True).size().to_numpy(int)
    groups = list(local_transitions.groupby("episode_idx", sort=False, observed=True))
    max_steps = max((len(group) for _, group in groups), default=0)
    shape = (len(groups), max_steps, len(SAFE_FEATURE_INDICES))
    state_values = np.full(shape, np.nan, dtype=np.float32)
    state_masks = np.zeros(shape, dtype=bool)
    recency = np.zeros(shape, dtype=np.float32)
    raw_imputed = np.full(shape, np.nan, dtype=np.float32)
    normalized = np.zeros(shape, dtype=np.float32)
    target_values = np.full(shape, np.nan, dtype=np.float32)
    target_masks = np.zeros(shape, dtype=bool)
    padded_actions = np.full((len(groups), max_steps), -1, dtype=np.int16)
    valid = np.zeros((len(groups), max_steps), dtype=bool)
    terminal = np.zeros((len(groups), max_steps), dtype=bool)
    order = np.full((len(groups), max_steps), -1, dtype=np.int16)
    reward = np.zeros((len(groups), max_steps, 1), dtype=np.float32)
    reward_mask = np.zeros_like(reward, dtype=bool)
    candidate_index = candidates.set_index("episode_idx")
    for row_index, (episode_id, group) in enumerate(groups):
        group = group.reset_index()
        length = len(group)
        episode = group["episode_idx"].to_numpy(int)
        state = group["state_idx"].to_numpy(int)
        target = group["target_idx"].to_numpy(int)
        state_values[row_index, :length] = arrays["values"][episode, state][:, SAFE_FEATURE_INDICES]
        state_masks[row_index, :length] = arrays["masks"][episode, state][:, SAFE_FEATURE_INDICES]
        target_values[row_index, :length] = arrays["values"][episode, target][:, SAFE_FEATURE_INDICES]
        target_masks[row_index, :length] = arrays["masks"][episode, target][:, SAFE_FEATURE_INDICES]
        local_imputed = filled[episode, state]
        raw_imputed[row_index, :length] = local_imputed
        local_imputed = np.where(np.isfinite(local_imputed), local_imputed, train_mean)
        recency[row_index, :length] = np.nan_to_num(full_recency[episode, state], nan=18.0, posinf=18.0, neginf=0.0)
        normalized[row_index, :length] = np.nan_to_num((local_imputed - train_mean) / train_scale, nan=0.0, posinf=0.0, neginf=0.0)
        padded_actions[row_index, :length] = local_actions[group["index"].to_numpy(int)]
        valid[row_index, :length] = True
        terminal[row_index, length - 1] = True
        order[row_index, :length] = group["relative_transition"].to_numpy(np.int16)
        if task in {"sepsis", "aki", "heart_failure"}:
            reward[row_index, length - 1, 0] = -1.0 if float(candidate_index.loc[int(episode_id), "mortality_90d"]) > 0.5 else 1.0
            reward_mask[row_index, length - 1, 0] = True
        elif task == "shock":
            j = list(SAFE_FEATURE_INDICES).index(FEATURE_INDEX["mbp"])
            reward_mask[row_index, :length, 0] = target_masks[row_index, :length, j]
            reward[row_index, :length, 0] = np.clip((target_values[row_index, :length, j] - 65.0) / 25.0, -1.0, 1.0)
        else:
            safe = list(SAFE_FEATURE_INDICES)
            s, m = safe.index(FEATURE_INDEX["spo2"]), safe.index(FEATURE_INDEX["mbp"])
            reward_mask[row_index, :length, 0] = target_masks[row_index, :length, s] & target_masks[row_index, :length, m]
            reward[row_index, :length, 0] = np.where((target_values[row_index, :length, s] >= 94) & (target_values[row_index, :length, s] <= 98), 1.0, -0.5)
            reward[row_index, :length, 0] += np.where((target_values[row_index, :length, m] >= 70) & (target_values[row_index, :length, m] <= 80), 1.0, -0.5)
    reward[~reward_mask] = 0.0
    continuation = valid & ~terminal
    transition_order = local_transitions[["relative_transition", "state_idx", "action_idx", "target_idx"]].to_numpy(np.int16) if len(local_transitions) else np.empty((0, 4), np.int16)
    return {
        "role": role,
        "subjects": int(local_candidates["subject_id"].nunique()),
        "episodes": int(len(episode_ids)),
        "decisions": int(len(local_transitions)),
        "action_counts": action_counts.tolist(),
        "minimum_horizon": int(lengths.min()) if len(lengths) else 0,
        "maximum_horizon": int(lengths.max()) if len(lengths) else 0,
        "digests": {
            "feature_digest": _array_digest(state_values), "mask_digest": _array_digest(state_masks),
            "delta_digest": _array_digest(recency), "raw_imputed_history_digest": _array_digest(raw_imputed),
            "imputed_history_digest": _array_digest(normalized), "action_digest": _array_digest(padded_actions),
            "reward_digest": _array_digest(reward), "reward_mask_digest": _array_digest(reward_mask),
            "termination_digest": _array_digest(terminal), "continuation_digest": _array_digest(continuation),
            "valid_step_digest": _array_digest(valid), "target_digest": _array_digest(target_values),
            "observed_target_mask_digest": _array_digest(target_masks), "episode_order_digest": _array_digest(order),
            "preprocessing_mean_digest": _array_digest(train_mean.astype(np.float32)),
            "preprocessing_scale_digest": _array_digest(train_scale.astype(np.float32)),
            "transition_order_digest": _array_digest(transition_order),
        },
    }


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
    train = transitions["role"].eq("train").to_numpy(bool)
    episode = transitions["episode_idx"].to_numpy(int)
    state = transitions["state_idx"].to_numpy(int)
    observed = arrays["values"][episode, state][:, SAFE_FEATURE_INDICES]
    observed_mask = arrays["masks"][episode, state][:, SAFE_FEATURE_INDICES]
    train_values = np.where(observed_mask[train], observed[train], 0.0)
    train_counts = observed_mask[train].sum(axis=0)
    train_mean = np.divide(train_values.sum(axis=0), train_counts, out=np.zeros(train_values.shape[1]), where=train_counts > 0)
    centered = np.where(observed_mask[train], observed[train] - train_mean, 0.0)
    train_scale = np.sqrt(np.divide((centered ** 2).sum(axis=0), train_counts, out=np.ones(train_values.shape[1]), where=train_counts > 0))
    train_scale = np.where(np.isfinite(train_scale) & (train_scale >= 1.0e-6), train_scale, 1.0)
    filled, full_recency = _full_past_only_arrays(arrays, candidates)
    role_summaries = [
        _role_surface(task, role, candidates, transitions, actions, arrays, train_mean, train_scale, filled, full_recency)
        for role in (*ROLE_ORDER, "all_roles")
    ]
    return {"subjects": int(local["subject_id"].nunique()), "episodes": int(len(episode_ids)), "decisions": int(len(transitions)), "action_counts": counts.astype(int).tolist(), "minimum_horizon": int(lengths.min()) if len(lengths) else 0, "maximum_horizon": int(lengths.max()) if len(lengths) else 0, "reward_contract": reward_contract, "reward_observed_decisions": reward_observed, "terminal_reward_count": terminal_rewards, "role_summaries": role_summaries}


def _runtime_resource_payload(started: float) -> dict[str, int | float]:
    return {
        "wall_seconds": round(max(0.0, time.perf_counter() - started), 6),
        "maximum_resident_set_size_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "temporary_disk_bytes": 0,
    }


def _write_streaming_instrumentation(output: Path, rows: list[dict[str, int | str]]) -> None:
    fields = (
        "table", "rows_read", "rows_retained", "chunks_processed",
        "maximum_retained_rows_per_chunk", "effective_chunk_size",
        "compression_encoding", "scan_count",
    )
    with (output / "streaming_instrumentation_aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_runtime_resource(output: Path, started: float, status: str) -> None:
    payload = {"status": status, **_runtime_resource_payload(started)}
    (output / "runtime_resource_aggregate.json").write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )


def _write_controlled_stop(
    output: Path,
    started: float,
    rows: list[dict[str, int | str]],
    error: BaseException,
    schema_directory: Path,
) -> None:
    if isinstance(error, ContractError):
        failure_code = "contract_error"
    elif isinstance(error, MemoryError):
        failure_code = "memory_error"
    else:
        failure_code = "runtime_error"
    payload = {
        "schema_version": "1.0.0",
        "release": RELEASE,
        "candidate_status": "controlled_stop",
        "failure_code": failure_code,
        "runtime": _runtime_resource_payload(started),
        "streaming": rows,
        "privacy": {
            "aggregate_only": True,
            "row_level_fields_exported": False,
            "private_paths_exported": False,
        },
    }
    controlled_schema = schema_directory / "credentialed_controlled_stop_receipt.schema.json"
    schema_object = json.loads(controlled_schema.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema_object)
    Draft202012Validator(schema_object).validate(payload)
    output.mkdir(parents=True, exist_ok=True)
    (output / "controlled_stop_receipt.json").write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    _write_streaming_instrumentation(output, rows)
    _write_runtime_resource(output, started, "controlled_stop")


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
    started = time.perf_counter()
    paths: dict[str, Path] | None = None
    streaming_audit: dict[str, dict[str, int | str]] = {}
    try:
        paths = validate_layout(root)
        ingestion_audit: dict[tuple[str, str, str], int] = {}
        stays, diagnoses, time_audit = load_core_with_time_audit(paths, ingestion_audit)
        candidates = build_anchors(
            paths, stays, diagnoses, chunk_rows=effective_chunk_rows,
            audit=ingestion_audit, streaming_audit=streaming_audit,
        )
        arrays = build_arrays(
            paths, candidates, chunk_rows=effective_chunk_rows,
            audit=ingestion_audit, streaming_audit=streaming_audit,
        )
        candidates = finalize_sepsis_anchors(candidates, arrays)
        rows: dict[str, dict[str, Any]] = {}
        respiratory_filter: dict[str, int] | None = None
        for task in TASKS:
            local, actions, contract, stages = frozen_action_and_membership_pipeline(
                task, candidates, arrays
            )
            if task == "respiratory_support":
                respiratory_filter = {
                    "candidate_transitions": stages["candidate_transitions"],
                    "retained_transitions": stages["final_transitions"],
                    "excluded_missing_action_transitions": stages["missing_action_exclusions"],
                    "candidate_episodes": int(build_transitions(candidates)[lambda x: x["task"].eq(task)]["episode_idx"].nunique()),
                    "retained_episodes": int(local["episode_idx"].nunique()),
                    "excluded_empty_episodes": int(build_transitions(candidates)[lambda x: x["task"].eq(task)]["episode_idx"].nunique() - local["episode_idx"].nunique()),
                }
            row = task_aggregate(task, candidates, local, actions, arrays)
            row["action_count"] = contract["K"]
            train_actions = actions[local["role"].eq("train").to_numpy(bool)]
            train_support = np.bincount(train_actions, minlength=contract["K"]) > 0
            row["support_mask_digest"] = _array_digest(train_support)
            row["transition_stages"] = stages
            row["cutpoints"] = [list(edge) for edge in contract["edges"]]
            row["cutpoint_hash"] = hashlib.sha256(
                json.dumps(contract["edges"], sort_keys=True).encode()
            ).hexdigest()
            rows[task] = row
        if respiratory_filter is None:
            raise ContractError("respiratory action filter did not execute")
        streaming_rows = _streaming_rows(streaming_audit, chunk_rows=effective_chunk_rows, paths=paths)
        receipt = aggregate_receipt(rows, source_hashes or {}, streaming_rows)
        schema_object = json.loads(schema.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema_object)
        Draft202012Validator(schema_object).validate(receipt)
        output.mkdir(parents=True)
        write_canonical_json(output / "aggregate_receipt.json", receipt)
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
        with (output / "respiratory_action_filter_aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(respiratory_filter), lineterminator="\n")
            writer.writeheader()
            writer.writerow(respiratory_filter)
        _write_streaming_instrumentation(output, streaming_rows)
        _write_runtime_resource(output, started, "complete")
        return receipt
    except Exception as error:
        streaming_rows = _streaming_rows(
            streaming_audit, chunk_rows=effective_chunk_rows, paths=paths
        )
        try:
            _write_controlled_stop(output, started, streaming_rows, error, schema.parent)
        except Exception as receipt_error:
            raise ContractError("controlled stop receipt failure") from receipt_error
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the current five-task aggregate receipt from MIMIC-IV 3.1 flat files")
    parser.add_argument("--mimiciv-root", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path)
    args = parser.parse_args()
    os.environ.setdefault("TZ", "UTC")
    reconstruct(args.mimiciv_root, args.output, args.schema, runtime_config=args.runtime_config)


if __name__ == "__main__":
    main()
