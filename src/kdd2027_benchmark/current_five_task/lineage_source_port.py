from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


HOURS_NS = 3_600_000_000_000


@dataclass(frozen=True, slots=True)
class SuspectedInfectionMatch:
    subject_id: int
    hadm_id: int
    stay_id: int
    antibiotic_time: pd.Timestamp
    culture_time: pd.Timestamp
    suspected_infection_time: pd.Timestamp


def compact_lineage_stays(stays: pd.DataFrame) -> pd.DataFrame:
    """KDD097 eligibility after ICU time-order validation."""
    return stays.loc[stays["anchor_age"].ge(18)].copy()


def large_lineage_stays(stays: pd.DataFrame, *, minimum_hours: float = 24.0) -> pd.DataFrame:
    """KDD121 adult-stay eligibility without hospice or diagnosis gates."""
    stay_hours = (stays["outtime"] - stays["intime"]).dt.total_seconds() / 3600.0
    keep = (
        stays["anchor_age"].ge(18)
        & stays["anchor_age"].notna()
        & stays["gender"].notna()
        & stays["dischtime"].notna()
        & stay_hours.ge(minimum_hours)
    )
    output = stays.loc[keep].copy()
    output["stay_hours"] = stay_hours.loc[keep]
    return output


def blood_culture_events(events: pd.DataFrame) -> pd.DataFrame:
    """Port of the frozen sepsis culture reader after streamed filtering."""
    if events.empty:
        return pd.DataFrame(columns=(
            "micro_specimen_id", "subject_id", "hadm_id", "culture_time",
            "spec_type_desc", "org_itemid", "org_name", "positive_culture",
        ))
    selected = events[
        events["spec_type_desc"].astype(str).str.contains("blood", case=False, na=False)
    ].copy()
    if selected.empty:
        return blood_culture_events(events.iloc[0:0])
    grouped = (
        selected.groupby("micro_specimen_id", dropna=False)
        .agg(
            subject_id=("subject_id", "max"),
            hadm_id=("hadm_id", "max"),
            chartdate=("chartdate", "max"),
            charttime=("charttime", "max"),
            spec_type_desc=("spec_type_desc", "max"),
            org_itemid=("org_itemid", "max"),
            org_name=("org_name", "max"),
        )
        .reset_index()
    )
    grouped["culture_time"] = grouped["charttime"].fillna(grouped["chartdate"])
    positive = grouped["org_name"].notna() & grouped["org_name"].astype(str).ne("")
    grouped["positive_culture"] = (positive & grouped["org_itemid"].ne(90856)).astype(np.int8)
    return grouped.loc[grouped["culture_time"].notna()].sort_values(
        ["subject_id", "culture_time", "micro_specimen_id"], kind="stable"
    ).reset_index(drop=True)


def match_suspected_infections(antibiotics: pd.DataFrame, cultures: pd.DataFrame) -> pd.DataFrame:
    """Exact 72-hour prior / 24-hour future suspected-infection match."""
    indexed: dict[int, list[tuple[int, object]]] = {}
    for subject_id, group in cultures.sort_values("culture_time", kind="stable").groupby("subject_id"):
        indexed[int(subject_id)] = [
            (int(pd.Timestamp(row.culture_time).value), row) for row in group.itertuples(index=False)
        ]
    matches: list[SuspectedInfectionMatch] = []
    ordered = antibiotics.sort_values(["subject_id", "stay_id", "antibiotic_time"], kind="stable")
    for row in ordered.itertuples(index=False):
        subject_cultures = indexed.get(int(row.subject_id))
        if subject_cultures is None:
            continue
        times = [item[0] for item in subject_cultures]
        antibiotic_time = pd.Timestamp(row.antibiotic_time)
        antibiotic_ns = int(antibiotic_time.value)
        prior_start = bisect.bisect_left(times, antibiotic_ns - 72 * HOURS_NS)
        prior_end = bisect.bisect_left(times, antibiotic_ns)
        if prior_start < prior_end:
            culture = subject_cultures[prior_start][1]
            onset = pd.Timestamp(culture.culture_time)
        else:
            future = bisect.bisect_right(times, antibiotic_ns)
            if future >= len(times) or times[future] > antibiotic_ns + 24 * HOURS_NS:
                continue
            culture = subject_cultures[future][1]
            onset = antibiotic_time
        matches.append(SuspectedInfectionMatch(
            subject_id=int(row.subject_id), hadm_id=int(row.hadm_id), stay_id=int(row.stay_id),
            antibiotic_time=antibiotic_time, culture_time=pd.Timestamp(culture.culture_time),
            suspected_infection_time=onset,
        ))
    if not matches:
        return pd.DataFrame(columns=("subject_id", "hadm_id", "stay_id", "suspected_infection_time"))
    output = pd.DataFrame({field: getattr(match, field) for field in SuspectedInfectionMatch.__dataclass_fields__} for match in matches)
    return output.sort_values(
        ["stay_id", "suspected_infection_time", "antibiotic_time", "culture_time"], kind="stable"
    ).drop_duplicates("stay_id", keep="first").reset_index(drop=True)


def sepsis_sofa_filter(
    candidates: pd.DataFrame,
    values: np.ndarray,
    masks: np.ndarray,
    *,
    sofa_index: int,
    minimum: float = 2.0,
) -> pd.DataFrame:
    """Frozen sepsis exporter gate: observed window and maximum SOFA >= 2."""
    output = candidates.copy()
    keep = np.ones(len(output), dtype=bool)
    sepsis_rows = output[output["task_id"].eq("sepsis")]
    if not sepsis_rows.empty:
        indices = sepsis_rows["episode_idx"].to_numpy(dtype=int)
        sofa = np.where(masks[indices, :, sofa_index].astype(bool), values[indices, :, sofa_index], np.nan)
        with np.errstate(all="ignore"):
            maxima = np.nanmax(sofa, axis=1)
        accepted = np.isfinite(maxima) & (maxima >= minimum)
        keep[indices] = accepted
    retained = output.loc[keep].copy()
    retained_sepsis = set(retained.loc[retained["task_id"].eq("sepsis"), "stay_id"].astype(int))
    return retained.loc[
        ~retained["task_id"].eq("heart_failure") | ~retained["stay_id"].astype(int).isin(retained_sepsis)
    ].copy()


def kdd097_interval_bins(
    start: pd.Timestamp,
    end: pd.Timestamp,
    window_start: pd.Timestamp,
    *,
    bin_hours: int,
    n_steps: int,
) -> range:
    """KDD097 presence/max-exposure bin allocation, including its end bin."""
    first = int(math.floor((start - window_start).total_seconds() / (bin_hours * 3600)))
    last = int(math.floor((end - window_start).total_seconds() / (bin_hours * 3600)))
    first = max(0, first)
    last = min(n_steps - 1, last)
    return range(first, last + 1) if last >= first else range(0)
