from __future__ import annotations

import math

import numpy as np
import pandas as pd


GCS_EYE_ITEMID = 220739
GCS_VERBAL_ITEMID = 223900
GCS_MOTOR_ITEMID = 223901
GCS_TOTAL_ITEMID = 227013
GCS_ITEMIDS = {GCS_EYE_ITEMID, GCS_VERBAL_ITEMID, GCS_MOTOR_ITEMID, GCS_TOTAL_ITEMID}

CLINICAL_RANGES: dict[str, tuple[float, float]] = {
    "weight": (20.0, 300.0), "heart_rate": (20.0, 250.0), "sbp": (30.0, 300.0),
    "mbp": (20.0, 200.0), "dbp": (10.0, 200.0), "respiratory_rate": (1.0, 80.0),
    "temperature_c": (25.0, 45.0), "spo2": (1.0, 100.0), "gcs_proxy": (3.0, 15.0),
    "fio2": (21.0, 100.0), "lactate": (0.0, 30.0), "pao2": (20.0, 700.0),
    "paco2": (10.0, 200.0), "ph": (6.5, 8.0), "base_excess": (-40.0, 40.0),
    "co2_bicarbonate": (0.0, 80.0), "pao2_fio2": (10.0, 700.0),
    "wbc": (0.1, 200.0), "platelet": (1.0, 2000.0), "bun": (0.0, 300.0),
    "creatinine": (0.0, 30.0), "ptt": (0.0, 300.0), "pt": (0.0, 200.0),
    "inr": (0.0, 30.0), "ast": (0.0, 20000.0), "alt": (0.0, 20000.0),
    "total_bilirubin": (0.0, 100.0), "magnesium": (0.0, 20.0),
    "ionized_calcium": (0.0, 5.0), "calcium": (0.0, 30.0),
    "urine_output": (0.0, 20000.0), "mechanical_ventilation": (0.0, 1.0),
}


def clean_feature_values(name: str, values: np.ndarray) -> np.ndarray:
    cleaned = values.astype(np.float64, copy=True)
    bounds = CLINICAL_RANGES.get(name)
    if bounds is None:
        return cleaned
    low, high = bounds
    valid = np.isfinite(cleaned) & (cleaned >= low) & (cleaned <= high)
    cleaned[~valid] = np.nan
    return cleaned


def gcs_total_rows(events: pd.DataFrame) -> pd.DataFrame:
    columns = ["episode_idx", "bin", "feature_value"]
    if events.empty:
        return pd.DataFrame(columns=columns)
    direct = events[events["itemid"].astype(int).eq(GCS_TOTAL_ITEMID)]
    direct = direct.groupby(["episode_idx", "bin"], observed=True)["feature_value"].median().reset_index()
    direct["priority"] = 1
    component = events[events["itemid"].astype(int).isin(GCS_ITEMIDS - {GCS_TOTAL_ITEMID})].copy()
    component["component"] = component["itemid"].astype(int).map(
        {GCS_EYE_ITEMID: "eye", GCS_VERBAL_ITEMID: "verbal", GCS_MOTOR_ITEMID: "motor"}
    )
    if component.empty:
        derived = pd.DataFrame(columns=columns + ["priority"])
    else:
        wide = component.groupby(["episode_idx", "bin", "component"], observed=True)["feature_value"].median().unstack("component")
        required = [name for name in ("eye", "verbal", "motor") if name in wide]
        wide = wide.dropna(subset=required) if len(required) == 3 else wide.iloc[0:0]
        derived = wide.reset_index()
        derived["feature_value"] = derived["eye"] + derived["verbal"] + derived["motor"] if not derived.empty else np.nan
        derived = derived[["episode_idx", "bin", "feature_value"]]
        derived["priority"] = 2
    output = pd.concat([direct, derived], ignore_index=True)
    if output.empty:
        return pd.DataFrame(columns=columns)
    output = output.sort_values(["episode_idx", "bin", "priority"], kind="stable").drop_duplicates(["episode_idx", "bin"], keep="first")
    output["feature_value"] = clean_feature_values("gcs_proxy", output["feature_value"].to_numpy(float))
    return output.loc[output["feature_value"].notna(), columns]


def overlap_bin_amounts(start: pd.Timestamp, end: pd.Timestamp, window_start: pd.Timestamp, *, bin_hours: int, n_steps: int, amount: float) -> list[tuple[int, float]]:
    if not math.isfinite(amount) or amount <= 0:
        return []
    duration = max((end - start).total_seconds(), 60.0)
    first = math.floor((start - window_start).total_seconds() / (bin_hours * 3600))
    last = math.floor((end - window_start).total_seconds() / (bin_hours * 3600))
    pieces: list[tuple[int, float]] = []
    for index in range(max(0, first), min(n_steps - 1, last) + 1):
        bin_start = window_start + pd.Timedelta(hours=index * bin_hours)
        bin_end = bin_start + pd.Timedelta(hours=bin_hours)
        overlap = max(0.0, (min(end, bin_end) - max(start, bin_start)).total_seconds())
        if overlap > 0:
            pieces.append((index, amount * overlap / duration))
    return pieces


def overlap_bins(start: pd.Timestamp, end: pd.Timestamp, window_start: pd.Timestamp, *, bin_hours: int, n_steps: int) -> range:
    first = max(0, int(math.floor((start - window_start).total_seconds() / (bin_hours * 3600))))
    last = min(n_steps - 1, int(math.ceil((end - window_start).total_seconds() / (bin_hours * 3600)) - 1))
    return range(first, last + 1) if last >= first else range(0)


def _zeros(values: np.ndarray) -> np.ndarray:
    return np.zeros(values.shape, dtype=np.int16)


def sirs_score(*, temperature_c: np.ndarray, heart_rate: np.ndarray, respiratory_rate: np.ndarray, paco2: np.ndarray, wbc: np.ndarray) -> np.ndarray:
    score = _zeros(heart_rate)
    score += (np.isfinite(temperature_c) & ((temperature_c >= 38) | (temperature_c <= 36))).astype(np.int16)
    score += (np.isfinite(heart_rate) & (heart_rate > 90)).astype(np.int16)
    score += ((np.isfinite(respiratory_rate) & (respiratory_rate >= 20)) | (np.isfinite(paco2) & (paco2 <= 32))).astype(np.int16)
    score += (np.isfinite(wbc) & ((wbc >= 12) | (wbc < 4))).astype(np.int16)
    return score


def compute_corrected_derived_features(features: np.ndarray, mask: np.ndarray, vaso: np.ndarray, *, feature_index: dict[str, int], bin_hours: int) -> None:
    def get(name: str) -> tuple[np.ndarray, np.ndarray]:
        index = feature_index[name]
        return features[:, :, index], mask[:, :, index].astype(bool)

    def put(name: str, values: np.ndarray, observed: np.ndarray) -> None:
        index = feature_index[name]
        cleaned = clean_feature_values(name, values.astype(np.float64)).astype(np.float32)
        keep = observed & np.isfinite(cleaned)
        features[:, :, index] = np.where(keep, cleaned, features[:, :, index])
        mask[:, :, index] = keep

    hr, hr_m = get("heart_rate"); sbp, sbp_m = get("sbp")
    put("shock_index", hr / np.maximum(sbp, 1e-6), hr_m & sbp_m & (sbp > 0))
    pao2, pao2_m = get("pao2"); fio2, fio2_m = get("fio2")
    fraction = np.where(fio2 > 1, fio2 / 100, fio2)
    put("pao2_fio2", pao2 / np.maximum(fraction, 1e-6), pao2_m & fio2_m & (fraction > 0))
    temp, temp_m = get("temperature_c"); rr, rr_m = get("respiratory_rate"); paco2, paco2_m = get("paco2"); wbc, wbc_m = get("wbc")
    put("sirs_proxy", sirs_score(temperature_c=np.where(temp_m, temp, np.nan), heart_rate=np.where(hr_m, hr, np.nan), respiratory_rate=np.where(rr_m, rr, np.nan), paco2=np.where(paco2_m, paco2, np.nan), wbc=np.where(wbc_m, wbc, np.nan)), temp_m | hr_m | rr_m | paco2_m | wbc_m)
    pf, pf_m = get("pao2_fio2"); platelets, platelets_m = get("platelet"); bilirubin, bilirubin_m = get("total_bilirubin"); mbp, mbp_m = get("mbp"); gcs, gcs_m = get("gcs_proxy"); creat, creat_m = get("creatinine"); urine, urine_m = get("urine_output")
    score = np.zeros(pf.shape, dtype=np.int16)
    for threshold in (400, 300, 200, 100): score += (pf_m & (pf < threshold)).astype(np.int16)
    for threshold in (150, 100, 50, 20): score += (platelets_m & (platelets < threshold)).astype(np.int16)
    for threshold in (1.2, 2, 6, 12): score += (bilirubin_m & (bilirubin >= threshold)).astype(np.int16)
    cardio = np.zeros_like(score); cardio[mbp_m & (mbp < 70)] = 1; cardio[mbp_m & (mbp < 65)] = 2; cardio[vaso > 0] = 3; cardio[vaso > .1] = 4; score += cardio
    cns = np.zeros_like(score); cns[gcs_m & (gcs < 15)] = 1; cns[gcs_m & (gcs <= 12)] = 2; cns[gcs_m & (gcs <= 9)] = 3; cns[gcs_m & (gcs <= 5)] = 4; score += cns
    renal = np.zeros_like(score); renal[creat_m & (creat >= 1.2)] = 1; renal[creat_m & (creat >= 2)] = 2; renal[creat_m & (creat >= 3.5)] = 3; renal[creat_m & (creat >= 5)] = 4
    urine_4h = np.where(urine_m, urine, 0.0) if bin_hours >= 4 else np.where(urine_m, urine, 0.0)
    renal[urine_m & (urine_4h < 84)] = np.maximum(renal[urine_m & (urine_4h < 84)], 3); renal[urine_m & (urine_4h < 34)] = 4; score += renal
    put("sofa_proxy", score, pf_m | platelets_m | bilirubin_m | mbp_m | gcs_m | creat_m | urine_m | (vaso > 0))
