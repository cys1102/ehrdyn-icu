from __future__ import annotations

import hashlib
import json
import locale
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .runtime_config import RUNTIME_CONFIG


class ContractError(RuntimeError):
    """Raised when the frozen reconstruction contract is violated."""


RELEASE = "MIMIC-IV-3.1"
TASKS = ("sepsis", "respiratory_support", "shock", "aki", "heart_failure")
ROLE_SALT = RUNTIME_CONFIG["roles"]["large_lineage_salt"]
ROLE_ORDER = ("train", "validation", "historical_other")
BIN_HOURS = RUNTIME_CONFIG["temporal"]["bin_hours"]
POST_ANCHOR_HOURS = RUNTIME_CONFIG["temporal"]["episode_post_anchor_hours"]
PRE_ANCHOR_HOURS = RUNTIME_CONFIG["temporal"]["episode_pre_anchor_hours"]
EPISODE_WINDOW_HOURS = RUNTIME_CONFIG["temporal"]["episode_window_hours"]
EPISODE_BINS = RUNTIME_CONFIG["temporal"]["episode_bins"]
RAW_EXTRACTION_POST_HOURS = RUNTIME_CONFIG["temporal"]["raw_rv01r_post_base_anchor_extraction_hours"]
SEPSIS_MAX_ANCHOR_SHIFT_HOURS = RUNTIME_CONFIG["temporal"]["sepsis_max_base_to_final_anchor_shift_hours"]
LONGEST_RECURSIVE_TARGET_HOURS = RUNTIME_CONFIG["temporal"]["longest_recursive_target_hours"]
RAW_EXTRACTION_BINS = (PRE_ANCHOR_HOURS + RAW_EXTRACTION_POST_HOURS) // BIN_HOURS
LINEAGE_ROUTER = RUNTIME_CONFIG["lineage_router"]

FEATURE_NAMES = (
    "age", "gender_male", "weight", "readmission", "elixhauser_score_proxy",
    "heart_rate", "sbp", "mbp", "dbp", "respiratory_rate",
    "temperature_c", "spo2", "shock_index", "sofa_proxy", "gcs_proxy",
    "fio2", "sirs_proxy", "lactate", "pao2", "paco2", "ph",
    "base_excess", "co2_bicarbonate", "pao2_fio2", "wbc", "platelet",
    "bun", "creatinine", "ptt", "pt", "inr", "ast", "alt",
    "total_bilirubin", "magnesium", "ionized_calcium", "calcium",
    "urine_output", "mechanical_ventilation", "step_id",
)
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
SAFE_FEATURE_INDICES = tuple(
    index for index, name in enumerate(FEATURE_NAMES)
    if name not in {"age", "gender_male", "weight", "readmission", "elixhauser_score_proxy", "sofa_proxy", "step_id"}
)
SAFE_FEATURE_NAMES = tuple(FEATURE_NAMES[index] for index in SAFE_FEATURE_INDICES)

CHART_ITEM_MAP = {
    220045: "heart_rate", 220050: "sbp", 220179: "sbp", 225309: "sbp",
    220051: "dbp", 220180: "dbp", 225310: "dbp", 220052: "mbp",
    220181: "mbp", 220210: "respiratory_rate", 224688: "respiratory_rate",
    224689: "respiratory_rate", 224690: "respiratory_rate",
    223761: "temperature_c", 223762: "temperature_c", 226329: "temperature_c",
    228242: "temperature_c", 220277: "spo2", 220739: "gcs_proxy",
    223900: "gcs_proxy", 223901: "gcs_proxy", 227013: "gcs_proxy",
    226754: "fio2", 227010: "fio2", 229280: "fio2", 224639: "weight",
    226512: "weight", 226531: "weight",
}
LAB_ITEM_MAP = {
    50813: "lactate", 52442: "lactate", 53154: "lactate", 50821: "pao2",
    52042: "pao2", 50818: "paco2", 52040: "paco2", 50820: "ph",
    50802: "base_excess", 52038: "base_excess", 50803: "co2_bicarbonate",
    50882: "co2_bicarbonate", 52039: "co2_bicarbonate", 51300: "wbc",
    51301: "wbc", 51755: "wbc", 51756: "wbc", 51265: "platelet",
    53189: "platelet", 51006: "bun", 52647: "bun", 50912: "creatinine",
    52024: "creatinine", 52546: "creatinine", 51275: "ptt", 52923: "ptt",
    51274: "pt", 52921: "pt", 51237: "inr", 51675: "inr", 50878: "ast",
    53088: "ast", 50861: "alt", 53084: "alt", 50885: "total_bilirubin",
    53089: "total_bilirubin", 50960: "magnesium", 50808: "ionized_calcium",
    51624: "ionized_calcium", 50893: "calcium", 52034: "calcium", 52035: "calcium",
}
DBP_ITEMIDS = (220051, 220180, 225310)
MBP_ITEMIDS = (220052, 220181)
SBP_ITEMIDS = (220050, 220179, 225309)
PEEP_ITEMIDS = (220339, 224700)
MECHVENT_ITEMIDS = (225792, 225794)
FIO2_ITEMIDS = (226754, 227010, 229280)
URINE_ITEMIDS = (226566, 226627, 226631, 227489)
VASO_ITEMIDS = (221289, 221653, 221662, 221749, 221906, 222315, 229617, 229630, 229631, 229632)
FLUID_ITEMIDS = (225158, 225159, 225823, 225825, 225827, 225828, 225941, 226089, 226364, 226375)
CREATININE_ITEMIDS = tuple(item for item, name in LAB_ITEM_MAP.items() if name == "creatinine")

ANTIBIOTIC_PATTERN = (
    r"vancomycin|cefepime|ceftriaxone|cefazolin|ceftazidime|ceftaroline|cefotaxime|cefuroxime|"
    r"piperacillin|tazobactam|meropenem|imipenem|ertapenem|aztreonam|ciprofloxacin|levofloxacin|"
    r"moxifloxacin|metronidazole|clindamycin|linezolid|daptomycin|gentamicin|tobramycin|amikacin|"
    r"ampicillin|amoxicillin|nafcillin|oxacillin|penicillin|doxycycline|minocycline|bactrim|"
    r"sulfamethoxazole|trimethoprim|tigecycline"
)
DIURETIC_PATTERN = r"furosemide|bumetanide|torsemide|hydrochlorothiazide|metolazone|chlorothiazide|ethacrynic|spironolactone"
RRT_PATTERN = r"dialysis|crrt|cvvh|hemofiltration|ultrafiltration"
RESPIRATORY_PATTERN = r"oxygen|high flow|hfnc|bipap|cpap|non[- ]?invasive|ventilat|intubat|respirator"
VASO_PATTERN = r"norepinephrine|phenylephrine|vasopressin|epinephrine|dopamine"
VASODILATOR_PATTERN = r"nitroglycerin|nitroprusside|hydralazine|nicardipine|clevidipine|isosorbide"


@dataclass(frozen=True)
class TableContract:
    relative: str
    required: tuple[str, ...]
    unique_key: tuple[str, ...] = ()


TABLES = (
    TableContract("hosp/patients", ("subject_id", "gender", "anchor_age", "dod"), ("subject_id",)),
    TableContract("hosp/admissions", ("subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "hospital_expire_flag", "discharge_location"), ("hadm_id",)),
    TableContract("hosp/diagnoses_icd", ("subject_id", "hadm_id", "icd_code", "icd_version")),
    TableContract("hosp/procedures_icd", ("subject_id", "hadm_id", "seq_num", "chartdate", "icd_code", "icd_version")),
    TableContract("hosp/prescriptions", ("subject_id", "hadm_id", "starttime", "stoptime", "drug", "route")),
    TableContract("hosp/microbiologyevents", ("microevent_id", "subject_id", "hadm_id", "micro_specimen_id", "chartdate", "charttime", "spec_type_desc", "org_itemid", "org_name"), ("microevent_id",)),
    TableContract("hosp/labevents", ("labevent_id", "subject_id", "hadm_id", "itemid", "charttime", "valuenum", "valueuom"), ("labevent_id",)),
    TableContract("hosp/d_labitems", ("itemid", "label", "fluid", "category"), ("itemid",)),
    TableContract("icu/icustays", ("subject_id", "hadm_id", "stay_id", "first_careunit", "last_careunit", "intime", "outtime"), ("stay_id",)),
    TableContract("icu/d_items", ("itemid", "label", "abbreviation", "linksto", "category", "unitname"), ("itemid",)),
    TableContract("icu/chartevents", ("subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum", "valueuom")),
    TableContract("icu/inputevents", ("subject_id", "hadm_id", "stay_id", "starttime", "endtime", "itemid", "amount", "amountuom", "rate", "rateuom", "ordercategoryname", "ordercategorydescription")),
    TableContract("icu/procedureevents", ("subject_id", "hadm_id", "stay_id", "starttime", "endtime", "itemid")),
    TableContract("icu/outputevents", ("subject_id", "hadm_id", "stay_id", "charttime", "itemid", "value")),
)


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def source_table(root: Path, relative: str) -> Path:
    plain = root / f"{relative}.csv"
    compressed = root / f"{relative}.csv.gz"
    found = [path for path in (plain, compressed) if path.is_file()]
    if len(found) != 1:
        raise ContractError(f"expected exactly one CSV or CSV.GZ source for {relative}")
    return found[0]


def validate_layout(root: Path) -> dict[str, Path]:
    if root.name != "3.1":
        raise ContractError("MIMIC-IV root must be the official 3.1 release directory")
    paths: dict[str, Path] = {}
    for table in TABLES:
        path = source_table(root, table.relative)
        try:
            columns = tuple(pd.read_csv(path, nrows=0).columns)
        except (UnicodeDecodeError, ValueError, OSError) as exc:
            raise ContractError(f"unsupported or malformed source encoding: {table.relative}") from exc
        missing = sorted(set(table.required) - set(columns))
        if missing:
            raise ContractError(f"missing columns in {table.relative}: {','.join(missing)}")
        paths[table.relative] = path
    return paths


def assert_unique(frame: pd.DataFrame, columns: Sequence[str], table: str) -> None:
    if columns and frame.duplicated(list(columns), keep=False).any():
        raise ContractError(f"duplicate identifiers in {table}: {','.join(columns)}")


def parse_times(frame: pd.DataFrame, columns: Iterable[str], table: str) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column not in output:
            continue
        raw = output[column]
        parsed = pd.to_datetime(raw, errors="coerce", utc=False)
        malformed = raw.notna() & raw.astype("string").str.strip().ne("") & parsed.isna()
        if malformed.any():
            raise ContractError(f"malformed timestamp in {table}.{column}")
        output[column] = parsed
    return output


def subject_role(subject_id: int) -> str:
    digest = hashlib.sha256(f"{ROLE_SALT}{int(subject_id)}".encode()).digest()
    role = RUNTIME_CONFIG["roles"]
    bucket = int.from_bytes(digest[:8], "big", signed=False) % sum(role["large_lineage_ranges"].values())
    train = role["large_lineage_ranges"]["train"]
    validation = train + role["large_lineage_ranges"]["validation"]
    return "train" if bucket < train else "validation" if bucket < validation else "historical_other"


def compact_lineage_role(subject_id: int) -> str:
    role = RUNTIME_CONFIG["roles"]
    bucket = (int(subject_id) * role["compact_lineage_multiplier"] + role["compact_lineage_increment"]) % role["compact_lineage_modulus"]
    train = role["compact_lineage_ranges"]["train"]
    validation = train + role["compact_lineage_ranges"]["validation"]
    return "train" if bucket < train else "validation" if bucket < validation else "historical_other"


def extraction_post_hours(task: str) -> int:
    if task not in TASKS:
        raise ContractError(f"unknown task: {task}")
    return RAW_EXTRACTION_POST_HOURS if task in {"respiratory_support", "shock", "aki"} else POST_ANCHOR_HOURS


def episode_interface_indices(base_anchor: pd.Timestamp, final_anchor: pd.Timestamp) -> tuple[int, ...]:
    shift = (pd.Timestamp(final_anchor) - pd.Timestamp(base_anchor)).total_seconds() / 3600
    if shift < 0 or shift > SEPSIS_MAX_ANCHOR_SHIFT_HOURS or shift % BIN_HOURS:
        raise ContractError("final anchor shift is outside the frozen aligned range")
    first = int(shift // BIN_HOURS)
    indices = tuple(range(first, first + EPISODE_BINS))
    if indices[-1] >= RAW_EXTRACTION_BINS:
        raise ContractError("raw extraction buffer does not cover the final episode interface")
    return indices


def fit_train_positive_edges(values: np.ndarray, observed: np.ndarray, roles: Sequence[str]) -> tuple[float, ...]:
    values = np.asarray(values, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    roles = np.asarray(roles, dtype=object)
    train = values[(roles == "train") & observed & np.isfinite(values) & (values > 0)]
    if not train.size:
        raise ContractError("no positive training action values")
    return tuple(float(value) for value in np.quantile(train, (0.25, 0.5, 0.75)))


def encode_five_levels(values: np.ndarray, observed: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    labels = np.full(values.shape, -1, dtype=np.int16)
    valid = observed & np.isfinite(values)
    labels[valid] = 0
    positive = valid & (values > 0)
    labels[positive] = np.searchsorted(np.asarray(edges, dtype=float), values[positive], side="right") + 1
    if np.any(labels[valid] > 4):
        raise ContractError("five-level action encoder exceeded K=5")
    return labels


def joint_codes(left: np.ndarray, right: np.ndarray, valid: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.int16)
    right = np.asarray(right, dtype=np.int16)
    valid = np.asarray(valid, dtype=bool)
    if left.shape != right.shape or left.shape != valid.shape:
        raise ContractError("joint action arrays differ in shape")
    codes = np.full(left.shape, -1, dtype=np.int16)
    usable = valid & (left >= 0) & (right >= 0)
    codes[usable] = left[usable] * 5 + right[usable]
    return codes


def eligible_transition_indices(anchor: pd.Timestamp, intime: pd.Timestamp, outtime: pd.Timestamp) -> tuple[int, ...]:
    anchor, intime, outtime = map(pd.Timestamp, (anchor, intime, outtime))
    step = pd.Timedelta(hours=BIN_HOURS)
    limit = min(outtime, anchor + pd.Timedelta(hours=POST_ANCHOR_HOURS))
    output: list[int] = []
    k = 0
    while anchor + (k + 2) * step <= limit:
        time = anchor + k * step
        if time - step >= intime and time + 2 * step <= outtime:
            output.append(k)
        k += 1
    return tuple(output)


DISPOSITION3_ANCHORS = {
    "structured_ventilation_chart_event",
    "structured_respiratory_support_event",
    "vasopressor_support",
    "time_stamped_rrt_start",
    "current_stay_decongestion_prescription",
    "current_stay_decongestion_input",
}


def apply_kdd201_temporal_repair(anchor_source: str, relative_transitions: Sequence[int]) -> np.ndarray:
    values = np.asarray(relative_transitions, dtype=int)
    return (values != 0) if anchor_source in DISPOSITION3_ANCHORS else np.ones(values.shape, dtype=bool)


def corrected_chart_value(itemid: int, value: object, unit: object = None) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not np.isfinite(numeric) or numeric in {-9999.0, -999.0, -99.0}:
        return math.nan
    itemid = int(itemid)
    if itemid in DBP_ITEMIDS:
        return float(numeric) if str(unit) == "mmHg" and 10 <= numeric <= 200 else math.nan
    if itemid == 223761:
        numeric = (numeric - 32.0) * 5.0 / 9.0
    if itemid == 228242 or itemid == 223835:
        return math.nan
    feature = CHART_ITEM_MAP.get(itemid)
    support = {
        "fio2": (21, 100), "gcs_proxy": (1, 15), "mbp": (20, 200),
        "temperature_c": (25, 45), "sbp": (20, 300), "spo2": (0, 100),
    }
    if feature in support and not support[feature][0] <= numeric <= support[feature][1]:
        return math.nan
    return float(numeric)


def reward_components(
    task: str,
    states: np.ndarray,
    state_masks: np.ndarray,
    targets: np.ndarray,
    target_masks: np.ndarray,
    valid: np.ndarray,
    outcomes: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    if task in {"sepsis", "aki", "heart_failure"}:
        names = ("terminal_discharge_origin_90d_proxy",)
    elif task == "shock":
        names = ("shock_next_mbp_component",)
    elif task == "respiratory_support":
        names = ("resp_meddreamer_spo2_mbp",)
    else:
        raise ContractError(f"unknown task: {task}")
    reward = np.zeros((*valid.shape, len(names)), dtype=np.float32)
    mask = np.zeros_like(reward, dtype=bool)
    if names[0] == "terminal_discharge_origin_90d_proxy":
        if outcomes is None or len(outcomes) != len(valid):
            raise ContractError("terminal outcomes missing")
        for episode, length in enumerate(valid.sum(axis=1).astype(int)):
            if length:
                reward[episode, length - 1, 0] = -1.0 if outcomes[episode] > 0.5 else 1.0
                mask[episode, length - 1, 0] = True
    elif task == "shock":
        index = FEATURE_INDEX["mbp"]
        mask[..., 0] = valid & target_masks[..., index]
        reward[..., 0] = np.clip((targets[..., index] - 65.0) / 25.0, -1.0, 1.0)
    else:
        spo2, mbp = FEATURE_INDEX["spo2"], FEATURE_INDEX["mbp"]
        mask[..., 0] = valid & target_masks[..., spo2] & target_masks[..., mbp]
        reward[..., 0] = np.where((targets[..., spo2] >= 94) & (targets[..., spo2] <= 98), 1.0, -0.5)
        reward[..., 0] += np.where((targets[..., mbp] >= 70) & (targets[..., mbp] <= 80), 1.0, -0.5)
    reward[~mask] = 0.0
    return reward, mask, names


def assert_subject_disjoint(frame: pd.DataFrame) -> None:
    if frame.groupby("subject_id", observed=True)["role"].nunique().gt(1).any():
        raise ContractError("subject role overlap")


def aggregate_receipt(
    task_rows: Mapping[str, Mapping[str, object]],
    source_hashes: Mapping[str, str],
    streaming_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if tuple(task_rows) != TASKS:
        raise ContractError("receipt task inventory or order mismatch")
    if len(streaming_rows) != 7:
        raise ContractError("streaming receipt must contain seven high-volume tables")
    return {
        "schema_version": "1.1.0",
        "release": RELEASE,
        "candidate_status": "credentialed_parity_pending",
        "source_hashes": dict(sorted(source_hashes.items())),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "timezone": "UTC",
            "locale": locale.setlocale(locale.LC_ALL, None),
        },
        "contracts": {
            "feature_order_sha256": hashlib.sha256("\n".join(SAFE_FEATURE_NAMES).encode()).hexdigest(),
            "role_assignment_sha256": hashlib.sha256((ROLE_SALT + "train:7000;validation:1500;historical_other:1500").encode()).hexdigest(),
            "bin_hours": BIN_HOURS,
            "pre_anchor_hours": PRE_ANCHOR_HOURS,
            "post_anchor_hours": POST_ANCHOR_HOURS,
            "episode_window_hours": EPISODE_WINDOW_HOURS,
            "episode_bins": EPISODE_BINS,
            "raw_extraction_post_base_anchor_hours": RAW_EXTRACTION_POST_HOURS,
            "sepsis_max_base_to_final_anchor_shift_hours": SEPSIS_MAX_ANCHOR_SHIFT_HOURS,
            "longest_recursive_target_hours": LONGEST_RECURSIVE_TARGET_HOURS,
        },
        "streaming": [dict(row) for row in streaming_rows],
        "tasks": [dict({"task_id": task}, **task_rows[task]) for task in TASKS],
        "privacy": {
            "aggregate_only": True,
            "subgroup_cells_exported": False,
            "row_level_fields_exported": False,
        },
    }
