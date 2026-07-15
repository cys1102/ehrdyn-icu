"""KDD-RV successor release-candidate contracts.

This namespace is deliberately separate from the historical KDD089 release.
It contains software contracts only and does not expose credentialed rows.
"""

from __future__ import annotations

from typing import Final

SUCCESSOR_BENCHMARK_VERSION: Final = "KDD-RV-SUCCESSOR-RC1"
SUCCESSOR_PACKAGE_VERSION: Final = "1.2.0rc1"
EVALUATION_CONTRACT_VERSION: Final = "KDD-RV-EVALUATION-CONTRACT-v1"
EVALUATION_RECEIPT_VERSION: Final = "KDD-RV-EVALUATOR-RECEIPT-v1"
MIN_PUBLIC_SUBJECT_CLUSTERS: Final = 100
MIN_PUBLIC_OBSERVED_CELLS: Final = 1000
ROLE_SALT: Final = "KDD-RV-SUBJECT-ROLE-v1|"
NORMALIZATION_RULE: Final = (
    "per-feature mean and population standard deviation fit from observed train "
    "pre-action state cells only; non-finite or sub-1e-6 scales set to one"
)
TASKS: Final = ("sepsis", "respiratory", "aki", "af_flutter", "heart_failure")
MODES: Final = ("one_step", "conditional_recursive")
TASK_VERSIONS: Final = {
    "sepsis": "KDD-RV-SEPSIS-SI-OD-v1.0.0",
    "respiratory": "KDD-RV-RESP-SUPPORT-v1.0.0",
    "aki": "KDD-RV-AKI-KDIGO-CR-RRT-v1.0.0",
    "af_flutter": "KDD-RV-AF-PRIOR-DX-RR-v1.0.0",
    "heart_failure": "KDD-RV-HF-PRIOR-DX-DECONG-v1.0.0",
}
REALIZED_ACTION_CLASSES: Final = {
    "sepsis": frozenset({"0", "1", "2"}),
    "respiratory": frozenset({"0", "1", "2"}),
    "aki": frozenset({"0", "1", "2", "3"}),
    "af_flutter": frozenset({"0", "1"}),
    "heart_failure": frozenset({"0", "1"}),
}
CORE_FAMILIES: Final = (
    "persistence_locf",
    "previous_window_deterministic",
    "local_ridge_residual",
    "local_hgb_residual",
    "local_gru_residual",
    "clean_gaussian_transition",
)
FEATURE_NAMES: Final = (
    "heart_rate",
    "sbp",
    "mbp",
    "dbp",
    "respiratory_rate",
    "temperature_c",
    "spo2",
    "shock_index",
    "gcs_proxy",
    "fio2",
    "sirs_proxy",
    "lactate",
    "pao2",
    "paco2",
    "ph",
    "base_excess",
    "co2_bicarbonate",
    "pao2_fio2",
    "wbc",
    "platelet",
    "bun",
    "creatinine",
    "ptt",
    "pt",
    "inr",
    "ast",
    "alt",
    "total_bilirubin",
    "magnesium",
    "ionized_calcium",
    "calcium",
    "urine_output",
    "mechanical_ventilation",
)
CLAIM_BOUNDARY: Final = (
    "Recorded-trajectory forecasting and benchmark-validity auditing only. "
    "Treatment, causal, counterfactual, policy-selection, clinical-utility, "
    "deployment, and autonomous-decision claims are blocked."
)
