from __future__ import annotations

import math
from collections.abc import Mapping

from .errors import ReleaseContractError


def sepsis_relative_gate(candidate: Mapping[str, object], reference: Mapping[str, object], limits: Mapping[str, object]) -> dict[str, object]:
    rmse_ratio = _number(candidate, "rmse") / _positive(reference, "rmse")
    coverage_difference = _number(candidate, "cov90") - _number(reference, "cov90")
    behavior_nll_ratio = _number(candidate, "behavior_nll") / _positive(reference, "behavior_nll")
    passed = (
        rmse_ratio <= _number(limits, "rmse_ratio_max")
        and abs(coverage_difference) <= _number(limits, "coverage_abs_difference_max")
        and behavior_nll_ratio <= _number(limits, "behavior_nll_ratio_max")
        and _number(candidate, "low_support_rate") <= _number(limits, "low_support_rate_max")
    )
    return {"gate": "sepsis_relative_comparability", "pass": passed, "rmse_ratio": rmse_ratio, "coverage_difference": coverage_difference, "behavior_nll_ratio": behavior_nll_ratio}


def absolute_policy_gate(metrics: Mapping[str, object]) -> dict[str, object]:
    required = ("ess", "ess_fraction", "wis", "wpdis", "fqe_finite", "clipping_stable", "denominator_ranking_stable", "reward_robust", "naive_policy_sanity", "ope_provenance_complete")
    missing = [name for name in required if name not in metrics]
    if missing:
        return {"gate": "absolute_policy_evaluability", "pass": False, "missing": ";".join(missing)}
    finite = all(math.isfinite(_number(metrics, name)) for name in ("wis", "wpdis"))
    passed = _number(metrics, "ess") >= 100 and _number(metrics, "ess_fraction") >= .01 and finite and all(bool(metrics[name]) for name in required[4:])
    return {"gate": "absolute_policy_evaluability", "pass": passed, "missing": ""}


def _number(values: Mapping[str, object], key: str) -> float:
    try:
        raw = values[key]
        if not isinstance(raw, int | float | str):
            raise TypeError
        value = float(raw)
    except (KeyError, TypeError, ValueError) as error:
        raise ReleaseContractError(f"Missing or invalid gate metric: {key}") from error
    if not math.isfinite(value):
        raise ReleaseContractError(f"Non-finite gate metric: {key}")
    return value


def _positive(values: Mapping[str, object], key: str) -> float:
    value = _number(values, key)
    if value <= 0:
        raise ReleaseContractError(f"Gate metric must be positive: {key}")
    return value
