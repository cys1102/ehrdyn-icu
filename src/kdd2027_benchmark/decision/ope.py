from __future__ import annotations

import numpy as np


def trajectory_is(
    rewards: np.ndarray,
    target_probabilities: np.ndarray,
    behavior_probabilities: np.ndarray,
    *,
    discount: float = 0.99,
    clip: float | None = None,
) -> dict[str, float]:
    ratios = _ratios(target_probabilities, behavior_probabilities, clip)
    weights = ratios.prod(axis=1)
    returns = _discounted_returns(rewards, discount)
    estimate = float(np.mean(weights * returns))
    return {"estimate": estimate, "ess": _ess(weights)}


def weighted_is(
    rewards: np.ndarray,
    target_probabilities: np.ndarray,
    behavior_probabilities: np.ndarray,
    *,
    discount: float = 0.99,
    clip: float | None = None,
) -> dict[str, float]:
    ratios = _ratios(target_probabilities, behavior_probabilities, clip)
    weights = ratios.prod(axis=1)
    returns = _discounted_returns(rewards, discount)
    denominator = float(weights.sum())
    estimate = float(np.dot(weights, returns) / denominator) if denominator > 0 else float("nan")
    return {"estimate": estimate, "ess": _ess(weights)}


def per_decision_is(
    rewards: np.ndarray,
    target_probabilities: np.ndarray,
    behavior_probabilities: np.ndarray,
    *,
    discount: float = 0.99,
    clip: float | None = None,
) -> dict[str, float]:
    ratios = _ratios(target_probabilities, behavior_probabilities, clip)
    cumulative = np.cumprod(ratios, axis=1)
    discounts = discount ** np.arange(rewards.shape[1])
    estimate = float(np.mean(np.sum(cumulative * rewards * discounts[None, :], axis=1)))
    return {"estimate": estimate, "ess": _ess(cumulative[:, -1])}


def weighted_per_decision_is(
    rewards: np.ndarray,
    target_probabilities: np.ndarray,
    behavior_probabilities: np.ndarray,
    *,
    discount: float = 0.99,
    clip: float | None = None,
) -> dict[str, float]:
    ratios = _ratios(target_probabilities, behavior_probabilities, clip)
    cumulative = np.cumprod(ratios, axis=1)
    discounts = discount ** np.arange(rewards.shape[1])
    estimate = 0.0
    for step in range(rewards.shape[1]):
        weights = cumulative[:, step]
        total = float(weights.sum())
        if total > 0:
            estimate += float(discounts[step] * np.dot(weights, rewards[:, step]) / total)
    return {"estimate": estimate, "ess": _ess(cumulative[:, -1])}


def _ratios(target: np.ndarray, behavior: np.ndarray, clip: float | None) -> np.ndarray:
    target = np.asarray(target, dtype=np.float64)
    behavior = np.asarray(behavior, dtype=np.float64)
    if target.shape != behavior.shape or target.ndim != 2:
        raise ValueError("Target and behavior probabilities must be equal-shape episode-by-step arrays")
    if np.any(behavior <= 0) or np.any(target < 0):
        raise ValueError("Logged-action probabilities must be positive under behavior and nonnegative under target")
    ratios = target / behavior
    return np.minimum(ratios, clip) if clip is not None else ratios


def _discounted_returns(rewards: np.ndarray, discount: float) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.ndim != 2:
        raise ValueError("Rewards must be an episode-by-step array")
    return rewards @ (discount ** np.arange(rewards.shape[1]))


def _ess(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    denominator = float(np.square(weights).sum())
    return float(weights.sum() ** 2 / denominator) if denominator > 0 else 0.0
