"""Reference implementation of the RV02R conditional-recursive state update."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..errors import ReleaseContractError


@dataclass(frozen=True, slots=True)
class LoggedTransition:
    sequence_key: str
    relative_index: int
    history: tuple[tuple[float, ...], ...]
    history_mask: tuple[tuple[int, ...], ...]
    history_recency: tuple[tuple[float, ...], ...]
    action: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RecursivePrediction:
    sequence_key: str
    relative_index: int
    segment_horizon: int
    mean: tuple[float, ...]
    scale: tuple[float, ...] | None


Predictor = Callable[
    [tuple[tuple[float, ...], ...], tuple[tuple[int, ...], ...], tuple[tuple[float, ...], ...], tuple[float, ...]],
    tuple[Sequence[float], Sequence[float] | None],
]


def conditional_recursive_rollout(
    transitions: Sequence[LoggedTransition],
    predictor: Predictor,
) -> list[RecursivePrediction]:
    """Roll out maximal consecutive segments without using later logged values.

    Each segment starts from its first logged pre-action history. Later values are
    prior predictions, while logged masks, recencies, and actions remain fixed.
    A relative-index gap starts a new segment.
    """
    output: list[RecursivePrediction] = []
    last_sequence = ""
    last_relative: int | None = None
    recursive_history: tuple[tuple[float, ...], ...] | None = None
    prior_mean: tuple[float, ...] | None = None
    horizon = 0
    for transition in transitions:
        _validate_transition(transition)
        consecutive = transition.sequence_key == last_sequence and last_relative is not None and transition.relative_index == last_relative + 1
        if not consecutive:
            recursive_history = transition.history
            horizon = 1
        else:
            if recursive_history is None or prior_mean is None:
                raise ReleaseContractError("Recursive predecessor state is unavailable")
            recursive_history = recursive_history[1:] + (prior_mean,)
            horizon += 1
        mean_raw, scale_raw = predictor(
            recursive_history,
            transition.history_mask,
            transition.history_recency,
            transition.action,
        )
        mean = tuple(float(value) for value in mean_raw)
        if len(mean) != len(recursive_history[-1]):
            raise ReleaseContractError("Recursive predictor returned the wrong feature count")
        scale = None if scale_raw is None else tuple(float(value) for value in scale_raw)
        if scale is not None and (len(scale) != len(mean) or any(value <= 0.0 for value in scale)):
            raise ReleaseContractError("Recursive predictor scales must be positive and feature-aligned")
        output.append(RecursivePrediction(transition.sequence_key, transition.relative_index, horizon, mean, scale))
        last_sequence = transition.sequence_key
        last_relative = transition.relative_index
        prior_mean = mean
    return output


def _validate_transition(transition: LoggedTransition) -> None:
    if not transition.sequence_key or transition.relative_index < 0 or len(transition.history) < 2:
        raise ReleaseContractError("Invalid recursive transition identity or history")
    width = len(transition.history[0])
    if width == 0 or any(len(row) != width for row in transition.history):
        raise ReleaseContractError("Recursive histories must be non-empty rectangular arrays")
    if len(transition.history_mask) != len(transition.history) or len(transition.history_recency) != len(transition.history):
        raise ReleaseContractError("Logged mask and recency histories must align with state history")
    if any(len(row) != width for row in transition.history_mask) or any(len(row) != width for row in transition.history_recency):
        raise ReleaseContractError("Logged masks and recencies must match the state feature width")
