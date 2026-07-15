"""Exact subject-role hashing frozen by KDD-RV00R/RV01R."""

from __future__ import annotations

import hashlib

from ..errors import ReleaseContractError
from . import ROLE_SALT


def subject_role(entity_key: str) -> str:
    """Return the role without returning or persisting the local entity key."""
    try:
        subject_key_internal = int(entity_key)
    except ValueError as error:
        raise ReleaseContractError("Successor split key must be a decimal integer") from error
    digest = hashlib.sha256(f"{ROLE_SALT}{subject_key_internal}".encode()).digest()
    bucket = int.from_bytes(digest[:8], byteorder="big", signed=False) % 10_000
    if bucket < 7_000:
        return "train"
    if bucket < 8_500:
        return "validation"
    return "sealed_test"


def assert_disjoint(role_keys: dict[str, set[str]]) -> dict[str, int]:
    expected = {"train", "validation", "sealed_test"}
    if set(role_keys) != expected:
        raise ReleaseContractError("Role-key audit requires train, validation, and sealed_test")
    for left, right in (("train", "validation"), ("train", "sealed_test"), ("validation", "sealed_test")):
        if role_keys[left] & role_keys[right]:
            raise ReleaseContractError(f"Successor subject roles overlap: {left}/{right}")
    return {role: len(keys) for role, keys in role_keys.items()}
