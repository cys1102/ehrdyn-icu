from __future__ import annotations

from .errors import ReleaseContractError

SPLIT_CONTRACT_HASH = "029e7452ae2ff99a974a76d31f0b2efa618b3412d071e17e4ef7dda36fb6b987"


def deterministic_split(entity_key: str) -> str:
    try:
        subject_key_internal = int(entity_key)
    except ValueError as error:
        raise ReleaseContractError("Internal split key must be an integer") from error
    bucket = (subject_key_internal * 1103515245 + 12345) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"
