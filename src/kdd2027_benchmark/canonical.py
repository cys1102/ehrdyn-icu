from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path

from .errors import ReleaseContractError


DECIMAL_PLACES = 12
QUANTUM = Decimal("1e-12")
ABSOLUTE_SEMANTIC_TOLERANCE = 5e-12


class NonFiniteConstant:
    def __init__(self, token: str):
        self.token = token


def load_strict_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=NonFiniteConstant)
    except OSError as error:
        raise ReleaseContractError(f"Cannot read JSON document: {error}") from error
    except json.JSONDecodeError as error:
        raise ReleaseContractError(
            f"JSON validation failed at / [json]: line {error.lineno} column {error.colno}: {error.msg}"
        ) from error


def canonical_bytes(value: object, *, trailing_newline: bool = True) -> bytes:
    text = _encode(value)
    if trailing_newline:
        text += "\n"
    return text.encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value, trailing_newline=False)).hexdigest()


def write_canonical_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def write_canonical_jsonl(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_bytes(row) for row in rows))


def _encode(value: object) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ReleaseContractError("Canonical JSON rejected a nonfinite value")
        rounded = Decimal(str(value)).quantize(QUANTUM, rounding=ROUND_HALF_EVEN)
        if rounded == 0:
            return "0"
        return format(rounded, "f").rstrip("0").rstrip(".")
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, list | tuple):
        return "[" + ",".join(_encode(item) for item in value) + "]"
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ReleaseContractError("Canonical JSON object keys must be strings")
        return "{" + ",".join(
            _encode(key) + ":" + _encode(value[key]) for key in sorted(value)
        ) + "}"
    raise ReleaseContractError(f"Canonical JSON does not support {type(value).__name__}")
