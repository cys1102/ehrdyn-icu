from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from .errors import ReleaseContractError


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def feature_order_sha256(config: dict[str, object], release_root: Path) -> str:
    relative = config.get("feature_dictionary")
    if not isinstance(relative, str):
        raise ReleaseContractError("Task feature dictionary is unavailable")
    path = release_root / relative
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    try:
        ordered = sorted(rows, key=lambda row: int(row["feature_index"]))
        names = [row["feature_name"] for row in ordered]
    except (KeyError, ValueError) as error:
        raise ReleaseContractError("Feature dictionary does not define an ordered feature surface") from error
    if not names or len(names) != len(set(names)):
        raise ReleaseContractError("Feature order must be nonempty and unique")
    payload = ("\n".join(names) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
