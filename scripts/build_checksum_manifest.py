#!/usr/bin/env python3
"""Regenerate the release checksum manifest from the current public file set."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IGNORED_PARTS = {".git", ".venv", ".pytest_cache", "__pycache__", "build", "dist"}


def main() -> None:
    lines = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.sha256" or _ignored(path):
            continue
        relative = path.relative_to(ROOT).as_posix()
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}")
    (ROOT / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ignored(path: Path) -> bool:
    return bool(set(path.parts) & IGNORED_PARTS) or any(part.endswith(".egg-info") for part in path.parts)


if __name__ == "__main__":
    main()
