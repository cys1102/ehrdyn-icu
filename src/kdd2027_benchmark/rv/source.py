"""Verify the exact frozen RV01R/RV02R backend source without importing it."""

from __future__ import annotations

import csv
import hashlib
import subprocess
from pathlib import Path

from ..errors import ReleaseContractError


def verify_backend_source(backend_root: Path, manifest: Path | None = None) -> dict[str, object]:
    manifest_path = manifest or Path(__file__).resolve().parent / "contracts/source_manifest.csv"
    try:
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise ReleaseContractError(f"Cannot read successor source manifest: {error}") from error
    if not rows:
        raise ReleaseContractError("Successor source manifest is empty")
    commits = {row["commit"] for row in rows}
    repositories = {row["source_repository"] for row in rows}
    if len(commits) != 1:
        raise ReleaseContractError("Successor source manifest must pin one backend commit")
    if len(repositories) != 1:
        raise ReleaseContractError("Successor source manifest must pin one backend repository")
    expected_commit = next(iter(commits))
    expected_repository = next(iter(repositories))
    actual_commit = _git(backend_root, "rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise ReleaseContractError(
            f"Frozen backend Git HEAD mismatch: expected {expected_commit}, found {actual_commit}"
        )
    actual_repository = _git(backend_root, "remote", "get-url", "origin")
    if _canonical_repository(actual_repository) != _canonical_repository(expected_repository):
        raise ReleaseContractError("Frozen backend origin repository mismatch")
    tracked = set(_git(backend_root, "ls-tree", "-r", "--name-only", expected_commit).splitlines())
    for row in rows:
        if row["path"] not in tracked:
            raise ReleaseContractError(f"Frozen backend source is not tracked at pinned commit: {row['path']}")
        path = backend_root / row["path"]
        if not path.is_file():
            raise ReleaseContractError(f"Frozen backend source is missing: {row['path']}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != row["sha256"]:
            raise ReleaseContractError(f"Frozen backend source hash mismatch: {row['path']}")
    return {
        "source_repository": expected_repository,
        "expected_commit": expected_commit,
        "actual_commit": actual_commit,
        "source_files_verified": len(rows),
        "tracked_at_pinned_commit": len(rows),
        "hash_mismatches": 0,
        "pass": True,
    }


def _git(root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as error:
        raise ReleaseContractError(f"Cannot execute Git for frozen backend verification: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown Git error"
        raise ReleaseContractError(f"Cannot verify frozen backend Git repository: {detail}")
    return completed.stdout.strip()


def _canonical_repository(value: str) -> str:
    normalized = value.strip().lower()
    if normalized.startswith("git@github.com:"):
        normalized = "https://github.com/" + normalized.removeprefix("git@github.com:")
    if normalized.startswith("ssh://git@github.com/"):
        normalized = "https://github.com/" + normalized.removeprefix("ssh://git@github.com/")
    return normalized.removesuffix("/").removesuffix(".git")
