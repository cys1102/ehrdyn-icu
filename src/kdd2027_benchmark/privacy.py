from __future__ import annotations

import csv
import hashlib
import io
import re
import tarfile
import zipfile
from pathlib import Path

from .errors import ReleaseContractError

RESTRICTED_SUFFIXES = {".pt", ".pth", ".ckpt", ".npy", ".npz", ".parquet", ".feather", ".jsonl", ".pyc", ".pkl"}
IGNORED_BUILD_PARTS = {".git", ".venv", ".pytest_cache", "__pycache__", "build", "dist"}
RESTRICTED_HEADERS = {
    "subject_id",
    "stay_id",
    "hadm_id",
    "timestamp",
    "charttime",
    "storetime",
    "raw_text",
    "probability_vector",
    "logit_vector",
    "patient_id",
    "episode_id",
    "split_membership",
    "prediction_vector",
}
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)(?:password|api[_-]?key|access[_-]?token)\s*=\s*[^<\s]+"),
    re.compile(r"/(?:home|data)/[^\s`]+"),
)


def scan_release(root: Path) -> dict[str, int | bool]:
    findings: list[str] = []
    files = [path for path in root.rglob("*") if path.is_file() and not _ignored_build_file(path)]
    for path in files:
        relative = path.relative_to(root)
        if path.suffix.lower() in RESTRICTED_SUFFIXES:
            findings.append(f"restricted_suffix:{relative}")
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            findings.append(f"secret_or_local_path:{relative}")
        if path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as handle:
                headers = set(next(csv.reader(handle), ()))
            if headers & RESTRICTED_HEADERS:
                findings.append(f"restricted_csv_header:{relative}")
    if findings:
        raise ReleaseContractError("Release privacy scan failed: " + ";".join(findings))
    return {"files_scanned": len(files), "findings": 0, "pass": True}


def verify_checksums(root: Path) -> dict[str, int | bool]:
    manifest = root / "MANIFEST.sha256"
    checked = 0
    listed: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", maxsplit=1)
        listed.add(relative)
        path = root / relative
        if path.resolve().parent != root.resolve() and root.resolve() not in path.resolve().parents:
            raise ReleaseContractError("Manifest path escapes release root")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ReleaseContractError(f"Checksum mismatch: {relative}")
        checked += 1
    if checked == 0:
        raise ReleaseContractError("Checksum manifest is empty")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "MANIFEST.sha256" and not _ignored_build_file(path)
    }
    if actual != listed:
        raise ReleaseContractError("Checksum manifest file set does not match release contents")
    return {"files_checked": checked, "mismatches": 0, "pass": True}


def audit_distribution_archives(dist_root: Path) -> dict[str, object]:
    archives = sorted(
        path for path in dist_root.iterdir() if path.is_file() and (path.suffix == ".whl" or path.name.endswith(".tar.gz"))
    )
    if not archives:
        raise ReleaseContractError("No wheel or source distribution was found")
    files_scanned = 0
    findings: list[str] = []
    digests: dict[str, str] = {}
    for archive in archives:
        digests[archive.name] = hashlib.sha256(archive.read_bytes()).hexdigest()
        if archive.suffix == ".whl":
            with zipfile.ZipFile(archive) as handle:
                members = [(item.filename, handle.read(item)) for item in handle.infolist() if not item.is_dir()]
        else:
            with tarfile.open(archive, mode="r:gz") as handle:
                members = []
                for item in handle.getmembers():
                    if not item.isfile():
                        continue
                    extracted = handle.extractfile(item)
                    if extracted is None:
                        continue
                    members.append((item.name, extracted.read()))
        for name, data in members:
            files_scanned += 1
            _scan_archive_member(archive.name, name, data, findings)
    if findings:
        raise ReleaseContractError("Distribution privacy scan failed: " + ";".join(findings))
    return {
        "archives_scanned": len(archives),
        "archive_files_scanned": files_scanned,
        "archive_sha256": digests,
        "findings": 0,
        "pass": True,
    }


def _scan_archive_member(archive: str, name: str, data: bytes, findings: list[str]) -> None:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        findings.append(f"archive_path_escape:{archive}:{name}")
        return
    if path.suffix.lower() in RESTRICTED_SUFFIXES:
        findings.append(f"restricted_suffix:{archive}:{name}")
        return
    text = data.decode("utf-8", errors="ignore")
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):
        findings.append(f"secret_or_local_path:{archive}:{name}")
    if path.suffix == ".csv":
        headers = set(next(csv.reader(io.StringIO(text)), ()))
        if headers & RESTRICTED_HEADERS:
            findings.append(f"restricted_csv_header:{archive}:{name}")


def _ignored_build_file(path: Path) -> bool:
    return bool(set(path.parts) & IGNORED_BUILD_PARTS) or any(part.endswith(".egg-info") for part in path.parts)
