#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kdd2027_benchmark.canonical import write_canonical_json  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        raw = Path(directory) / "raw.json"
        subprocess.run([
            "uv", "export", "--frozen", "--no-dev", "--no-emit-project",
            "--format", "cyclonedx1.5", "--output-file", str(raw),
        ], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        document = json.loads(raw.read_text(encoding="utf-8"))
    document.pop("serialNumber", None)
    metadata = document.get("metadata", {})
    if isinstance(metadata, dict):
        metadata.pop("timestamp", None)
        metadata["properties"] = [{
            "name": "ehrdyn-icu:uv-lock-sha256",
            "value": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
        }]
    write_canonical_json(ROOT / "sbom/kdd215.cdx.json", document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
