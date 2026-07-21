#!/usr/bin/env python3
"""Aggregate-only memory probe for the AR6 and AR7 retained-frame designs."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import resource
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

reconstruction = importlib.import_module(
    "kdd2027_benchmark.current_five_task.reconstruct"
)


def _generate(path: Path, rows: int, write_rows: int) -> None:
    header = True
    for start in range(0, rows, write_rows):
        stop = min(rows, start + write_rows)
        order = np.arange(start, stop, dtype=np.int64)
        frame = pd.DataFrame({
            "stay_id": order % 50_003 + 1,
            "charttime": pd.Timestamp("2020-01-01") + pd.to_timedelta(order % 100_000, unit="s"),
            "itemid": np.where(order % 2, 220_052, 220_045),
            "valuenum": (order % 10_000).astype(np.float64) / 100.0,
            "valueuom": np.where(order % 3, "mmHg", "percent"),
            "__source_order": order,
        })
        frame.to_csv(path, mode="w" if header else "a", header=header, index=False)
        header = False


def _digest(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame.itertuples(index=False, name=None):
        digest.update("|".join(map(str, row)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _legacy(path: Path, chunk_rows: int) -> pd.DataFrame:
    retained: list[pd.DataFrame] = []
    source_order = 0
    for chunk in pd.read_csv(path, chunksize=chunk_rows, low_memory=False):
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="raise")
        chunk["__source_order"] = np.arange(source_order, source_order + len(chunk), dtype=np.int64)
        source_order += len(chunk)
        retained.append(chunk)
    return pd.concat(retained, ignore_index=True).sort_values(
        ["stay_id", "charttime", "__source_order"], kind="stable"
    ).reset_index(drop=True)


def _bounded(path: Path, chunk_rows: int, workspace: Path) -> pd.DataFrame:
    audit = reconstruction._StageResourceAudit(workspace)
    reconstruction._ACTIVE_TEMPORARY_DIRECTORY = workspace
    reconstruction._ACTIVE_STAGE_AUDIT = audit
    store = reconstruction._PartitionStore(workspace, "stress-parts")
    source_order = 0
    try:
        for chunk in pd.read_csv(path, chunksize=chunk_rows, low_memory=False):
            chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="raise")
            chunk["__source_order"] = np.arange(source_order, source_order + len(chunk), dtype=np.int64)
            source_order += len(chunk)
            store.append(chunk)
        return store.materialize(
            ["stay_id", "charttime", "itemid", "valuenum", "valueuom", "__source_order"],
            ("stay_id", "charttime"),
        ).reset_index(drop=True)
    finally:
        reconstruction._ACTIVE_STAGE_AUDIT = None
        reconstruction._ACTIVE_TEMPORARY_DIRECTORY = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("ar6", "ar7"), required=True)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--chunk-rows", type=int, default=25_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rows <= 0 or args.chunk_rows <= 0:
        raise ValueError("rows and chunk rows must be positive")
    with tempfile.TemporaryDirectory(prefix="kdd220ar7-stress-") as directory:
        root = Path(directory)
        source = root / "generated_chartevents.csv"
        _generate(source, args.rows, args.chunk_rows)
        frame = (
            _legacy(source, args.chunk_rows)
            if args.mode == "ar6"
            else _bounded(source, args.chunk_rows, root / "bounded-workspace")
        )
        payload = {
            "mode": args.mode,
            "input_rows": args.rows,
            "retained_rows": int(len(frame)),
            "parser_chunk_rows": args.chunk_rows,
            "canonical_row_digest": _digest(frame),
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        }
        args.output.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
