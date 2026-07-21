from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

from kdd2027_benchmark.current_five_task.contracts import (
    BIN_HOURS,
    EPISODE_BINS,
    EPISODE_WINDOW_HOURS,
    LONGEST_RECURSIVE_TARGET_HOURS,
    POST_ANCHOR_HOURS,
    PRE_ANCHOR_HOURS,
    RAW_EXTRACTION_BINS,
    RAW_EXTRACTION_POST_HOURS,
    SEPSIS_MAX_ANCHOR_SHIFT_HOURS,
    ContractError,
    eligible_transition_indices,
    episode_interface_indices,
    extraction_post_hours,
)
from kdd2027_benchmark.current_five_task.runtime_config import load_runtime_config


ROOT = Path(__file__).resolve().parents[1]


class KDD220ATemporalContractTests(unittest.TestCase):
    def test_source_closed_temporal_invariants(self) -> None:
        config = load_runtime_config()
        self.assertEqual(BIN_HOURS, 4)
        self.assertEqual((PRE_ANCHOR_HOURS, POST_ANCHOR_HOURS), (24, 48))
        self.assertEqual((EPISODE_WINDOW_HOURS, EPISODE_BINS), (72, 18))
        self.assertEqual(RAW_EXTRACTION_POST_HOURS, SEPSIS_MAX_ANCHOR_SHIFT_HOURS + POST_ANCHOR_HOURS)
        self.assertEqual(RAW_EXTRACTION_BINS, 30)
        self.assertEqual(LONGEST_RECURSIVE_TARGET_HOURS, 44)
        self.assertEqual(config["temporal"]["raw_rv01r_post_base_anchor_extraction_hours"], 96)

    def test_raw_buffer_and_final_episode_are_distinct(self) -> None:
        base = pd.Timestamp("2020-01-01 00:00:00")
        self.assertEqual(len(episode_interface_indices(base, base)), 18)
        shifted = episode_interface_indices(base, base + pd.Timedelta(hours=48))
        self.assertEqual((shifted[0], shifted[-1]), (12, 29))
        with self.assertRaises(ContractError):
            episode_interface_indices(base, base + pd.Timedelta(hours=52))
        self.assertEqual(extraction_post_hours("heart_failure"), 48)
        self.assertEqual(extraction_post_hours("respiratory_support"), 96)

    def test_final_transitions_are_stay_bounded_and_capped_at_48h(self) -> None:
        anchor = pd.Timestamp("2020-01-02 00:00:00")
        full = eligible_transition_indices(anchor, anchor - pd.Timedelta(hours=24), anchor + pd.Timedelta(hours=96))
        self.assertEqual(full, tuple(range(11)))
        self.assertEqual((max(full) + 1) * BIN_HOURS, LONGEST_RECURSIVE_TARGET_HOURS)
        short = eligible_transition_indices(anchor, anchor - pd.Timedelta(hours=24), anchor + pd.Timedelta(hours=28))
        self.assertLess(len(short), len(full))

    def test_runtime_config_has_exact_task_router(self) -> None:
        router = load_runtime_config()["lineage_router"]
        self.assertEqual(router, {
            "sepsis": "kdd121_large_lineage",
            "respiratory_support": "rv01r_kdd097_compact_lineage",
            "shock": "rv01r_kdd097_compact_lineage",
            "aki": "rv01r_kdd097_compact_lineage",
            "heart_failure": "kdd121_large_lineage",
        })

    def test_runtime_source_has_no_result_or_private_path_dependency(self) -> None:
        source = ROOT / "src" / "kdd2027_benchmark" / "current_five_task"
        forbidden = ("results/", "artifacts/", "/" + "home" + "/", "episode_manifest")
        hits = []
        for path in source.glob("*"):
            if path.suffix not in {".py", ".json"}:
                continue
            text = path.read_text(encoding="utf-8")
            hits.extend((path.name, token) for token in forbidden if token in text)
        self.assertEqual(hits, [])

    def test_modified_config_fails_closed(self) -> None:
        config = json.loads((ROOT / "src/kdd2027_benchmark/current_five_task/runtime_config.json").read_text())
        config["temporal"]["raw_rv01r_post_base_anchor_extraction_hours"] = 48
        temporary = ROOT / "tests" / ".kdd220a-invalid-runtime.json"
        try:
            temporary.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(Exception):
                load_runtime_config(temporary)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
