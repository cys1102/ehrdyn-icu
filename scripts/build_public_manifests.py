#!/usr/bin/env python3
"""Regenerate the paper-to-public-evidence manifests from frozen aggregate files."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    ("kdd2027_sepsis_vasopressor_3bin", "primary", "vasopressor__three_bin", 27236, 490248),
    ("kdd2027_respiratory_peep_5bin", "primary", "peep_setting__five_bin", 16389, 295002),
    ("kdd2027_aki_diuretic_rrt_factorized_3bin", "primary", "factorized_2_action_three_bin", 16453, 296154),
    ("kdd2027_af_rate_anticoag_compact_3bin", "primary", "compact_joint_2_action", 14580, 262440),
    ("kdd2027_hf_diuretic_binary", "primary", "diuretic__binary", 32552, 585936),
    ("kdd2027_ami_hemodynamic_compact_3bin", "extended", "compact_joint_2_action", 5055, 90990),
    ("kdd2027_shock_fluid_bolus_binary", "extended", "fluid_bolus__binary", 30563, 550134),
)


def main() -> None:
    contracts = ROOT / "contracts"
    with (contracts / "paper_task_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ("task_id", "paper_role", "config_path", "clinical_packet_path", "primary_action_view", "evidence_task_selector", "expected_episodes", "expected_windows")
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for task_id, role, action, episodes, windows in TASKS:
            writer.writerow({
                "task_id": task_id,
                "paper_role": role,
                "config_path": f"configs/tasks/{task_id}.json",
                "clinical_packet_path": f"clinical_review/core_task_packets/{task_id}.md",
                "primary_action_view": action,
                "evidence_task_selector": task_id,
                "expected_episodes": episodes,
                "expected_windows": windows,
            })

    source = ROOT / "evidence" / "core" / "horizon_rank_stability.csv"
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    fields = ("contract_id", "task_id", "config_path", "action_view", "action_level", "timing_convention", "frozen_reference", "leakage_negative_control", "leaderboard_selector")
    with (contracts / "paper_contract_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item["contract_id"]):
            task_id = row["frozen_task_id"]
            writer.writerow({
                "contract_id": row["contract_id"],
                "task_id": task_id,
                "config_path": f"configs/tasks/{task_id}.json",
                "action_view": row["action_view"],
                "action_level": row["action_level"],
                "timing_convention": row["timing_convention"],
                "frozen_reference": row["frozen_reference"],
                "leakage_negative_control": row["leakage_negative_control"],
                "leaderboard_selector": f"contract_id={row['contract_id']}",
            })


if __name__ == "__main__":
    main()
