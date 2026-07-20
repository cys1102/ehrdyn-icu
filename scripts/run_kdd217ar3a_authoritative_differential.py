#!/usr/bin/env python3
"""Compare the public pure contracts with a source-hash-verified author tree.

This development utility never reads MIMIC. Its output contains only synthetic
contract comparisons. The public reconstruction runtime does not import or
depend on the author tree.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kdd2027_benchmark.current_five_task import contracts as public


EXPECTED = {
    "kdd_rv01r/contracts.py": "ef88dbd98ed84c72eb7513a1acbceb50c81a01df910f605e9a303e356de8db5d",
    "kdd_benchmark_discovery/kdd097_contract.py": "324f465a15f4534de4277c1adc2fe9ad631619916bdc05d6c09b81a5374ca4f5",
    "kdd_benchmark_discovery/kdd152v2a_repaired_interfaces.py": "6402ee9eb64526ab41f122584342011c13bf709c9846efa43e5f16eecaeb66a7",
    "kdd_benchmark_discovery/kdd200_temporal_audit.py": "f155bc91335e912e075379909e48de9b53e60afea7c781e2faeb875f589706a1",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for relative, expected in EXPECTED.items():
        observed = digest(args.author_root / relative)
        if observed != expected:
            raise RuntimeError(f"author source hash mismatch: {relative}: {observed}")
    sys.path.insert(0, str(args.author_root))
    sys.path.insert(0, str(args.author_root / "scripts"))
    author_contract = importlib.import_module("kdd_rv01r.contracts")
    author_action = importlib.import_module("kdd_benchmark_discovery.kdd097_contract")
    author_reward = importlib.import_module("kdd_benchmark_discovery.kdd152v2a_repaired_interfaces")
    author_temporal = importlib.import_module("kdd_benchmark_discovery.kdd200_temporal_audit")
    rows: list[dict[str, object]] = []

    def exact(name: str, left: object, right: object) -> None:
        if isinstance(left, np.ndarray):
            equal = np.array_equal(left, right)
        else:
            equal = left == right
        rows.append({"comparison": name, "comparison_type": "exact", "status": "pass" if equal else "fail", "maximum_absolute_difference": 0 if equal else "NA"})

    for subject in (1, 2, 3, 17, 3408, 99999):
        expected = author_contract.subject_role(subject)
        if expected == "sealed_test": expected = "historical_other"
        exact(f"subject_role_{subject}", public.subject_role(subject), expected)
    anchor = pd.Timestamp("2020-01-02"); intime = pd.Timestamp("2020-01-01"); outtime = pd.Timestamp("2020-01-04")
    exact("eligible_transition_indices", public.eligible_transition_indices(anchor, intime, outtime), author_contract.eligible_transition_indices(anchor=anchor, intime=intime, outtime=outtime, bin_hours=4, post_anchor_hours=48))
    values = np.array([0, 1, 2, 3, 4, 999.0]); observed = np.ones(6, bool); roles = ["train"] * 5 + ["validation"]
    public_edges = public.fit_train_positive_edges(values, observed, roles); author_edges = author_action.train_positive_edges(values, observed, roles)
    exact("train_positive_edges", public_edges, author_edges)
    left_public = public.encode_five_levels(values, observed, public_edges); left_author = author_action.encode_five_levels(values, observed, author_edges)
    exact("encode_five_levels", left_public, left_author)
    exact("joint_codes", public.joint_codes(left_public, left_public, observed), author_action.joint_codes(left_author, left_author, observed))
    source = np.repeat(["sustained_hypotension", "vasopressor_support"], 3); relative = np.tile([0, 1, 2], 2)
    frame = pd.DataFrame({"anchor_source": source, "relative_transition": relative})
    public_keep = np.concatenate([public.apply_kdd201_temporal_repair(item, [step]) for item, step in zip(source, relative)])
    exact("kdd201_temporal_repair", public_keep, author_temporal.retained_mask(frame))

    shape = (2, 3, len(public.FEATURE_NAMES)); states = np.zeros(shape); targets = np.zeros(shape); masks = np.ones(shape, bool); valid = np.array([[1, 1, 0], [1, 1, 1]], bool); outcomes = np.array([0, 1])
    pub_reward, pub_mask, _ = public.reward_components("sepsis", states, masks, targets, masks, valid, outcomes)
    auth_reward, auth_mask, auth_names = author_reward._reward_values("sepsis", public.FEATURE_NAMES, states, masks, targets, masks, valid, outcomes)
    terminal = auth_names.index("terminal_discharge_origin_90d_proxy")
    exact("sepsis_terminal_reward", pub_reward[..., 0], auth_reward[..., terminal]); exact("sepsis_terminal_mask", pub_mask[..., 0], auth_mask[..., terminal])
    targets[..., public.FEATURE_INDEX["mbp"]] = 75
    pub_reward, pub_mask, _ = public.reward_components("shock", states, masks, targets, masks, valid, None)
    auth_reward, auth_mask, auth_names = author_reward._reward_values("shock", public.FEATURE_NAMES, states, masks, targets, masks, valid, None)
    primary = auth_names.index("shock_next_mbp_component")
    exact("shock_reward", pub_reward[..., 0], auth_reward[..., primary]); exact("shock_reward_mask", pub_mask[..., 0], auth_mask[..., primary])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    if any(row["status"] != "pass" for row in rows):
        raise SystemExit("one or more synthetic differential comparisons failed")


if __name__ == "__main__":
    main()
