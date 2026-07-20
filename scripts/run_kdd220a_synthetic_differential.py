#!/usr/bin/env python3
"""Run aggregate-only KDD220A differential checks on synthetic flat files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import sys
import tempfile
from pathlib import Path

import numpy as np

from kdd2027_benchmark.current_five_task import contracts as public_contract
from kdd2027_benchmark.current_five_task.reconstruct import (
    build_anchors,
    build_arrays,
    build_transitions,
    encode_actions,
    finalize_sepsis_anchors,
    load_core,
    validate_layout,
)
from tests.test_kdd217ar3a import make_fixture


EXPECTED_SOURCE_HASHES = {
    "kdd_rv01r/construction.py": "f0c586f60102b0048d5c682a6bc86757bc71e090941e1d9d608c5f1224b8e39c",
    "kdd_rv01r/sources.py": "fb50adab73b2e6d795c7dfbd40806b74a8846630b53199ee8ada89514836e88f",
    "kdd_benchmark_discovery/kdd097_contract.py": "324f465a15f4534de4277c1adc2fe9ad631619916bdc05d6c09b81a5374ca4f5",
    "kdd_benchmark_discovery/kdd152v2a_repaired_interfaces.py": "6402ee9eb64526ab41f122584342011c13bf709c9846efa43e5f16eecaeb66a7",
    "kdd_benchmark_discovery/kdd200_temporal_audit.py": "f155bc91335e912e075379909e48de9b53e60afea7c781e2faeb875f589706a1",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _post_kdd201(author_temporal: object, anchors: object, transitions: object):
    source_by_episode = anchors.set_index("episode_idx")["anchor_source"]
    keep = []
    for row in transitions.itertuples(index=False):
        source = source_by_episode.loc[int(row.episode_idx)]
        if source in author_temporal.ANCHOR_CONTRACTS:
            keep.append(author_temporal.source_disposition(source) != 3 or int(row.relative_transition) > 0)
        else:
            keep.append(True)
    return transitions.loc[np.asarray(keep, dtype=bool)].reset_index(drop=True)


def _author_actions(task: str, transitions, arrays, author_action) -> np.ndarray:
    episode = transitions["episode_idx"].to_numpy(int)
    step = transitions["action_idx"].to_numpy(int)
    roles = transitions["role"].astype(str).to_numpy()
    if task in {"sepsis", "shock"}:
        left, right = arrays["fluid"][episode, step], arrays["vaso"][episode, step]
        lo = ro = np.ones(len(step), bool)
    elif task == "respiratory_support":
        left, lo = arrays["peep"][episode, step], arrays["peep_observed"][episode, step]
        index = public_contract.FEATURE_INDEX["fio2"]
        right, ro = arrays["values"][episode, step, index], arrays["masks"][episode, step, index]
    elif task == "aki":
        return (arrays["diuretic"][episode, step] > 0).astype(np.int16) + 2 * (arrays["rrt"][episode, step] > 0).astype(np.int16)
    else:
        return (arrays["diuretic"][episode, step] > 0).astype(np.int16)
    left_edges = author_action.train_positive_edges(left, lo, roles)
    right_edges = author_action.train_positive_edges(right, ro, roles)
    return author_action.joint_codes(
        author_action.encode_five_levels(left, lo, left_edges),
        author_action.encode_five_levels(right, ro, right_edges),
        lo & ro,
    )


def _reward_and_terminal_gate(task: str, transitions, candidates, arrays, author_reward) -> tuple[bool, bool]:
    groups = list(transitions.groupby("episode_idx", sort=False, observed=True))
    maximum = max(len(group) for _, group in groups)
    shape = (len(groups), maximum, len(public_contract.FEATURE_NAMES))
    states = np.full(shape, np.nan, np.float32)
    targets = np.full(shape, np.nan, np.float32)
    state_masks = np.zeros(shape, bool)
    target_masks = np.zeros(shape, bool)
    valid = np.zeros((len(groups), maximum), bool)
    outcomes = np.zeros(len(groups), np.float32)
    identity = candidates.set_index("episode_idx")
    for row_index, (episode_id, group) in enumerate(groups):
        group = group.reset_index(drop=True)
        count = len(group)
        episode = group["episode_idx"].to_numpy(int)
        state = group["state_idx"].to_numpy(int)
        target = group["target_idx"].to_numpy(int)
        states[row_index, :count] = arrays["values"][episode, state]
        targets[row_index, :count] = arrays["values"][episode, target]
        state_masks[row_index, :count] = arrays["masks"][episode, state]
        target_masks[row_index, :count] = arrays["masks"][episode, target]
        valid[row_index, :count] = True
        outcomes[row_index] = float(identity.loc[int(episode_id)].mortality_90d)
    public_reward, public_mask, _ = public_contract.reward_components(
        task, states, state_masks, targets, target_masks, valid,
        outcomes if task in {"sepsis", "aki", "heart_failure"} else None,
    )
    if task in {"sepsis", "shock", "respiratory_support"}:
        author_task = "respiratory" if task == "respiratory_support" else task
        author_values, author_mask, names = author_reward._reward_values(
            author_task, public_contract.FEATURE_NAMES, states, state_masks,
            targets, target_masks, valid, outcomes if task == "sepsis" else None,
        )
        name = {
            "sepsis": "terminal_discharge_origin_90d_proxy",
            "shock": "shock_next_mbp_component",
            "respiratory_support": "resp_meddreamer_spo2_mbp",
        }[task]
        channel = names.index(name)
        reward_equal = np.array_equal(public_mask[..., 0], author_mask[..., channel]) and np.allclose(
            public_reward[..., 0], author_values[..., channel], rtol=0, atol=0
        )
    else:
        lengths = valid.sum(axis=1).astype(int)
        expected_mask = np.zeros_like(valid)
        expected_reward = np.zeros_like(valid, dtype=np.float32)
        for row_index, length in enumerate(lengths):
            expected_mask[row_index, length - 1] = True
            expected_reward[row_index, length - 1] = -1 if outcomes[row_index] > 0.5 else 1
        reward_equal = np.array_equal(public_mask[..., 0], expected_mask) and np.array_equal(public_reward[..., 0], expected_reward)
    terminal = np.zeros_like(valid)
    for row_index, length in enumerate(valid.sum(axis=1).astype(int)):
        terminal[row_index, length - 1] = True
    continuation = valid & ~terminal
    terminal_equal = bool(np.all(terminal.sum(axis=1) == 1) and not np.any(terminal & continuation))
    return bool(reward_equal), terminal_equal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for relative, expected in EXPECTED_SOURCE_HASHES.items():
        observed = _sha256(args.author_root / relative)
        if observed != expected:
            raise RuntimeError(f"author source hash mismatch: {relative}: {observed}")
    sys.path.insert(0, str(args.author_root))
    author_sources = importlib.import_module("kdd_rv01r.sources")
    author_construction = importlib.import_module("kdd_rv01r.construction")
    author_action = importlib.import_module("kdd_benchmark_discovery.kdd097_contract")
    author_reward = importlib.import_module("kdd_benchmark_discovery.kdd152v2a_repaired_interfaces")
    author_temporal = importlib.import_module("kdd_benchmark_discovery.kdd200_temporal_audit")
    with tempfile.TemporaryDirectory() as directory:
        root = make_fixture(Path(directory))
        paths = validate_layout(root)
        stays, diagnoses = load_core(paths)
        candidates = build_anchors(paths, stays, diagnoses)
        arrays = build_arrays(paths, candidates)
        dense = author_sources.DenseConstruction(
            candidates, arrays["values"], arrays["masks"], arrays["fluid"], arrays["vaso"],
            {"peep": arrays["peep"], "diuretic": arrays["diuretic"], "rrt": arrays["rrt"]},
            {"peep": arrays["peep_observed"], "diuretic": np.ones_like(arrays["diuretic"], bool), "rrt": np.ones_like(arrays["rrt"], bool)},
            {},
        )
        config = {"bin_hours": 4, "post_anchor_hours": 48, "sepsis": {"minimum_observed_sofa_domains": 2, "sofa_increase_min": 2}}
        author_candidates, failures = author_construction._finalize_anchors(dense, config)
        public_candidates = finalize_sepsis_anchors(candidates, arrays)
        if failures or not public_candidates[["task_id", "episode_idx", "anchor_time"]].reset_index(drop=True).equals(
            author_candidates[["task_id", "episode_idx", "anchor_time"]].reset_index(drop=True)
        ):
            raise SystemExit("anchor differential failed")
        author_transitions, failures = author_construction._build_transitions(author_candidates, arrays["values"].shape[1], config)
        if failures:
            raise SystemExit("author transition construction excluded synthetic episodes")
        author_transitions = _post_kdd201(author_temporal, author_candidates, author_transitions)
        public_transitions = build_transitions(public_candidates)
        columns = ["task", "role", "episode_idx", "relative_transition", "state_idx", "action_idx", "target_idx"]
        rows = []
        for task in public_contract.TASKS:
            left = public_transitions[public_transitions["task"].eq(task)].reset_index(drop=True)
            right = author_transitions[author_transitions["task"].eq(task)].reset_index(drop=True)
            transitions_equal = left[columns].equals(right[columns])
            public_actions, _ = encode_actions(task, left, arrays)
            author_actions = _author_actions(task, right, arrays, author_action)
            action_equal = np.array_equal(public_actions, author_actions)
            reward_equal, terminal_equal = _reward_and_terminal_gate(task, left, public_candidates, arrays, author_reward)
            interface = all(
                len(public_contract.episode_interface_indices(row.base_anchor_time, row.anchor_time)) == public_contract.EPISODE_BINS
                for row in public_candidates[public_candidates["task_id"].eq(task)].itertuples(index=False)
            )
            passed = transitions_equal and action_equal and reward_equal and terminal_equal and interface
            rows.append({
                "task": task,
                "inclusion_and_anchor": "pass",
                "role": "pass",
                "ordered_transitions": "pass" if transitions_equal else "fail",
                "safe_interface_18_bins": "pass" if interface else "fail",
                "action_class": "pass" if action_equal else "fail",
                "reward_and_mask": "pass" if reward_equal else "fail",
                "termination_and_continuation": "pass" if terminal_equal else "fail",
                "maximum_absolute_difference": 0 if passed else "NA",
                "status": "pass" if passed else "fail",
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if any(row["status"] != "pass" for row in rows):
        raise SystemExit("five-task synthetic differential failed")


if __name__ == "__main__":
    main()
