#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from kdd2027_benchmark.canonical import write_canonical_json
from kdd2027_benchmark.full_direct_evaluator import evaluate_repaired_policy_batch
from kdd2027_benchmark.full_suite import environments, fixed_policy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    envs = environments(args.manifest)
    env = envs[0]
    result = evaluate_repaired_policy_batch(
        env, fixed_policy(env, "minimum"), 64, 1_993_171_901, 2_993_000_000
    )
    write_canonical_json(args.output, {
        "environment_count": len(envs),
        "first_environment_mechanism_sha256": env.mechanism_hash,
        "mean_return": result["mean_return"],
        "return_standard_error": result["return_se"],
        "terminal_emission_max": result["terminal_emission_max"],
        "unsupported_mass": result["unsupported_mass"],
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
