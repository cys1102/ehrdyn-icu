# EHRDyn-ICU decision-evaluation benchmark

This directory is the anonymous release surface for the paper
*A Decision-Evaluation Benchmark for EHR World Models and Offline
Reinforcement Learning*.

It is intentionally separate from the historical forecasting benchmark and
the five-task RV release candidate at the repository root. Nothing in those
older lineages is relabeled as a decision-benchmark result.

## What is released

- six EHR world-model task contracts;
- four known-value policy-evaluation extensions (respiratory, shock,
  AKI--RRT, and heart failure);
- all 2,448 aggregate known-value policy rows used by the paper;
- all 16,128 reference and task-matched OPE tuple metrics and dispositions;
- the anonymous manuscript source and compiled PDF;
- a portable contract/evidence validator; and
- a synthetic known-value smoke test for exact planning, common random
  numbers, support masking, null-response invariance, and basic OPE formulas.

The paper-to-code mapping is in [`CODE_MAP.md`](CODE_MAP.md), and the exact
unrestricted versus credentialed boundary is in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). Byte-identical
world-model architecture and exact-evaluator snapshots are under
`reference_code/`, with hashes in `reference_code/source_manifest.csv`.

No MIMIC-IV rows, identifiers, split membership, exact timestamps,
trajectories, row-level predictions, or checkpoints are included.

## Clean-room quick start

Python 3.11 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

ehrdyn-icu decision-validate --root .
ehrdyn-icu decision-smoke --output /tmp/ehrdyn-decision-smoke.json --seed 3408
python -m unittest tests.test_decision
ehrdyn-icu scan-release --root .
ehrdyn-icu verify-checksums --root .
```

The smoke test checks executable invariants only; it does not reproduce paper
numbers or use EHR data. Numerical paper evidence is validated against
`decision/evidence/manifest.csv`.

## Evidence layers

The release keeps four layers distinct:

1. EHR world-model prediction and uncertainty on development roles;
2. retrospective action support, collapse, and observability diagnostics;
3. true policy returns in constructed known-value environments; and
4. exact OPE tuple authorization within those environments.

No task completes the trained-target-policy and post-training overlap gates
needed for retrospective EHR policy-value scoring. The release therefore does
not contain an EHR policy leaderboard.

## Credentialed reconstruction

MIMIC-IV reconstruction must occur in an authorized environment. Public code
and task contracts define the inputs and aggregate outputs, while restricted
arrays remain local. Synthetic clean-room success is not presented as
credentialed numerical parity.
