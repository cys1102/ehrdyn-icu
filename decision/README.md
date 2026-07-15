# EHRDyn-ICU multi-cohort decision-evaluation benchmark

This directory is the anonymous release surface for the paper
*A Multi-Cohort Decision-Evaluation Benchmark for EHR World-Model
Components and Offline Reinforcement Learning*.

It is intentionally separate from the historical forecasting benchmark and
the five-task RV release candidate at the repository root. Nothing in those
older lineages is relabeled as a decision-benchmark result.

## What is released

- six EHR P/R/T-component task contracts and the complete 36-row
  cohort-by-transition-method matrix;
- an AI-Clinician-aligned K25 sepsis scaffold, labeled as an alignment
  surface rather than an exact reproduction;
- 24 heterogeneous exact-finite task--mechanism environments over respiratory,
  shock, AKI--RRT, and heart failure, each with dynamic-programming truth;
- the complete layer-specific inventory of six transition methods, five
  uncertainty-capable methods, 34 policy/planner labels, and nine OPE
  estimators;
- a cross-layer baseline atlas and cohort-specific uncertainty leader table;
- all 4,080 heterogeneous policy--seed rows, all 2,880 world-model--planner
  rows, all exploitation-gap rows, and the earlier adaptive sensitivity;
- all 2,448 historical monotone policy rows, retained as evaluator
  smoke/sensitivity evidence rather than the primary policy benchmark;
- 10,368 repeated-dataset coverage rows and 1,728 revised OPE dispositions,
  plus all 16,128 historical reference/task-matched diagnostic tuples;
- the 16-row EHR-to-known-value model-family bridge and all contract mappings;
- a source-verified representative literature landscape;
- the anonymous manuscript source and compiled PDF;
- a portable contract/evidence validator; and
- a synthetic known-value smoke test for exact planning, common random
  numbers, support masking, null-response invariance, and basic OPE formulas.

The paper-to-code mapping is in [`CODE_MAP.md`](CODE_MAP.md), and the exact
unrestricted versus credentialed boundary is in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). Byte-identical
world-model architecture and exact-evaluator snapshots are under
`reference_code/`, with original-source and packaged hashes in
`reference_code/source_manifest.csv`. Three credentialed runners replace only
machine-local default paths with explicit portable placeholders.

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
numbers or use EHR data. Numerical paper evidence, including every complete
baseline row, is checksum- and row-count-validated against
`decision/evidence/manifest.csv`.

## Evidence layers

The release keeps four layers distinct:

1. EHR P/R/T-component prediction and uncertainty on development roles;
2. retrospective action support, collapse, and observability diagnostics;
3. exact policy returns in heterogeneous constructed known-value environments;
4. repeated-dataset OPE calibration and authorization within known-value
   environments.

The older adaptive-composite and monotone families are explicitly secondary
construction/evaluator sensitivities. They are not used to claim a universal
policy winner.

No task completes the trained-target-policy and post-training overlap gates
needed for retrospective EHR policy-value scoring. The release therefore does
not contain an EHR policy leaderboard.

## Credentialed reconstruction

MIMIC-IV reconstruction must occur in an authorized environment. Public code
and task contracts define the inputs and aggregate outputs, while restricted
arrays remain local. Independent credentialed reconstruction means that a
second MIMIC-credentialed user regenerates aggregate results from a clean
clone; it is a reproducibility check, not external-site, clinical, or
generalization validation. Synthetic clean-room success is not presented as
credentialed numerical parity.
