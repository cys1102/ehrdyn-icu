# EHRDyn-ICU v1.0

EHRDyn-ICU is a frozen, multi-cohort benchmark contract for recorded ICU
trajectory forecasting and offline-RL diagnostics. The immutable scientific
contract identifier is `KDD2027-E060-4H-v1.0.0`; this repository is the
`KDD089` public artifact build.

This repository contains software, task definitions, synthetic fixtures, and
aggregate evidence. It does **not** contain MIMIC-IV rows, patient identifiers,
split membership, timestamps, trajectories, model checkpoints, row-level
predictions, or credentials.

## Benchmark Surfaces

The paper uses several intentionally separate evidence surfaces. Their exact
roles and non-pooling rules are defined in [CANONICAL_SURFACES.md](CANONICAL_SURFACES.md).
The offline-policy estimand and estimator conventions are defined in
[OPE_CONTRACT.md](OPE_CONTRACT.md). Clinical task definitions are versioned in
`configs/tasks/` and `task_cards/`; independent clinician review is tracked in
[CLINICAL_REVIEW.md](CLINICAL_REVIEW.md).

## Quick Start

Python 3.11 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

ehrdyn-icu validate-config --config-dir configs/tasks
ehrdyn-icu generate-fixture --output /tmp/ehrdyn_fixture.csv
ehrdyn-icu evaluate \
  --fixture /tmp/ehrdyn_fixture.csv \
  --task-config configs/tasks/sepsis_original25_reference.json \
  --output /tmp/ehrdyn_metrics.json
ehrdyn-icu validate-submission \
  --submission submission/leaderboard_submission_template.json \
  --config-dir configs/tasks
python -m unittest discover -s tests
ehrdyn-icu scan-release --root .
ehrdyn-icu verify-checksums --root .
```

The historical `kdd2027` command remains available as a compatibility alias.

## Repository Contents

- `configs/tasks/`: seven frozen task or exclusion contracts.
- `contracts/`: preprocessing and layered gate definitions.
- `dictionaries/`: feature, action, reward, metric, and baseline provenance.
- `src/`: aggregate evaluator, fixture, split, gate, privacy, and submission code.
- `fixtures/`: schema-compatible synthetic data only.
- `evidence/`: publication-safe aggregate dynamics, temporal, and policy diagnostics.
- `task_cards/`: human-readable task summaries.
- `schemas/` and `submission/`: leaderboard interfaces.
- `tests/`: clean-room, privacy, checksum, and contract tests.

## Credentialed Benchmark Execution

MIMIC-IV remains governed by PhysioNet credentialing and is not redistributed.
See [MIMIC_ACCESS.md](MIMIC_ACCESS.md). Extraction must run in an authorized
environment. Only aggregate evaluator outputs may be exported into this public
artifact.

## Evidence Boundary

The dynamics track evaluates one-step and conditional recursive forecasts under
logged future actions. It does not simulate outcomes under a new treatment
policy. The policy track diagnoses support and estimator sensitivity; it is not
a clinical policy leaderboard. See [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md)
and [ETHICS_AND_MISUSE.md](ETHICS_AND_MISUSE.md).

Unsupported uses include treatment recommendation, causal effect estimation,
counterfactual benefit claims, clinical deployment, and autonomous decisions.

## Citation

Citation metadata are in [CITATION.cff](CITATION.cff). Repository:
<https://github.com/cys1102/ehrdyn-icu>.

