# EHRDyn-ICU

EHRDyn-ICU is a frozen, multi-cohort benchmark contract for recorded ICU
trajectory forecasting and offline-RL readiness diagnostics. Frozen scientific
contract identifiers remain in the machine-readable task files. The public
repository includes the credentialed-construction path, paper-to-artifact
manifests, action-cardinality parity checks, overlap-aware action construction,
and the KDD187 Rocky Linux author-side reconstruction contract. KDD187 does not
convert the externally blocked KDD182 run into independent evidence.

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

`contracts/paper_task_manifest.csv` maps every paper task to one canonical
config and clinical packet. `contracts/paper_contract_manifest.csv` maps all 41
headline contracts to the 533 public leaderboard rows. Rich K25 audits are
separate under `configs/rich_action/`.

Post-freeze audit evidence is in `evidence/audits/`. KDD094 identifies limits in
target-policy provenance and OPE documentation; those limits take precedence
over earlier aggregate summaries.

## Quick Start

Python 3.11, 3.12, or 3.13 is required. Runtime dependencies are frozen in
`uv.lock`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

ehrdyn-icu validate-config --config-dir configs/tasks
ehrdyn-icu validate-schemas --schema-dir schemas
ehrdyn-icu generate-fixture --output /tmp/ehrdyn_fixture.csv
ehrdyn-icu evaluate \
  --fixture /tmp/ehrdyn_fixture.csv \
  --task-config configs/tasks/kdd2027_sepsis_vasopressor_3bin.json \
  --output /tmp/ehrdyn_metrics.json
ehrdyn-icu validate-manifest \
  --task-manifest contracts/paper_task_manifest.csv \
  --contract-manifest contracts/paper_contract_manifest.csv \
  --evidence evidence/core/contract_transition_leaderboard.csv
ehrdyn-icu validate-submission \
  --submission submission/leaderboard_submission_template.json \
  --config-dir configs/tasks
python -m unittest discover -s tests
ehrdyn-icu scan-release --root .
ehrdyn-icu verify-checksums --root .
```

The KDD212 public constructed-POMDP, repeated-OPE, transition entrant, and
manuscript-bundle workflow is documented in
[KDD212_PUBLIC_ENTRANT.md](KDD212_PUBLIC_ENTRANT.md). These synthetic and
aggregate-only interfaces remain separate from credentialed EHR reconstruction.

The historical `kdd2027` command remains available as a compatibility alias.

Submission documents are bound to their released Draft 2020-12 schemas; see
[SCHEMA_VALIDATION.md](SCHEMA_VALIDATION.md). Generated aggregate JSON uses the
portable byte contract in [CANONICAL_SERIALIZATION.md](CANONICAL_SERIALIZATION.md).

## Repository Contents

- `configs/tasks/`: five primary and two extended compact paper contracts.
- `configs/rich_action/`: separate K25/reference/exclusion audit contracts.
- `contracts/`: preprocessing and layered gate definitions.
- `credentialed/`: public SQL, preprocessing, action encoding, and aggregate parity targets for authorized local MIMIC-IV execution.
- `dictionaries/`: feature, action, reward, metric, and baseline provenance.
- `src/`: aggregate evaluator, fixture, split, gate, privacy, and submission code.
- `fixtures/`: schema-compatible synthetic data only.
- `evidence/`: publication-safe aggregate dynamics, temporal, and policy diagnostics.
- `clinical_review/`: seven aggregate-safe external-review packets with blank responses.
- `task_cards/`: human-readable task summaries.
- `schemas/` and `submission/`: leaderboard interfaces.
- `tests/`: clean-room, privacy, checksum, and contract tests.

## Credentialed Benchmark Execution

MIMIC-IV remains governed by PhysioNet credentialing and is not redistributed.
See [MIMIC_ACCESS.md](MIMIC_ACCESS.md). Extraction and preprocessing must run in
an authorized environment. `ehrdyn-icu evaluate-local` converts a local
cell-level prediction file to aggregate metrics without exporting row keys.
Only separately privacy-reviewed aggregate outputs may be contributed.
The complete author-side setup, bounded smoke, and credentialed commands are in
[PUBLIC_RECONSTRUCTION.md](PUBLIC_RECONSTRUCTION.md).

## Evidence Boundary

The dynamics track evaluates one-step and conditional recursive forecasts under
logged future actions. It does not simulate outcomes under a new treatment
policy. Existing policy evidence is quarantined as a frozen aggregate diagnostic
and is not accepted by the public leaderboard validator. See [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md)
and [ETHICS_AND_MISUSE.md](ETHICS_AND_MISUSE.md).

Unsupported uses include treatment recommendation, causal effect estimation,
counterfactual benefit claims, clinical deployment, and autonomous decisions.

## Citation

Citation metadata are in [CITATION.cff](CITATION.cff). Repository:
<https://github.com/cys1102/ehrdyn-icu>.
