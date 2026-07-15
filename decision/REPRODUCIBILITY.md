# Decision-release reproducibility

The unrestricted path validates the released contracts, aggregate evidence,
planner invariants, common-random-number null-response gate, and core OPE
formulas. It uses no MIMIC-IV rows.

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

The executed adaptive respiratory/shock/AKI/HF benchmark, historical
sensitivity benchmark, full OPE grid, P/R/T-component training, model-free
diagnostics, and AI-Clinician-aligned sepsis materialization runner/configuration
snapshots are under `reference_code/` with hashes in
`reference_code/source_manifest.csv`. They are provided for exact source
audit. They retain their frozen upstream module and aggregate-input
dependencies and are not represented as a one-command clean-room numerical
reproduction.

Full paper-number regeneration currently requires the frozen credentialed
`world-ehr` inputs and the reader-package builder described in
`manuscript/reproducibility.md`. This is a remaining release-engineering gap,
not an authorization to redistribute restricted arrays or patient-level data.

An independent credentialed reconstruction is complete only when another
MIMIC-credentialed user starts from a clean clone, executes the frozen local
pipeline inside an authorized environment, and matches the privacy-reviewed
aggregate targets. This is an artifact reproducibility check. It is not
external-data, clinical, or methodological-generalization validation.
