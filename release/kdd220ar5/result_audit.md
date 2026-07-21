# KDD220AR5 result audit

## Scope

KDD220AR5 starts from public KDD220AR4 commit
`d8c99bd48c74049b16b3d4ddc5b7ca15fe73b521` on the isolated branch
`kdd220ar5-exact-current-lineage-port`. No MIMIC-IV source, credential,
restricted KDD152 interface, patient row, identifier, timestamp, checkpoint,
model output, policy output, OPE output, or expected clinical aggregate count
was opened.

All KDD220A through KDD220BR3 and KDD220C release receipts are inherited from
the base commit and remain byte-for-byte unchanged. The stopped KDD220BR3
decision remains stopped evidence and is not reinterpreted.

## Source closure

The source map pins the six requested authoritative commits and records file,
line, and symbol hashes. The port restores E060/KDD121 sepsis construction,
KDD121 non-diagnosis HF anchoring, KDD097 compact task eligibility and action
aggregation, KDD152-v2A recorded-action and SAFE-target filtering, KDD184 DBP
cleaning, and the named KDD201 first-decision dispositions.

The public source now distinguishes task-specific compact and large-lineage
eligibility. It does not apply a universal gender, discharge-time, hospice, or
diagnosis exclusion. Respiratory action observations use dedicated recorded
PEEP and FiO2 arrays rather than the SAFE forecasting mask.

## Synthetic evidence

The focused suite covers blood-only suspected infection, both E060 matching
directions, SOFA>=2 without anchor movement, HF without prior diagnosis,
nonsepsis overlap removal, role functions, recorded respiratory actions with
an unobserved SAFE FiO2 feature, any-SAFE target validity, KDD097 interval end
bins, KDD201 source-specific removal, and the full five-task fixture.

The synthetic fixture materializes all five tasks with K25/K25/K25/K4/K2 and
preserves stable task, episode, and transition order. Chunk sizes 2 and 17 and
CSV/CSV.GZ source forms produce identical semantic outputs; declared streaming
instrumentation remains configuration-specific.

## Verification status

Focused and full public tests, schema validation, checksum verification,
privacy scans, and clean-install results are recorded in
`source_schema_test_checksum_privacy_receipt.csv`. The 17-test focused AR4/AR5
suite and the complete 107-test public suite passed under Python 3.11. All six
released Draft 2020-12 schemas validated, including their positive and negative
fixture coverage. The finalized checksum manifest passed the public verifier,
and the release scan covered 467 files with zero privacy findings.

An isolated Python 3.11 environment installed the credentialed extra from the
candidate source, validated all schemas, ran the public privacy scanner, loaded
the frozen five-task/72-hour/250,000-row chunk contract, and completed the
synthetic five-task reconstruction smoke without undocumented intervention.
`compileall` and `git diff --check` also passed.

The failure ledger contains no unresolved item. Every required KDD220AR5
output is present, and no historical release path differs from the base commit.

## Claim boundary

This is a synthetic source-port result. It authorizes a separately frozen
author-side credentialed KDD220BR4 parity run from the exact pushed commit. It
does not prove real-EHR reconstruction, preserve or invalidate any scientific
metric by itself, authorize retraining, or provide independent reconstruction.
