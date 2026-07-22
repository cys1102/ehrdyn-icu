# KDD220AR8 localized respiratory source-repair contract

## Frozen scope

KDD220AR8 is a code-only source-parity repair based on public commit
`f030c5793e4519a2b961eda1085d0bbeef29dbd0`. It preserves the five tasks,
subject roles, four-hour grid, 72-hour episode, separate 96-hour compact-lineage
raw buffer, 33-feature SAFE interface, action bins, rewards, termination,
continuation, censoring, preprocessing, and split logic. KDD220AR7 and the
stopped KDD220BR7 aggregate receipt remain immutable.

No MIMIC-IV input, credential, patient row, identifier, membership list,
row-level differential, checkpoint, model output, policy output, OPE output,
or restricted historical result was opened. The stopped BR7 aggregate counts
were not supplied to the constructor or used to choose the repair.

## Source-justified discrepancy

The pinned KDD097 `scan_icu_intervals` source constructs respiratory-support
events from the complete frozen oxygenation-support label pattern in both
`procedureevents` and positive `inputevents`. The AR7 public constructor used a
shortened procedure-only expression. The repair makes the already-public
`RESPIRATORY_PATTERN` authoritative for both tables and applies KDD097's
positive `max(amount, rate) > 0` input-event gate before ICU interval and
first-anchor selection.

The source order is unchanged after anchor construction: stay-bounded
candidate transitions; pre-repair observed PEEP and FiO2 action arrays;
train-only positive cutpoints; K25 encoding; original target/action membership;
KDD151 SAFE repair; and KDD201 disposition-3 first-decision removal.

## Acceptance gates

- The new input-support fixture fails on AR7 and passes after the repair.
- ICU start is inclusive and ICU outtime is exclusive; invalid stays remain
  excluded before anchors.
- Stable duplicate ordering, missing-action exclusion, K25 range, and the
  action/SAFE dual surface remain unchanged.
- The unchanged five-task fixture retains the AR6/AR7 scientific-surface hash.
- All existing tests, schemas, checksums, Python portability checks, clean
  installation, and privacy scans must pass.

## Claim boundary

Completion authorizes only an author-side KDD220BR8 reconstruction from the
exact reviewed candidate. It is not credentialed parity evidence, independent
reconstruction, external validation, a scientific rerun, or clinical evidence.
