# KDD220AR4 frozen respiratory missing-action contract

Frozen before implementation on 2026-07-20 from public commit
`78a823fc428a0d7b241242f874e4926e3ad65417`.

## Preserved scientific surface

- Tasks remain sepsis, respiratory support, shock, AKI, and heart failure.
- Subject definitions, role assignment, splits, support rules, normalization,
  censoring, rewards, termination, state targets, and cohort rules do not
  change.
- Decisions remain four-hour bins in a 72-hour, 18-bin interface.
- Respiratory action remains observed PEEP-5 x observed FiO2-5, K=25.
- Missing PEEP or FiO2 makes that respiratory transition invalid. It is never
  encoded as class 0, carried forward, taken from the future, or filled from a
  state-imputation value.
- PEEP and FiO2 cutpoints use positive, finite, observed train-role settings
  only and are then applied unchanged to all roles.
- The high-volume default chunk size remains 250,000 rows.

## Authorized code repair

Candidate transitions are encoded against explicit PEEP and FiO2 observation
masks. Missing-action transitions are removed before task aggregation. An
episode remains if at least one transition remains, and task counts, horizons,
and action counts are recomputed from retained transitions. Any retained class
outside `[0, 24]` fails closed. No non-respiratory filtering changes are
authorized.

High-volume scan instrumentation becomes aggregate output. For each documented
table it reports the sum of rows read and retained across deterministic public
scan passes, scan and chunk counts, maximum retained rows in one chunk,
effective chunk size, and `csv` or `csv_gz` encoding. It exports no row,
identifier, timestamp, filename, or private path. Controlled stops preserve
only allowlisted aggregate runtime and scan diagnostics when available.

KDD220AR4 is code-only. It does not access MIMIC-IV, KDD152 aggregate
references, restricted arrays, trained artifacts, policies, OPE outputs, or
historical expected counts, and it cannot authorize a scientific rerun.
