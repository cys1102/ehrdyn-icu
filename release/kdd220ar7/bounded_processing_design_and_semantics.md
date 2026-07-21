# Bounded processing design and semantic invariants

Each high-volume parser keeps one input chunk and one filtered chunk live. A
filtered chunk is written to a mode-0600 partition in a mode-0700 run
workspace; the constructor retains paths and aggregate row counts, not a list
of DataFrames. Materialization preallocates canonical numeric/datetime columns
as memory maps, reads one partition at a time, and applies the original stable
sort keys plus `__source_order`. The final event working set remains available
to unchanged downstream scientific logic without also retaining raw chunks,
concat copies, and sorted copies in RAM.

The past-only history builder reads one episode from the frozen source arrays
at a time. Role aggregation uses the already canonical contiguous episode
boundaries rather than materializing `list(groupby(...))`. Padded state,
target, mask, action, recency, imputation, normalization, reward, termination,
and continuation surfaces are private memory maps. Their existing aggregate
digests are computed before the task-local maps are closed and unlinked.

The run workspace includes a public-identity binding containing only release,
config hash, schema hash, and parser chunk size. Cross-run resume is not
supported. The entire workspace is deleted after completion or controlled
stop. No partition, map, path, row, identifier, timestamp, or clinical value
is copied into `release/`.

Scientific ordering, time windows, missingness, dual respiratory FiO2
surfaces, pre-filter train-only cutpoint fitting, action classes, KDD201 row
subsetting, rewards, and all canonical digests are unchanged. The stress probe
tests the former retained-frame design and the new spill design on identical
generated rows and requires identical canonical row digests.
