# Frozen KDD220M2I canonical scientific-identity contract

This contract is frozen before either seven-file candidate is opened for
content comparison. It does not change a scientific gate or authorize a
constructor invocation.

## Inventory and tree construction

Each input must contain exactly these seven regular files and no others:

1. `aggregate_receipt.json`
2. `icu_time_order_eligibility_aggregate.csv`
3. `nullable_key_and_timestamp_exclusion_aggregate.csv`
4. `respiratory_action_filter_aggregate.csv`
5. `runtime_resource_aggregate.json`
6. `stage_resource_instrumentation_aggregate.json`
7. `streaming_instrumentation_aggregate.csv`

Raw and canonical tree hashes update SHA-256 with each sorted POSIX relative
path encoded as UTF-8 followed immediately by the corresponding raw or
canonical bytes. Canonical JSON uses sorted keys, compact separators, UTF-8,
and one trailing newline. Canonical CSV preserves the frozen header and row
order and uses LF line endings.

The canonicalizer must reject unknown or missing files, extra or missing
fields, schema changes, duplicate streaming table rows, altered inventory or
row order, and any difference outside the allowlist below.

## Exact allowlist

### `runtime_resource_aggregate.json`

Only `/wall_seconds`, `/maximum_resident_set_size_kib`, and
`/temporary_disk_bytes` are replaced by the fixed string sentinel
`__KDD220M2I_EXECUTION_RESOURCE__`. All three fields must exist. `status` and
the complete object structure remain exact.

### `stage_resource_instrumentation_aggregate.json`

Within every `/stages/*` object, only `elapsed_seconds`,
`temporary_disk_high_water_bytes`, `rss_entry_kib`, `rss_exit_kib`, and
`peak_rss_kib` are replaced by the same fixed sentinel. All fields must exist.
Top-level schema version, status, privacy, stage inventory and order, stage
names, rows read, rows retained, and chunk size remain exact.

### `streaming_instrumentation_aggregate.csv`

Only `compression_encoding` is replaced by the fixed string sentinel
`__KDD220M2I_EQUIVALENT_ENCODING__`. This is permitted only when the frozen
KDD220M2L receipt has decision
`complete_single_encoding_view_ready_for_kdd220m2s2` and proves all 14
plain-versus-decompressed streams equal with valid required headers. Table,
rows read, rows retained, chunks processed, maximum retained rows per chunk,
effective chunk size, scan count, row order, and table inventory remain exact.

### `aggregate_receipt.json`

Only `compression_encoding` inside each `/streaming/*` row is replaced by
`__KDD220M2I_EQUIVALENT_ENCODING__`, under the same KDD220M2L evidence gate.
The entire remaining receipt is exact, including runtime software versions,
source hashes, tasks, roles, contracts, scientific hashes, counts, cutpoints,
support and component digests, transition stages, rewards, termination,
privacy, and every other value.

### Remaining CSV files

`icu_time_order_eligibility_aggregate.csv`,
`nullable_key_and_timestamp_exclusion_aggregate.csv`, and
`respiratory_action_filter_aggregate.csv` require raw path-plus-bytes equality.

No difference may be excluded merely because it differs. Unknown or
unclassified differences are terminal failures.
