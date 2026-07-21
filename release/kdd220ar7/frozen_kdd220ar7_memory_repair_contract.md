# KDD220AR7 frozen bounded-memory repair contract

KDD220AR7 is a code-only public-constructor repair based on
`94e3362fcf20888f8744e435174d9f3292314b0b`. It did not access MIMIC-IV,
KDD152 row-level references, private reconstruction outputs, checkpoints, or
expected clinical counts. The aggregate-only stopped KDD220BR5 resource
receipt was used only to bind the observed 80.85 GiB failure.

## Frozen scientific surface

The five tasks, cohort and anchor logic, subject roles, splits, four-hour bins,
72-hour episode, feature and mask semantics, recency, train-only action
cutpoints, action classes, rewards, termination, continuation, ordering, and
canonical scientific digests remain those of KDD220AR6. The 32 GiB design
target and 64 GiB credentialed hard gate are unchanged. Performance logic may
not branch on expected clinical aggregates.

## Authorized implementation change

High-volume scans spill each filtered parser chunk to a private 0700 run
workspace instead of retaining a list of DataFrames. Canonical columns are
materialized through disk-backed arrays and stable `__source_order` sorting.
Large padded role surfaces and the past-only history tensors are private
memory maps released after each task. The workspace is bound to the frozen
release/config/schema/chunk contract, is not reusable across runs, and is
deleted on success or controlled stop.

Stage instrumentation is aggregate-only. It records stage, elapsed time,
rows read and retained, parser chunk size, temporary-disk high-water mark,
RSS at entry and exit, and process peak RSS. It prohibits paths, keys,
timestamps, values, rows, arrays, and patient membership.

## Acceptance boundary

Synthetic fixtures must preserve the frozen AR6 scientific hash
`f412a1c03d2325339543628384c4aad14dd0ffbf92160a4d51d11c4c42b750a3`.
Python 3.11 must also reproduce the AR6 aggregate-receipt bytes. Parser chunk,
compression, input order, and Python 3.11/3.12/3.13 must not change the
scientific hash. The generated one-million-row stress workload must preserve
the canonical row digest and use less peak RSS than the AR6 retained-frame
design. These synthetic checks do not prove the real MIMIC memory gate;
KDD220BR6 remains required.
