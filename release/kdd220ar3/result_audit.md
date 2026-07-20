# KDD220AR3 result audit

KDD220AR3 began from the exact pushed KDD220AR2 commit
`c752060608e70b991bac84343d6867a9dfaecc48`. Before editing, both stopped
aggregate-only receipt trees passed their independently named path-plus-bytes
and path-plus-per-file-digest identities, expected counts, anchor hashes, and
exact inventories. The stopped receipts were neither copied nor modified.

The prior public AKI implementation loaded the full labevents table, retained
nullable admission keys until integer conversion, and used one seven-day
minimum for both the 48-hour absolute-delta and seven-day ratio rules. The
repair filters required creatinine event fields before conversion, preserves
the row's admission association, uses stable admission-local ordering, and
maintains separate monotonic 48-hour and seven-day minima.

Chartevents, labevents, inputevents, outputevents, and procedureevents now use
required-column chunked readers with early item, key, and candidate-time-window
filters. Prescriptions and microbiology events use the same bounded pattern.
The default chunk size is frozen at 250,000 rows. Stable source order resolves
timestamp ties independently of chunk boundaries. Missing required keys or
times are counted and excluded; nonempty malformed values fail without row
content.

Synthetic tests cover both AKI windows, the old false-positive case, nullable
keys, malformed values, ties, multiple admissions, sparse chunks, two chunk
sizes, CSV/CSV.GZ parity, internal five-task interface equality, and exact
KDD220A aggregate receipt parity. The fixture peak RSS is implementation-only
evidence and does not satisfy the credentialed 64 GiB gate.

No MIMIC-IV, KDD152 reference, patient row, identifier, timestamp, restricted
array, model, checkpoint, policy output, OPE output, or hidden expected result
was accessed. KDD220AR3 changes no scientific task, interface, reward, split,
model, policy, estimator, or threshold and authorizes no scientific rerun.
