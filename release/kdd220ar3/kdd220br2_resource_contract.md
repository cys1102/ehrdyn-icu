# KDD220BR2 resource contract

KDD220BR2 must start from the exact immutable KDD220AR3 commit and freeze its
source, runtime configuration, schema, public command, and single-encoding
MIMIC-IV v3.1 input view before construction.

- Default high-volume chunk size: 250,000 source rows.
- Design target: peak resident memory no greater than 32 GiB.
- Hard release gate: peak resident memory no greater than 64 GiB.
- Measurement: `/usr/bin/time -v` or a documented equivalent capturing peak
  RSS, wall time, CPU time, filesystem I/O, exit status, and worker count.
- Temporary disk: no copied raw source and no raw event spill; only local
  package/install state, final aggregate receipt, and aggregate audit sidecars.
- Input encoding: exactly one read-only CSV or CSV.GZ representation per table;
  a symlink view may select one existing encoding without copying source bytes.

The complete five-task constructor must finish before any KDD152 aggregate or
restricted parity reference is opened. Synthetic fixture memory does not
satisfy the real-data gate. A memory failure, constructor failure, or parity
mismatch must be preserved without changing chunk size, science contracts, or
tolerances after observing the outcome.
