# KDD220BR8 result audit

## Outcome

Decision: `stop_constructor_schema_privacy_or_runtime_failure`.

The exact AR8P-pushed commit was fetched into a fresh detached clean clone.
Source, configuration, schemas, dependency lock, Python environment, command,
MIMIC-IV v3.1 input contract, chunk size, and aggregate comparison axes were
frozen before execution. Public checksums passed for 512 files with zero
mismatches.

## Execution and stop

The constructor was invoked once. It exited with status 1 in 0.56 seconds and
peak RSS 78,848 KiB because the private candidate output directory already
existed. The frozen source checks this precondition before input-layout
validation; consequently no restricted input table was opened. Zero candidate
files were written. No retry, source edit, configuration change, tolerance
change, or in-stage repair was performed.

Because no candidate aggregate existed, schema validation, checksum freeze,
privacy scanning of candidate output, and respiratory-first parity were not
run. No authoritative reference for respiratory or any later task was opened.
Every scientific parity axis remains `not_run`; this stop is not a scientific
interface mismatch and authorizes no scientific rerun.

## Privacy and claim boundary

This release package contains aggregate execution facts only. It contains no
patient identifiers, timestamps, membership, rows, trajectories, arrays,
credentials, restricted source paths, or reference values. KDD220BR8 is
author-side only and does not establish independent reconstruction, external
validation, clinical utility, or generalization. KDD220C8 is not authorized.
