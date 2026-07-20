# KDD217AR3A result audit

## Outcome

The stage produced a public flat-file candidate, a strict aggregate receipt
schema, synthetic fixtures, and hash-gated pure-contract differential tooling.
No MIMIC data or credentials were accessed. The runtime does not depend on an
E060 manifest, author result tree, private path, patient manifest, checkpoint,
or split manifest.

Pure contracts for roles, valid horizons, cutpoints, K25 encoding,
disposition-specific KDD201 repair, sepsis terminal reward, and shock primary
dense reward matched the authoritative functions exactly. Four-task synthetic
task inclusion and post-KDD201 transition keys also matched.

The required full differential gate did not pass. Observation masks differed
for static, SIRS/SOFA, and ventilation channels, and shared floating values had
a maximum absolute difference of 1.0. The current large-lineage heart-failure
path and complete five-task action/reward/termination comparison were therefore
not accepted. These failures cannot be treated as credentialed-parity evidence.

## Verification

- `uv run python -m unittest discover -s tests -q`: 66 tests passed.
- Focused KDD217AR3A tests: 5 tests passed, including flat-file validation,
  variable horizons, action/reward timing, schema rejection, and deterministic
  aggregate receipt reconstruction.
- `python -m py_compile`: candidate module, differential runner, and focused
  tests compiled successfully.
- `git diff --check`: passed.
- Required-output parse audit: 18 required artifacts present, 13 CSV files
  parseable, and both JSON Schema documents parseable.
- Private-path scan: no workstation, temporary-worktree, PhysioNet data-root,
  ResearchWiki, or author result-tree path was present in the candidate or
  release receipts.

## Decision and Git disposition

Decision: `stop_dependency_closure_intermediate_regeneration_or_differential_failure`.

Because commit and push were authorized only on complete, the candidate and
receipts remain uncommitted in the isolated worktree. No final handoff tag was
created and KDD217AR2 remains unchanged.
