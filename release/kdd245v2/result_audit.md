# KDD245V2 result audit

Decision: `not_executed_missing_canonical_v2_refresh`.

The software-only branch was opened from the immutable KDD220M1 commit, but no
scorer implementation or publication action was authorized. KDD220M2 ended in
`stop_refresh_source_training_evaluation_or_privacy_failure` before canonical-v2
interface freeze, model training, or historical-reference comparison. The
required exact agreement between the scorer contract and refreshed evidence
lineage therefore cannot be established.

- Remote `v2.0.0` collision check: pass; the tag was absent when checked.
- Existing `v1.3.0` and all earlier tags were not changed.
- `release/kdd245/` remains byte-identical with 14 files and tree digest
  `5734bf6a819d20655769b418ca683bd02645698571cfab3803a295bf7763084e`.
- No `release/kdd245r/` tree exists in the immutable M1 source. Its absence was
  recorded; no receipt was invented or reconstructed.
- No scorer code, schema, fixture, archive, tag, GitHub Release, DOI, MIMIC
  access, author-side execution, or independent-executor handoff was produced.

KDD245V2 may be rerun only after a separately repaired canonical-v2 constructor
supports a complete KDD220M2 decision and refreshed evidence identity.
