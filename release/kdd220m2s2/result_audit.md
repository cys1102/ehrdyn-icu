# KDD220M2S2 exact source-equivalence audit

## Outcome

Decision: `stop_source_test_constructor_identity_or_privacy_failure`.

The exact pushed KDD220M2S source commit
`79deff45c5d4e5349e4ce648e212e6fbce1a27bd` was used. The constructor was
invoked exactly once against the KDD220M2L read-only single-encoding view and
completed successfully. The complete seven-file candidate was schema-validated,
privacy-scanned, and checksum-frozen before the KDD220M1 aggregate references
were opened.

Three scientific identities were exact:

- scientific surface: `e759eae39d308312bf9beeea0248dba20d580a7765e26b2dda2383a55ba299d9`;
- feature order: `2843f8071d1ef4df81db613af2ffc8510060ca0c604a30d94da352e84957167b`;
- role assignment: `da30bcd94094c9f249b0e4ff309336a69a9c6eca7a240604180894f8fa951f5f`.

All 100 task/role/count/action/reward/termination comparisons were exact. The
candidate schema and aggregate-only privacy boundary passed. However, the
seven-file path-plus-bytes tree was
`3ac30824efeeb1b3b46184e7e353b2b79f66c81b02eb47db21f6fae091e1bb25`,
not the required M1 identity
`8b6ecaab4ae4a42c387f3f34612f9cb0135b10e29bcd3a7e95b83056dca25d29`.
Runtime and stage telemetry contain execution-specific measurements and are at
least one demonstrated contributor to byte-level tree inequality. The prompt
requires all four identities to match, so the tree mismatch cannot be waived.

## Authorization boundary

KDD220M2R is not authorized because the exact candidate-tree identity gate did
not pass. KDD220M1R is not authorized because no scientific surface differed.
If a later separately authorized contract resolves the byte-level candidate
identity requirement without changing the scientific contract, its constructor
source identity must remain
`79deff45c5d4e5349e4ce648e212e6fbce1a27bd`.

No model training, OPE, scorer work, manuscript edit, tag, release, M2R, or M1R
was performed.
