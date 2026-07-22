# KDD220M1 result audit

Decision: `complete_canonical_v2_lineage_freeze_with_bounded_refresh_scope`.

The public constructor identity is commit
`0331aa36e5b09824d5a1a04dc5b189976458ddaa`. The existing BR8R2 candidate was
reused without rerunning MIMIC: all seven files reproduce the frozen path-plus-bytes
digest `8b6ecaab4ae4a42c387f3f34612f9cb0135b10e29bcd3a7e95b83056dca25d29`,
and its schema, checksums, and privacy freeze remain valid. Historical BR8R2 and
KDD245 release trees were verified and left byte-identical.

The public source, runtime configuration, JSON Schema, dependency lock, Python
runtime, five task definitions, role functions, temporal contract, 33-feature SAFE
interface, action encoders, reward timing, termination, continuation, and aggregate
candidate are bound by the canonical manifest. Historical exact equality is not a
release gate.

All five tasks are affected. All-role candidate versus retained differences are:

- sepsis: -765 subjects, -968 episodes, -7,861 decisions;
- respiratory support: +12 subjects, +14 episodes, +87 decisions;
- shock: +10 subjects, +15 episodes, +168 decisions;
- AKI: +165 subjects, +262 episodes, +1,923 decisions; and
- heart failure: +448 subjects, +574 episodes, +5,308 decisions.

Action distributions also differ, and shape-dependent feature, mask, target,
recency, normalization, reward, termination, continuation, and ordering digests are
therefore treated as affected rather than inferred equivalent. The reused candidate
does not expose per-feature observation rates or numeric quantiles; this is a named
aggregate evidence gap routed to KDD220M2, not filled from restricted rows.

KDD220M2 must refresh every real-EHR cohort, prediction, uncertainty, behavior,
policy-diagnostic, and retrospective-OPE consumer. Constructed POMDP experiments are
explicitly excluded. KDD245V2 software-only work may begin against the frozen v2
contract, but no v2.0.0 release or manuscript edit is authorized by M1.

## Verification

- Draft 2020-12 schema self-validation and validation of the reused candidate:
  passed.
- Complete public unit suite with the locked credentialed extra: 121/121 passed.
- `compileall` over `src`, `scripts`, and `tests`: passed.
- Seven released JSON Schemas: passed.
- Exact checksum manifest after adding KDD220M1: 550/550 files passed.
- Aggregate privacy scan after adding KDD220M1: 551 files, zero findings.
- CSV rectangularity, JSON parsing, `git diff --check`, BR8R2 preservation
  digests, and KDD245 preservation digests: passed.
