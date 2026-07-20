# Static versus computed hash contract

Static source artifacts use SHA-256 over their released bytes. KDD212 remains recoverable at its immutable commit; KDD214 schema bytes changed only for the schema-binding repair. They were not reformatted to manufacture computed parity. Current schema hashes:

- `aggregate_metrics.schema.json`: `55ee319aca4345b96d91f41bfe26866d06d39b012d0cf05e981d5d457955512d`
- `leaderboard_submission.schema.json`: `7dfe21da05e4c5d96402baa92da7dd3affc9aee030e00f288f6f6d2a230b8249`
- `transition_submission.schema.json`: `c8a734b17847ea1f45672cde0192914ca134d038a1e8655c421a81db0feee80b`

Computed aggregate outputs use the canonical writer and are labeled `computed_canonical_sha256`; the frozen probe hash is `b11682c9ba0e4b0638670f2dc6c52af4e910a74f271abaa0528250d65f691592`. Static and computed hashes are never pooled or reinterpreted.
