# KDD220M2S result audit

## Outcome

Decision: `stop_source_test_constructor_identity_or_privacy_failure`.

The focused repair is reviewable and passed every pre-access software gate.
The credentialed constructor was invoked exactly once. It stopped at the
input-layout contract before scanning any high-volume table because a required
table resolved to more than one CSV/CSV.GZ encoding. The controlled-stop
receipt is schema-valid and aggregate-only. No rerun was attempted.

## Frozen identities

- Base source commit: `0331aa36e5b09824d5a1a04dc5b189976458ddaa`.
- Preserved KDD220M1 commit: `eaa08728b09fa150e5e23999f35766f44218d48b`.
- Patched source SHA-256: `08d65ba14a0ad7df9afd6805c4fa04ed3fadc888bb822c1f12ce77336fcbc020`.
- Runtime config SHA-256: `d731ea37e4cb47e20e76dc7d6de8160fe361da88ebf6d2385d2a50b096f84c89`.
- Aggregate schema SHA-256: `1f229de694f995b9a500270f3e411dab8bddfee46b15ac038bdb1a21b561f117`.
- Dependency lock SHA-256: `20fdf5d204803e8b94e3afaf70d9b43e15d2d3aed34f0c6ea09be48f2d64e223`.

## Verification

- Focused null-safe culture tests: 6/6 passed.
- Complete public unit suite: 127/127 passed under Python 3.11.15.
- Seven Draft 2020-12 schemas validated.
- `compileall` and `git diff --check` passed.
- Pre-access frozen MANIFEST: 513/513 entries passed.
- Final packaged MANIFEST: 525/525 entries passed.
- Constructor invocations: exactly one.
- Constructor result: controlled stop, `contract_error`.
- High-volume rows read: zero for every instrumented table.
- Frozen controlled-stop package: four files, path-plus-bytes SHA-256
  `4a02d7890d77570f3e2afff2a6983e7b802465f7bce2e0872acfa33d71536e96`.
- Privacy: aggregate-only; no identifiers, rows, timestamps, credentials, or
  private paths exported.

## Claim and authorization boundary

The tests support the null-safe deterministic behavior of the patch. They do
not establish equivalence with the KDD220M1 credentialed candidate. The M1
candidate, scientific-surface, feature-order, and role-assignment identities
were not compared because no complete repaired candidate existed. KDD220M2R
and KDD220M1R are both unauthorized by this stopped execution.

No commit, push, tag, release, model training, OPE, scorer work, or manuscript
edit was performed.
