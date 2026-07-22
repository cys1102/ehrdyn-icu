# KDD220M2I canonical scientific-identity audit

## Decision

`authorize_kdd220m2r_scientific_and_canonical_identity_exact`

No constructor was invoked and no MIMIC table or clinical row was accessed.
The exact frozen KDD220M1 and KDD220M2S2 seven-file aggregate candidates were
opened read-only after the canonical comparison contract was frozen.

Both required raw path-plus-bytes tree hashes were reproduced:

- KDD220M1: `8b6ecaab4ae4a42c387f3f34612f9cb0135b10e29bcd3a7e95b83056dca25d29`;
- KDD220M2S2: `3ac30824efeeb1b3b46184e7e353b2b79f66c81b02eb47db21f6fae091e1bb25`.

Five files were already raw byte-exact. The 98 field-level differences were
confined to the frozen runtime/stage resource allowlist; no compression field
actually differed and no scientific or unclassified difference was found.
After deterministic sentinel replacement, all seven per-file canonical hashes
and both canonical trees were exact. The shared canonical tree is
`aa39bf00f30efd07362473ac8ac7074924223fdd1741d5fc9e6bcd33621d9572`.

The scientific surface, feature order, role assignment, raw aggregate receipt,
three fixed aggregate CSV files, and all 100 KDD220M2S2 scientific comparisons
remain exact. Aggregate and stage schemas and privacy boundaries pass.

KDD220M2R is authorized only from source commit
`79deff45c5d4e5349e4ce648e212e6fbce1a27bd`. KDD220M1R remains unauthorized
because no scientific surface differs. This stage did not execute either run,
train a model, compute OPE, edit the manuscript, or create a release.
